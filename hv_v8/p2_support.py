from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import ContractError, lane_uid, lane_uid_str, read_json, read_jsonl, require, sha256_bytes, write_json, write_jsonl
from .p1_native_audit import (
    DEFAULT,
    apply_rift_config,
    carla_transform,
    cumulative_route,
    import_carla_and_grp,
    is_within_distance_ahead,
    preview_waypoint,
    route_waypoints,
    distance2,
)


def local_xy(ego_transform: Any, loc: Any) -> tuple[float, float]:
    yaw = math.radians(float(ego_transform.rotation.yaw))
    dx = float(loc.x - ego_transform.location.x)
    dy = float(loc.y - ego_transform.location.y)
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def d_interval_for_box(cx: float, cy: float, yaw_rad: float, lane_width: float,
                       slot: int, half_width: float = 0.0) -> tuple[float, float] | None:
    # Model parameter d is right-positive relative to the lane tangent.
    # World right-normal = (-sin(theta), cos(theta)).
    nx, ny = -math.sin(yaw_rad), math.cos(yaw_rad)
    lo, hi = -0.5 * lane_width, 0.5 * lane_width

    if slot == 0:
        xmin, xmax = 1e-6, 32.0
    elif slot == 1:
        xmin, xmax = -32.0, 32.0
    else:
        raise ValueError(slot)
    ymin, ymax = -32.0, 32.0

    for a, b, lower, upper in [
        (cx, nx, xmin + half_width, xmax - half_width),
        (cy, ny, ymin + half_width, ymax - half_width),
    ]:
        if abs(b) < 1e-9:
            if not (lower <= a <= upper):
                return None
            continue
        d1 = (lower - a) / b
        d2 = (upper - a) / b
        lo = max(lo, min(d1, d2))
        hi = min(hi, max(d1, d2))
        if lo > hi:
            return None
    return (lo, hi)


def path_lane_uids(trace: list[Any]) -> list[list[int]]:
    seen = []
    for wp, _ in trace:
        uid = [int(wp.road_id), int(wp.section_id), int(wp.lane_id)]
        if not seen or seen[-1] != uid:
            seen.append(uid)
    return seen


class SupportCompiler:
    """
    Clean-room P2 compiler.

    Input to the compiler is only:
      * CARLA map
      * ego current transform
      * ego future route
      * output slot definition

    It never receives GT actor state. The RIFT route planner is used only to
    determine whether a directed map approach can reach an ego-route anchor.
    This creates the model's structured support; it is not a post-hoc repair.
    """

    def __init__(self, carla: Any, grp: Any, carla_map: Any, scene: dict[str, Any],
                 sample_m: float, map_radius_m: float):
        self.carla = carla
        self.grp = grp
        self.map = carla_map
        self.scene = scene
        self.sample_m = sample_m
        self.map_radius_m = map_radius_m
        self.ego = carla_transform(carla, scene["ego"])
        self.route, self.route_checks = route_waypoints(carla_map, scene)
        self.ego_dist = cumulative_route(self.route)
        self.trace_cache: dict[tuple, tuple] = {}

    def trace(self, origin_wp: Any, target_idx: int):
        target = self.route[target_idx]
        key = (
            int(origin_wp.road_id), int(origin_wp.section_id), int(origin_wp.lane_id),
            round(float(origin_wp.s), 2), int(target_idx)
        )
        if key not in self.trace_cache:
            try:
                self.trace_cache[key] = self.grp.trace_route(
                    origin_wp.transform.location, target.transform.location
                )
            except Exception:
                self.trace_cache[key] = ([], {"road_ids": [], "lane_ids": []}, math.inf)
        return self.trace_cache[key]

    def compile(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        samples = self.map.generate_waypoints(self.sample_m)
        route_indices = list(self.ego_dist.keys())
        groups: dict[tuple, dict[str, Any]] = {}

        considered = 0
        local_domain_rejected = 0
        reachable = 0

        for wp in samples:
            if wp.lane_type != self.carla.LaneType.Driving:
                continue
            if distance2(wp.transform.location, self.ego.location) > self.map_radius_m:
                continue
            considered += 1
            cx, cy = local_xy(self.ego, wp.transform.location)
            yaw = math.radians(float(wp.transform.rotation.yaw))
            slot_domains = {}
            for slot in (0, 1):
                di = d_interval_for_box(cx, cy, yaw, float(wp.lane_width), slot)
                if di is not None:
                    slot_domains[slot] = di
            if not slot_domains:
                local_domain_rejected += 1
                continue

            # P2 is deliberately a support-domain compiler, not a native
            # classifier. The approach exists if the directed CARLA/RIFT
            # route can reach an ego-route anchor. FOV and route-distance
            # compatibility are measured below but are NOT used to delete
            # support. Otherwise P2 would reproduce the V5/V6 candidate leak.
            for ridx in route_indices:
                target = self.route[ridx]
                trace, route_ids, cbv_dist = self.trace(wp, ridx)
                if not trace or not math.isfinite(float(cbv_dist)):
                    continue
                diff = abs(float(cbv_dist) - float(self.ego_dist[ridx]))
                reachable += 1
                lanes = path_lane_uids(trace)
                key = (
                    int(wp.road_id), int(wp.section_id), int(wp.lane_id), int(ridx),
                    tuple(tuple(x) for x in lanes)
                )
                g = groups.setdefault(key, {
                    "lane_uid": [int(wp.road_id), int(wp.section_id), int(wp.lane_id)],
                    "anchor_route_index": int(ridx),
                    "anchor_ego_distance_m": float(self.ego_dist[ridx]),
                    "path_lane_uids": lanes,
                    "s_min": float(wp.s),
                    "s_max": float(wp.s),
                    "sample_count": 0,
                    "local_x_min": cx, "local_x_max": cx,
                    "local_y_min": cy, "local_y_max": cy,
                    "slot_d_intervals_by_s": {
                        str(k): [[float(wp.s), float(v[0]), float(v[1])]]
                        for k, v in slot_domains.items()
                    },
                    "centerline_samples_local": [[float(wp.s), float(cx), float(cy)]],
                    "route_distance_diffs_m": [],
                })
                g["s_min"] = min(g["s_min"], float(wp.s))
                g["s_max"] = max(g["s_max"], float(wp.s))
                g["sample_count"] += 1
                g["local_x_min"] = min(g["local_x_min"], cx)
                g["local_x_max"] = max(g["local_x_max"], cx)
                g["local_y_min"] = min(g["local_y_min"], cy)
                g["local_y_max"] = max(g["local_y_max"], cy)
                g["route_distance_diffs_m"].append(float(diff))
                g["centerline_samples_local"].append([float(wp.s), float(cx), float(cy)])
                for sk, new in slot_domains.items():
                    g["slot_d_intervals_by_s"].setdefault(str(sk), []).append(
                        [float(wp.s), float(new[0]), float(new[1])]
                    )

        approaches = []
        for g in groups.values():
            if g["sample_count"] <= 0 or not g["slot_d_intervals_by_s"]:
                continue
            raw_id = json.dumps({
                "lane_uid": g["lane_uid"],
                "anchor_route_index": g["anchor_route_index"],
                "path_lane_uids": g["path_lane_uids"],
                "s_min": round(g["s_min"], 3),
                "s_max": round(g["s_max"], 3),
            }, sort_keys=True, separators=(",", ":")).encode()
            g["approach_id"] = "a_" + hashlib.sha1(raw_id).hexdigest()[:16]
            for sk in g["slot_d_intervals_by_s"]:
                g["slot_d_intervals_by_s"][sk].sort(key=lambda x: x[0])
            g["support_semantics"] = "map_only_directed_reachability"
            g["native_certified"] = False
            approaches.append(g)

        approaches.sort(key=lambda x: x["approach_id"])
        report = {
            "schema_version": "hv_v8_p2_support_v1",
            "record_index": self.scene["record_index"],
            "scene_id": self.scene["scene_id"],
            "sample_m": self.sample_m,
            "map_radius_m": self.map_radius_m,
            "approach_count": len(approaches),
            "sampled_driving_waypoints": len(samples),
            "considered_waypoints": considered,
            "local_domain_rejected_waypoints": local_domain_rejected,
            "successful_map_route_checks": reachable,
            "approaches": approaches,
            "route_checks": self.route_checks,
            "forbidden_fallbacks_used": [],
        }
        return approaches, report


def target_support_audit(carla: Any, grp: Any, carla_map: Any, scene: dict[str, Any],
                         approaches: list[dict[str, Any]]) -> dict[str, Any]:
    route, _ = route_waypoints(carla_map, scene)
    ego_tr = carla_transform(carla, scene["ego"])
    target_ids = set(scene.get("target_actor_ids", []))
    if not target_ids:
        target_ids = {a.get("actor_id") for a in scene.get("actors", []) if a.get("is_target")}

    by_key = defaultdict(list)
    for a in approaches:
        by_key[(
            tuple(a["lane_uid"]),
            int(a["anchor_route_index"]),
        )].append(a)

    rows = []
    for actor in [a for a in scene.get("actors", []) if a.get("actor_id") in target_ids]:
        tr = carla_transform(carla, actor)
        pwp = preview_waypoint(carla, carla_map, tr)
        distance = float(ego_tr.location.distance(tr.location))
        candidate_filter = {
            "distance_m": distance,
            "distance_pass": DEFAULT["min_distance_m"] < distance <= DEFAULT["search_radius_m"],
            "preview_exists": pwp is not None,
            "junction_heading_pass": True,
        }
        if pwp is not None and pwp.is_junction:
            candidate_filter["junction_heading_error_deg"] = float(
                tr.get_forward_vector().get_vector_angle(pwp.transform.get_forward_vector())
            )
            candidate_filter["junction_heading_pass"] = candidate_filter["junction_heading_error_deg"] <= 20.0
        candidate_filter["candidate_filter_pass"] = (
            candidate_filter["distance_pass"]
            and candidate_filter["preview_exists"]
            and candidate_filter["junction_heading_pass"]
        )
        exact = {
            "actor_id": actor.get("actor_id"),
            "candidate_filter": candidate_filter,
            "preview_exists": pwp is not None,
            "preview_lane_uid": None,
            "native_interaction": False,
            "native_positive": False,
            "selected_anchor": None,
            "mapped_to_approach": False,
            "failure_stage": None,
        }
        if pwp is None:
            exact["failure_stage"] = "no_preview_waypoint"
            rows.append(exact)
            continue
        exact["preview_lane_uid"] = [int(pwp.road_id), int(pwp.section_id), int(pwp.lane_id)]
        # Store the chart coordinates of the raw vehicle center relative to
        # the preview lane. This is diagnostic only; no GT value is used by
        # the map-only compiler.
        f = pwp.transform.get_forward_vector().make_unit_vector()
        r = carla.Vector3D(-f.y, f.x, 0.0)
        # CARLA's local +y is right; for heading theta the right normal is
        # (-sin(theta), cos(theta)).
        delta = tr.location - pwp.transform.location
        exact["raw_to_preview_lane"] = {
            "lateral_d_m": float(delta.dot(r)),
            "longitudinal_m": float(delta.dot(f)),
            "heading_residual_deg": float(((float(tr.rotation.yaw) - float(pwp.transform.rotation.yaw) + 180) % 360) - 180),
            "lane_half_width_m": float(pwp.lane_width) / 2.0,
        }

        # Reuse the exact RIFT interaction test, but never feed its result back
        # into the compiler.
        ego_dist = cumulative_route(route)
        best = None
        for idx, ed in ego_dist.items():
            target = route[idx]
            if not is_within_distance_ahead(
                target.transform, pwp.transform,
                DEFAULT["max_interaction_dis_m"],
                DEFAULT["max_interaction_fov_deg"] / 2.0,
            ):
                continue
            trace, route_ids, cbv_dist = grp.trace_route(
                pwp.transform.location, target.transform.location
            )
            if not trace:
                continue
            diff = abs(float(cbv_dist) - float(ed))
            if best is None or diff < best["distance_diff_m"]:
                best = {
                    "anchor_route_index": int(idx),
                    "distance_diff_m": float(diff),
                    "cbv_distance_m": float(cbv_dist),
                    "ego_distance_m": float(ed),
                    "path_lane_uids": path_lane_uids(trace),
                }
        if best is None or best["distance_diff_m"] >= DEFAULT["min_interaction_dis_diff_m"]:
            exact["failure_stage"] = "no_native_interaction"
            rows.append(exact)
            continue
        exact["native_interaction"] = True
        exact["native_positive"] = bool(candidate_filter["candidate_filter_pass"])
        exact["selected_anchor"] = best

        if not exact["native_positive"]:
            exact["failure_stage"] = "interaction_positive_but_rift_candidate_filter_failed"
            rows.append(exact)
            continue

        candidates = by_key.get((
            tuple(exact["preview_lane_uid"]),
            int(best["anchor_route_index"]),
        ), [])
        best_path = best.get("path_lane_uids", [])
        for a in candidates:
            if a.get("path_lane_uids") != best_path:
                continue
            if float(a["s_min"]) - 1.5 <= float(pwp.s) <= float(a["s_max"])+1.5:
                exact["mapped_to_approach"] = True
                exact["approach_id"] = a["approach_id"]
                exact["mapped_s_m"] = float(pwp.s)
                exact["failure_stage"] = None
                break
        if not exact["mapped_to_approach"]:
            same_lane = [
                a for a in approaches
                if tuple(a["lane_uid"]) == tuple(exact["preview_lane_uid"])
            ]
            same_anchor = [
                a for a in same_lane
                if int(a["anchor_route_index"]) == int(best["anchor_route_index"])
            ]
            same_path = [
                a for a in same_anchor
                if a.get("path_lane_uids") == best.get("path_lane_uids", [])
            ]
            if not same_lane:
                stage = "approach_lane_missing"
            elif not same_anchor:
                stage = "approach_anchor_missing"
            elif not same_path:
                stage = "approach_path_variant_missing"
            else:
                stage = "approach_s_interval_missing"
            exact["failure_stage"] = stage
        rows.append(exact)

    total = len(rows)
    native = sum(r["native_positive"] for r in rows)
    mapped = sum(r["mapped_to_approach"] for r in rows)
    failure_counts = defaultdict(int)
    for r in rows:
        if r.get("failure_stage"):
            failure_counts[r["failure_stage"]] += 1
    return {
        "schema_version": "hv_v8_p2_gt_audit_v1",
        "record_index": scene["record_index"],
        "scene_id": scene["scene_id"],
        "target_rows": total,
        "native_interaction_rows": native,
        "mapped_rows": mapped,
        "support_recall_over_native_rows": (mapped / native if native else None),
        "failure_counts": dict(failure_counts),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="V8 clean-room P2 support compiler and GT audit")
    p.add_argument("--manifest", required=True, help="JSONL with record_index, scene_id, snapshot_path")
    p.add_argument("--rift-root", required=True)
    p.add_argument("--rift-recognition-config")
    p.add_argument("--rift-mode", default="eval")
    p.add_argument("--carla-pythonapi")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=31120)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-m", type=float, default=1.0)
    p.add_argument("--map-radius-m", type=float, default=140.0)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    try:
        apply_rift_config(args.rift_recognition_config, args.rift_mode)
        require(args.sample_m > 0, "--sample-m must be >0")
        require(args.map_radius_m >= DEFAULT["max_interaction_dis_m"],
                "--map-radius-m must cover the native interaction radius")
        carla, grp_cls = import_carla_and_grp(args.rift_root, args.carla_pythonapi)
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        rows = read_jsonl(Path(args.manifest))
        require(rows, "P2 manifest is empty")
        if args.limit:
            rows = rows[:args.limit]
        out = Path(args.output).resolve()
        require(not out.exists(), f"Refuse overwrite: {out}")
        out.mkdir(parents=True)

        current = client.get_world()
        require(len(current.get_actors().filter("vehicle.*")) == 0,
                "Refuse P2 on non-empty CARLA world; use a dedicated server")

        summaries = []
        for row in rows:
            scene = read_json(Path(row["snapshot_path"]))
            world = client.load_world(str(scene.get("town") or scene.get("map_name")))
            cmap = world.get_map()
            grp = grp_cls(cmap, 2.0)
            compiler = SupportCompiler(carla, grp, cmap, scene, args.sample_m, args.map_radius_m)
            approaches, report = compiler.compile()
            gt = target_support_audit(carla, grp, cmap, scene, approaches)
            report["gt_audit"] = gt
            write_json(out / f"record_{int(row['record_index']):04d}.json", report)
            summaries.append({
                "record_index": row["record_index"],
                "scene_id": row["scene_id"],
                "approach_count": len(approaches),
                "native_rows": gt["native_interaction_rows"],
                "mapped_rows": gt["mapped_rows"],
                "support_recall": gt["support_recall_over_native_rows"],
                "failure_counts": gt["failure_counts"],
            })

        native = sum(x["native_rows"] for x in summaries)
        mapped = sum(x["mapped_rows"] for x in summaries)
        out_summary = {
            "schema_version": "hv_v8_p2_summary_v1",
            "records": len(summaries),
            "native_rows": native,
            "mapped_rows": mapped,
            "support_recall": (mapped / native if native else None),
            "all_native_rows_mapped": native == mapped,
            "forbidden_fallbacks_used": [],
            "stop_rule": "Do not train if native-positive GT is unmapped without an explicit non-model cause.",
            "records": summaries,
        }
        write_json(out / "summary.json", out_summary)
        print(json.dumps(out_summary, ensure_ascii=False, indent=2))
        raise SystemExit(0 if out_summary["all_native_rows_mapped"] else 3)
    except (ContractError, FileNotFoundError, ImportError) as e:
        print(f"P2_FAIL {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
