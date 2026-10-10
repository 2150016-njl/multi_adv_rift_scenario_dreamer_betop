"""Reconstruct target-to-conflict RIFT paths for existing labeled snapshots.

This is an offline label-enrichment pass, not scenario collection and not an
online generation component.  It loads each CARLA Town map, applies RIFT's
own ``GlobalRoutePlanner.trace_route`` from the saved actor pose to the saved
RIFT interaction waypoint, and writes a sidecar JSONL.  Raw snapshots remain
untouched.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import carla

from rift.scenario.tools.global_route_planner import GlobalRoutePlanner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument("--sampling_resolution", type=float, default=1.0)
    parser.add_argument(
        "--max_label_distance_abs_error_m", type=float, default=2.0,
        help=("Reject a reconstructed path when its RIFT planner distance differs "
              "from the saved RIFT label by more than this tolerance."),
    )
    parser.add_argument("--max_input_files", type=int, default=-1)
    parser.add_argument(
        "--target_manifest", type=Path, default=None,
        help=("Optional rift_route_target_manifest_v1 JSONL. When given, trace only the two "
              "targets selected by the V4C dataset build."),
    )
    parser.add_argument(
        "--require_complete_target_manifest", action="store_true", default=False,
        help=("Exit non-zero after writing diagnostics unless every manifest target has one "
              "accepted trace_route sidecar entry. Use this for V4C training data."),
    )
    return parser.parse_args()


def load_scene(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("scene", payload) if isinstance(payload, dict) else {}


def load_target_manifest(path):
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    selected = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                key = str(item["source_file_sha1"])
                actor_ids = {int(value) for value in item["target_actor_ids"]}
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("invalid target manifest at line %d: %s" % (line_number, exc))
            if len(actor_ids) != 2 or key in selected:
                raise ValueError("manifest entry must have one source hash and exactly two actor IDs at line %d" % line_number)
            selected[key] = actor_ids
    return selected


def point_dict(waypoint):
    location = waypoint.transform.location
    return {"x": float(location.x), "y": float(location.y), "z": float(location.z)}


def preview_waypoint(world_map, actor):
    """Recreate BaseCBVRecog.get_preview_waypoint() for a saved raw actor."""
    transform = carla.Transform(
        carla.Location(x=float(actor["x"]), y=float(actor["y"]), z=float(actor.get("z", 0.0))),
        carla.Rotation(
            pitch=float(actor.get("pitch_deg", 0.0)),
            yaw=float(actor["yaw_deg"]),
            roll=float(actor.get("roll_deg", 0.0)),
        ),
    )
    forward = transform.get_forward_vector().make_unit_vector()
    preview = carla.Location(
        x=transform.location.x + 2.0 * forward.x,
        y=transform.location.y + 2.0 * forward.y,
        z=transform.location.z + 2.0 * forward.z,
    )
    return world_map.get_waypoint(preview, project_to_road=None, lane_type=carla.LaneType.Driving)


def conflict_waypoint(world_map, waypoint):
    """Resolve the serialized ego-route conflict point back to a driving waypoint."""
    location = carla.Location(
        x=float(waypoint["x"]), y=float(waypoint["y"]), z=float(waypoint.get("z", 0.0)),
    )
    return world_map.get_waypoint(location, project_to_road=None, lane_type=carla.LaneType.Driving)


def map_matches_town(world, town):
    name = str(world.get_map().name or "")
    return name.rsplit("/", 1)[-1] == str(town)


def load_planner(client, town, resolution):
    world = client.get_world()
    if not map_matches_town(world, town):
        world = client.load_world(str(town))
    return world, GlobalRoutePlanner(world.get_map(), float(resolution))


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite route sidecar directory: %s" % args.output_dir)
    paths = sorted(args.input_dir.rglob("*.json"))
    if args.max_input_files > 0:
        paths = paths[:args.max_input_files]
    if not paths:
        raise ValueError("no JSON snapshots found")

    target_manifest = load_target_manifest(args.target_manifest)
    expected_manifest_targets = {
        (source_key, int(actor_id))
        for source_key, actor_ids in (target_manifest or {}).items()
        for actor_id in actor_ids
    }
    written_manifest_targets = set()
    # Keep a target-level ledger as well as aggregate counters.  Aggregate
    # counters cannot identify which selected training labels failed route
    # reconstruction, which makes an incomplete sidecar impossible to repair
    # or deliberately exclude without guessing.
    target_outcomes = {
        (source_key, actor_id): {
            "source_file_sha1": source_key,
            "actor_id": int(actor_id),
            "status": "source_snapshot_not_found",
        }
        for source_key, actor_id in expected_manifest_targets
    }
    # Grouping avoids rebuilding CARLA's route graph for every snapshot while
    # keeping the source-file hash identical to the dataset builder's key.
    jobs_by_town = defaultdict(list)
    counters = Counter()
    for path in paths:
        counters["processed_files"] += 1
        try:
            scene = load_scene(path)
            town = scene.get("town")
            actors = {int(actor["actor_id"]): actor for actor in (scene.get("agents") or []) if "actor_id" in actor}
            labels = scene.get("rift_labels") or []
        except (OSError, ValueError, TypeError, KeyError):
            counters["invalid_snapshot"] += 1
            continue
        if not town or not actors or not isinstance(labels, list):
            counters["not_rift_labeled_snapshot"] += 1
            continue
        source_key = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
        required_actor_ids = None if target_manifest is None else target_manifest.get(source_key)
        if target_manifest is not None and required_actor_ids is None:
            counters["snapshot_not_selected_by_manifest"] += 1
            continue
        selected = []
        labels_by_actor_id = {}
        for label in labels:
            try:
                labels_by_actor_id[int(label["actor_id"])] = label
            except (KeyError, TypeError, ValueError):
                continue
            if not bool((label or {}).get("is_interactive", False)):
                continue
            waypoint = (label or {}).get("interaction_waypoint")
            try:
                actor = actors[int(label["actor_id"])]
                float(waypoint["x"]); float(waypoint["y"])
            except (KeyError, TypeError, ValueError):
                counters["interactive_label_missing_actor_or_waypoint"] += 1
                continue
            if required_actor_ids is None or int(label["actor_id"]) in required_actor_ids:
                selected.append((actor, label))
        if required_actor_ids is not None:
            for actor_id in required_actor_ids:
                outcome = target_outcomes[(source_key, int(actor_id))]
                outcome.update({"town": str(town), "scene_id": scene_id_from_path(path)})
                actor = actors.get(int(actor_id))
                label = labels_by_actor_id.get(int(actor_id))
                if actor is None:
                    outcome["status"] = "actor_missing"
                elif not bool((label or {}).get("is_interactive", False)):
                    outcome["status"] = "target_label_not_interactive_or_missing"
                else:
                    waypoint = (label or {}).get("interaction_waypoint")
                    try:
                        float(waypoint["x"]); float(waypoint["y"])
                    except (KeyError, TypeError, ValueError):
                        outcome["status"] = "interaction_waypoint_missing"
                    else:
                        outcome["status"] = "pending_route_trace"
        if selected:
            if required_actor_ids is not None:
                found = {int(actor["actor_id"]) for actor, _label in selected}
                missing = required_actor_ids - found
                if missing:
                    counters["manifest_target_missing_or_not_interactive"] += len(missing)
            jobs_by_town[str(town)].append((path, selected))
            counters["source_snapshots_with_interactive_labels"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_jsonl = args.output_dir / "route_interaction_sidecar.jsonl"
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(float(args.timeout_s))
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for town in sorted(jobs_by_town):
            counters["town_map_loads"] += 1
            try:
                world, planner = load_planner(client, town, args.sampling_resolution)
            except Exception as exc:
                counters["town_planner_error:%s" % type(exc).__name__] += len(jobs_by_town[town])
                for path, selected in jobs_by_town[town]:
                    source_key = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
                    for actor, _label in selected:
                        target_outcomes[(source_key, int(actor["actor_id"]))]["status"] = (
                            "town_planner_error:%s" % type(exc).__name__
                        )
                continue
            for path, selected in jobs_by_town[town]:
                source_key = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
                for actor, label in selected:
                    counters["interactive_targets_requested"] += 1
                    try:
                        waypoint = label["interaction_waypoint"]
                        origin_waypoint = preview_waypoint(world.get_map(), actor)
                        destination_waypoint = conflict_waypoint(world.get_map(), waypoint)
                        if origin_waypoint is None or destination_waypoint is None:
                            counters["preview_or_conflict_waypoint_missing"] += 1
                            target_outcomes[(source_key, int(actor["actor_id"]))]["status"] = (
                                "preview_or_conflict_waypoint_missing"
                            )
                            continue
                        route, _route_ids, planner_distance = planner.trace_route(
                            origin_waypoint.transform.location,
                            destination_waypoint.transform.location,
                        )
                        polyline = [point_dict(item[0]) for item in route if item]
                    except Exception as exc:
                        counters["trace_error:%s" % type(exc).__name__] += 1
                        target_outcomes[(source_key, int(actor["actor_id"]))]["status"] = (
                            "trace_error:%s" % type(exc).__name__
                        )
                        continue
                    if len(polyline) < 2:
                        counters["trace_too_short"] += 1
                        target_outcomes[(source_key, int(actor["actor_id"]))]["status"] = "trace_too_short"
                        continue
                    try:
                        label_distance = float(label["cbv_distance"])
                    except (KeyError, TypeError, ValueError):
                        counters["rift_label_distance_missing"] += 1
                        target_outcomes[(source_key, int(actor["actor_id"]))]["status"] = "rift_label_distance_missing"
                        continue
                    distance_error = abs(float(planner_distance) - label_distance)
                    if distance_error > float(args.max_label_distance_abs_error_m):
                        counters["planner_distance_label_mismatch"] += 1
                        target_outcomes[(source_key, int(actor["actor_id"]))]["status"] = "planner_distance_label_mismatch"
                        continue
                    item = {
                        "schema_version": "rift_target_route_sidecar_v1",
                        "source_file_sha1": source_key,
                        "scene_id": scene_id_from_path(path),
                        "town": town,
                        "actor_id": int(actor["actor_id"]),
                        "target_route_to_interaction": polyline,
                        "planner_distance_m": float(planner_distance),
                        "rift_label_cbv_distance_m": label_distance,
                        "planner_distance_abs_error_to_rift_label_m": distance_error,
                        "preview_waypoint": point_dict(origin_waypoint),
                        "conflict_waypoint": point_dict(destination_waypoint),
                    }
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    counters["route_targets_written"] += 1
                    if target_manifest is not None:
                        written_manifest_targets.add((source_key, int(actor["actor_id"])))
                        target_outcomes[(source_key, int(actor["actor_id"]))].update({
                            "status": "written",
                            "planner_distance_m": float(planner_distance),
                            "rift_label_cbv_distance_m": label_distance,
                            "planner_distance_abs_error_to_rift_label_m": distance_error,
                        })
    missing_manifest_targets = sorted(expected_manifest_targets - written_manifest_targets)
    if target_manifest is not None:
        counters["manifest_target_count"] = len(expected_manifest_targets)
        counters["manifest_target_routes_written"] = len(written_manifest_targets)
        counters["manifest_target_routes_missing"] = len(missing_manifest_targets)
    summary = {
        "schema_version": "rift_target_route_sidecar_summary_v1",
        "input_dir": str(args.input_dir), "output_dir": str(args.output_dir),
        "route_interaction_sidecar": str(output_jsonl), "counters": dict(sorted(counters.items())),
        "target_manifest": str(args.target_manifest) if args.target_manifest else None,
        "target_manifest_complete": (
            None if target_manifest is None else not missing_manifest_targets
        ),
        "missing_manifest_target_examples": [
            {"source_file_sha1": source_key, "actor_id": actor_id}
            for source_key, actor_id in missing_manifest_targets[:30]
        ],
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    if target_manifest is not None:
        outcome_path = args.output_dir / "target_route_outcomes.jsonl"
        with outcome_path.open("w", encoding="utf-8") as handle:
            for key in sorted(target_outcomes):
                handle.write(json.dumps(target_outcomes[key], ensure_ascii=False) + "\n")
        summary["target_route_outcomes"] = str(outcome_path)
        summary["target_route_outcome_counts"] = dict(sorted(Counter(
            item["status"] for item in target_outcomes.values()
        ).items()))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.require_complete_target_manifest and missing_manifest_targets:
        raise SystemExit(
            "route sidecar is incomplete: %d target routes are missing; inspect summary.json"
            % len(missing_manifest_targets)
        )


def scene_id_from_path(path):
    # Diagnostic only; matching uses source_file_sha1, never scene id.
    return path.stem


if __name__ == "__main__":
    main()
