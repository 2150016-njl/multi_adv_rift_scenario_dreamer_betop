#!/usr/bin/env python
"""Plan balanced V4-A forward-HV snapshot collection without mutating data.

The planner uses the exact V4-A offline target-selection contract to find
which historical routes have yielded eligible examples.  It emits a manifest
for collection workers; it never down-samples Town12, edits snapshots, or
relaxes the V4-A dataset gate.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--existing_v4a_diagnostics", type=Path, required=True)
    parser.add_argument("--routes_xml", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--rift_root", type=Path, required=True)
    parser.add_argument("--rift_hd_map_dir", type=Path, required=True)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--max_town_fraction", type=float, default=0.35)
    parser.add_argument("--max_topology_fraction", type=float, default=0.75,
                        help="Final V4-A dataset gate recorded in the manifest.  The builder, not this planner, enforces it.")
    parser.add_argument("--min_eligible_episodes_per_topology", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed_start", type=int, default=8)
    parser.add_argument("--max_routes_per_town", type=int, default=16)
    parser.add_argument("--max_exploration_routes_per_town", type=int, default=4)
    parser.add_argument("--target_total_accepted_records", type=int, default=0,
                        help="Optional scale goal for V4-A accepted records. Zero preserves the legacy balance-only plan.")
    parser.add_argument("--min_accepted_records_per_town", type=int, default=0,
                        help="Optional Town coverage floor used with target_total_accepted_records.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def route_catalog(path: Path):
    root = ET.parse(path).getroot()
    catalog = {}
    for node in root.findall(".//route"):
        route_id, town = node.get("id"), node.get("town")
        if route_id and town:
            catalog[str(route_id)] = str(town)
    if not catalog:
        raise ValueError(f"no route id/town entries in {path}")
    return catalog


def route_id_from_scene(scene):
    scenario = scene.get("scenario") if isinstance(scene.get("scenario"), dict) else {}
    values = (scene.get("route_id"), scenario.get("route_id"), scenario.get("name"),
              scene.get("scene_id"), scene.get("episode_key"))
    for value in values:
        text = str(value or "")
        match = re.search(r"RouteScenario_(\d+)", text)
        if match:
            return match.group(1)
        if re.fullmatch(r"\d+", text):
            return text
    return None


def town_from_scene(scene):
    scenario = scene.get("scenario") if isinstance(scene.get("scenario"), dict) else {}
    return str(scene.get("town") or scenario.get("town") or "")


def v4_args(args):
    """Only fields consumed by V4-A normalization and record selection."""
    return SimpleNamespace(
        actor_radius=args.actor_radius, fov=args.fov,
        min_rift_positive_agents=2, num_generate_agents=args.num_generate_agents,
        forward_min_x=args.forward_min_x, target_fov_margin=args.target_fov_margin,
        max_num_agents=61, default_lane_half_width=1.75,
        max_target_lane_distance=args.max_target_lane_distance,
        positive_target_policy="route_s_diversity", source="planning_only",
    )


def validate_contract(args, diagnostics):
    if not args.input_dir.is_dir():
        raise ValueError(f"input_dir does not exist: {args.input_dir}")
    if args.output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite collection manifest: {args.output_manifest}")
    if args.num_generate_agents != 2 or abs(args.fov - 64.0) > 1e-6:
        raise ValueError("V4-A planning requires fov=64 and exactly two targets")
    if not 0.0 <= args.forward_min_x < args.fov / 2.0:
        raise ValueError("forward_min_x must be in [0, fov/2)")
    if not 0.0 <= args.target_fov_margin < args.fov / 2.0:
        raise ValueError("target_fov_margin must be in [0, fov/2)")
    if args.forward_min_x > args.fov / 2.0 - args.target_fov_margin:
        raise ValueError("forward target interval is empty")
    if not 0.0 < args.max_town_fraction < 1.0 or not 0.0 < args.max_topology_fraction <= 1.0:
        raise ValueError("max_town_fraction and max_topology_fraction must be in (0, 1]")
    if args.workers <= 0 or args.max_routes_per_town <= 0 or args.target_total_accepted_records < 0 or args.min_accepted_records_per_town < 0:
        raise ValueError("workers and max_routes_per_town must be positive")
    if diagnostics.get("schema_version") != "hv_scene_dataset_v4a_full_fov_forward":
        raise ValueError("existing diagnostics are not from the V4-A builder")
    prior = diagnostics.get("args") or {}
    for key, expected in (("fov", args.fov), ("forward_min_x", args.forward_min_x),
                          ("target_fov_margin", args.target_fov_margin),
                          ("num_generate_agents", args.num_generate_agents)):
        if key not in prior or abs(float(prior[key]) - float(expected)) > 1e-6:
            raise ValueError(f"diagnostic contract mismatch for {key}: {prior.get(key)!r} != {expected!r}")


def distribute_quota(existing, candidate_towns, total_needed):
    """Deterministically favor towns with fewer accepted V4-A records."""
    allocation = Counter()
    if total_needed <= 0:
        return allocation
    if not candidate_towns:
        raise ValueError("no non-dominant town has historical V4-A route evidence")
    for _ in range(total_needed):
        town = min(candidate_towns, key=lambda name: (int(existing.get(name, 0)) + allocation[name], name))
        allocation[town] += 1
    return allocation


def select_routes(route_stats, allocation, topology_deficits, dominant_town, args):
    by_town = defaultdict(list)
    by_topology = defaultdict(list)
    for route_id, row in route_stats.items():
        if row["town"] == dominant_town:
            continue
        by_town[row["town"]].append((route_id, row))
        for topology in row["topology_candidate_counts"]:
            by_topology[topology].append((route_id, row))

    selected = {}
    unresolved = []
    for town, quota in sorted(allocation.items()):
        choices = sorted(by_town.get(town, []), key=lambda pair: (-pair[1]["candidate_records"], pair[0]))
        if not choices:
            unresolved.append(f"no_historical_candidate_route_for_town:{town}")
            continue
        for route_id, row in choices[:args.max_routes_per_town]:
            selected[route_id] = row
    for topology, deficit in sorted(topology_deficits.items()):
        if deficit <= 0:
            continue
        choices = sorted(
            by_topology.get(topology, []),
            key=lambda pair: (-len(pair[1]["topology_episode_counts"].get(topology, set())),
                              -pair[1]["candidate_records"], pair[0]),
        )
        if not choices:
            unresolved.append(f"no_non_dominant_historical_candidate_route_for_topology:{topology}")
            continue
        for route_id, row in choices[:args.max_routes_per_town]:
            selected[route_id] = row
    return selected, unresolved


def worker_plan(selected, catalog, allocation, args):
    workers = [{"worker_index": index, "seed": args.seed_start + index, "routes": [], "expected_candidate_records": 0}
               for index in range(args.workers)]
    ranked = sorted(selected.items(), key=lambda pair: (-pair[1]["candidate_records"], pair[0]))
    for route_id, row in ranked:
        worker = min(workers, key=lambda item: (item["expected_candidate_records"], item["worker_index"]))
        worker["routes"].append(route_id)
        worker["expected_candidate_records"] += int(row["candidate_records"])
    for worker in workers:
        towns = sorted({catalog[route_id] for route_id in worker["routes"]})
        worker["routes_subset"] = ",".join(worker["routes"])
        worker["towns"] = towns
        worker["accepted_record_quota_by_town"] = {}
    for town, quota in sorted(allocation.items()):
        candidates = [worker for worker in workers if town in worker["towns"]]
        if not candidates:
            continue
        for _ in range(int(quota)):
            worker = min(candidates, key=lambda item: (
                sum(item["accepted_record_quota_by_town"].values()), item["worker_index"]
            ))
            worker["accepted_record_quota_by_town"][town] = (
                int(worker["accepted_record_quota_by_town"].get(town, 0)) + 1
            )
    return workers


def main():
    args = parse_args()
    diagnostics = read_json(args.existing_v4a_diagnostics)
    validate_contract(args, diagnostics)
    catalog = route_catalog(args.routes_xml)

    # These imports intentionally occur after argument validation.  They use
    # the same V4-A record contract as the dataset builder but are never used
    # to create a training record or modify a snapshot.
    import build_high_value_scene_dataset as base
    import build_high_value_scene_dataset_v3 as v3
    import build_high_value_scene_dataset_v4 as v4

    extractor = v3.HDMapLaneExtractor(args.rift_root, args.rift_hd_map_dir)
    route_stats = {}
    invalid_snapshots = Counter()
    candidate_rejections = Counter()
    candidate_rejections_by_town = defaultdict(Counter)
    candidate_exception_examples = []
    unmapped_routes = []
    topology_episodes = defaultdict(set)
    planner_args = v4_args(args)
    paths = sorted(args.input_dir.rglob("*.json"))
    for path in paths:
        try:
            scene = base.load_snapshot(path)
        except Exception as exc:
            invalid_snapshots[f"load:{type(exc).__name__}"] += 1
            continue
        if not v4._is_rift_labeled_snapshot(scene):
            invalid_snapshots["not_rift_labeled_snapshot"] += 1
            continue
        route_id = route_id_from_scene(scene)
        if route_id is None or route_id not in catalog:
            invalid_snapshots["route_id_unmapped"] += 1
            if len(unmapped_routes) < 20:
                unmapped_routes.append({"path": str(path), "route_id": route_id, "town": scene.get("town")})
            continue
        if town_from_scene(scene) != catalog[route_id]:
            invalid_snapshots["route_town_mismatch"] += 1
            continue
        try:
            scene["map_lanes"] = extractor.lanes(scene, args.fov)
            lanes = v3.local_lanes(scene, args.fov)
            norm = base.normalize_snapshot(scene, actor_radius=args.actor_radius, fov=args.fov)
            record, reason = v4._build_record(scene, lanes, planner_args, norm=norm)
        except Exception as exc:
            invalid_snapshots[f"candidate:{type(exc).__name__}"] += 1
            candidate_rejections_by_town[catalog[route_id]][f"exception:{type(exc).__name__}"] += 1
            if len(candidate_exception_examples) < 20:
                candidate_exception_examples.append({
                    "path": str(path), "route_id": route_id, "town": catalog[route_id],
                    "type": type(exc).__name__, "message": str(exc),
                })
            continue
        if not record:
            reason = str(reason or "unknown_record_rejection")
            candidate_rejections[reason] += 1
            candidate_rejections_by_town[catalog[route_id]][reason] += 1
            continue
        topology = str((record.get("topology_v4") or {}).get("bucket") or "unknown")
        episode = str(record["episode_key"])
        row = route_stats.setdefault(route_id, {
            "town": catalog[route_id], "candidate_records": 0,
            "candidate_episode_keys": set(), "topology_candidate_counts": Counter(),
            "topology_episode_counts": defaultdict(set),
        })
        row["candidate_records"] += 1
        row["candidate_episode_keys"].add(episode)
        row["topology_candidate_counts"][topology] += 1
        row["topology_episode_counts"][topology].add(episode)
        topology_episodes[topology].add(episode)

    distribution = diagnostics.get("distribution") or {}
    town_counts = Counter({str(key): int(value) for key, value in (distribution.get("town_counts") or {}).items()})
    if not town_counts:
        raise ValueError("existing diagnostics lack post-filter town_counts; run the non-diagnostics V4-A build first")
    dominant_town, dominant_count = max(town_counts.items(), key=lambda pair: (pair[1], pair[0]))
    total_existing = int(sum(town_counts.values()))
    target_total = int(math.ceil(dominant_count / args.max_town_fraction))
    required_non_dominant = max(0, target_total - total_existing)
    candidate_towns = sorted({row["town"] for row in route_stats.values() if row["town"] != dominant_town})
    # A zero-candidate result is itself useful evidence.  Persist it below
    # rather than throwing before the caller can distinguish target scarcity,
    # lane rejection, and a planner exception.
    allocation = Counter()
    unresolved = []
    scale_required = max(0, int(args.target_total_accepted_records) - total_existing)
    floor_allocation = Counter({town: max(0, int(args.min_accepted_records_per_town) - int(town_counts.get(town, 0)))
                                for town in candidate_towns})
    requested_non_dominant = max(required_non_dominant, scale_required, sum(floor_allocation.values()))
    if requested_non_dominant > 0:
        if candidate_towns:
            allocation = Counter(floor_allocation)
            allocation.update(distribute_quota(town_counts + allocation, candidate_towns,
                                               requested_non_dominant - sum(allocation.values())))
        else:
            unresolved.append("no_non_dominant_town_has_historical_v4a_candidate_evidence")
    expected_topologies = set((distribution.get("topology_bucket_counts") or {}).keys())
    expected_topologies.update(topology_episodes)
    topology_deficits = {
        topology: max(0, args.min_eligible_episodes_per_topology - len(topology_episodes.get(topology, set())))
        for topology in sorted(expected_topologies)
    }
    selected, selection_unresolved = select_routes(route_stats, allocation, topology_deficits, dominant_town, args)
    unresolved.extend(selection_unresolved)

    serializable_routes = {}
    for route_id, row in sorted(selected.items()):
        serializable_routes[route_id] = {
            "town": row["town"],
            "candidate_records": int(row["candidate_records"]),
            "candidate_episode_count": len(row["candidate_episode_keys"]),
            "topology_candidate_counts": dict(sorted(row["topology_candidate_counts"].items())),
            "topology_episode_counts": {key: len(value) for key, value in sorted(row["topology_episode_counts"].items())},
        }
    workers = worker_plan(serializable_routes, catalog, allocation, args)
    selected_candidate_records_by_town = Counter()
    for row in serializable_routes.values():
        selected_candidate_records_by_town[str(row["town"])] += int(row["candidate_records"])
    one_pass_shortfall = {
        town: max(0, int(quota) - int(selected_candidate_records_by_town.get(town, 0)))
        for town, quota in sorted(allocation.items())
    }
    approximate_seed_passes_lower_bound = {
        town: (
            None if int(selected_candidate_records_by_town.get(town, 0)) <= 0
            else int(math.ceil(int(quota) / int(selected_candidate_records_by_town[town])))
        )
        for town, quota in sorted(allocation.items())
    }
    forecast_town_counts = Counter(town_counts)
    forecast_town_counts.update(allocation)
    all_routes_by_town = defaultdict(list)
    for route_id, town in catalog.items():
        if town != dominant_town and route_id not in route_stats:
            all_routes_by_town[town].append(route_id)
    exploration = {
        town: sorted(route_ids)[:args.max_exploration_routes_per_town]
        for town, route_ids in sorted(all_routes_by_town.items()) if town in allocation
    }

    manifest = {
        "schema_version": "hv_v4a_targeted_collection_manifest_v1",
        "planner_mode": "read_only_historical_evidence",
        "collection_contract": {
            "fov": args.fov, "num_generate_agents": args.num_generate_agents,
            "forward_min_x": args.forward_min_x, "target_fov_margin": args.target_fov_margin,
            "min_forward_rift_positive": 2, "map_source": "hd_map",
            "raw_state_postprocessing": "none",
            "max_topology_fraction_final_dataset_gate": args.max_topology_fraction,
        },
        "inputs": {
            "input_dir": str(args.input_dir), "existing_v4a_diagnostics": str(args.existing_v4a_diagnostics),
            "existing_v4a_diagnostics_sha256": sha256_file(args.existing_v4a_diagnostics),
            "routes_xml": str(args.routes_xml), "routes_xml_sha256": sha256_file(args.routes_xml),
        },
        "existing_distribution": {
            "total_records": total_existing, "town_counts": dict(sorted(town_counts.items())),
            "dominant_town": dominant_town, "dominant_town_records": dominant_count,
            "max_town_fraction": args.max_town_fraction,
        },
        "required_balance": {
            "minimum_total_records_if_dominant_unchanged": target_total,
            "required_new_non_dominant_accepted_records": required_non_dominant,
            "accepted_record_quota_by_town": dict(sorted(allocation.items())),
            "forecast_town_counts_if_quotas_met": dict(sorted(forecast_town_counts.items())),
            "dominant_town_policy": "no_new_generic_collection",
        },
        "scale_collection_goal": {
            "target_total_accepted_records": int(args.target_total_accepted_records),
            "minimum_accepted_records_per_town": int(args.min_accepted_records_per_town),
            "existing_accepted_records": total_existing,
            "requested_new_accepted_records": requested_non_dominant,
            "warning": "This is an accepted-record planning target. Historical candidate evidence is not an accepted-record guarantee; collect in waves and rebuild V4-A strictly after each wave.",
        },
        "historical_one_pass_feasibility": {
            "selected_candidate_records_by_town": dict(sorted(selected_candidate_records_by_town.items())),
            "quota_shortfall_against_one_historical_pass_by_town": one_pass_shortfall,
            "one_historical_pass_meets_all_town_quotas": not any(one_pass_shortfall.values()),
            "approximate_seed_passes_lower_bound_by_town": approximate_seed_passes_lower_bound,
            "warning": "Historical candidate counts are route/seed evidence, not accepted-record guarantees. Explore additional routes and rebuild the V4-A dataset before training.",
        },
        "topology_episode_evidence_before_collection": {
            key: len(value) for key, value in sorted(topology_episodes.items())
        },
        "topology_episode_deficits": topology_deficits,
        "selected_routes": serializable_routes,
        "workers": workers,
        "exploration_route_ids_by_town": exploration,
        "unresolved_planning_reasons": unresolved,
        "input_rejections": dict(sorted(invalid_snapshots.items())),
        "candidate_rejections": dict(sorted(candidate_rejections.items())),
        "candidate_rejections_by_town": {
            town: dict(sorted(values.items())) for town, values in sorted(candidate_rejections_by_town.items())
        },
        "candidate_exception_examples": candidate_exception_examples,
        "unmapped_route_examples": unmapped_routes,
        "training_gate": "Do not train V4-A until new data is merged and build_high_value_scene_dataset_v4.py passes its strict distribution audit.",
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not serializable_routes:
        raise SystemExit("manifest has no eligible non-dominant routes; inspect candidate_rejections and candidate_exception_examples")


if __name__ == "__main__":
    main()
