#!/usr/bin/env python
"""Build an independent directional ordinary-vehicle dataset from RIFT/CARLA.

This is deliberately *not* an HV dataset.  RIFT interaction labels are never
read for target selection and do not appear in a written record.  A record is
one current-time scene-completion example:

    rear / already-observed half (x < 0) -> current front half (x >= 0)

It is not a future-trajectory prediction task.  The emitted empirical count
model estimates ``P(N_after | map bucket, route bucket, N_before)`` from the
training split only, so validation/test information cannot leak into online
count selection.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import build_high_value_scene_dataset as base
import build_high_value_scene_dataset_v3 as v3
from high_value_init.rift_dataset import state_generation_schema_is_clean, write_jsonl


SCHEMA = "scenario_dreamer_background_directional_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rift_root", type=Path, required=True)
    parser.add_argument("--rift_hd_map_dir", type=Path, required=True)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--rear_max_x", type=float, default=0.0,
                        help="Condition is x < rear_max_x; default preserves Scenario Dreamer's half-FOV split.")
    parser.add_argument("--front_min_x", type=float, default=0.0,
                        help="Target is x >= front_min_x. Must equal rear_max_x for a gap-free partition.")
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_target_agents", type=int, default=8)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--max_town_fraction", type=float, default=0.35)
    parser.add_argument("--max_topology_fraction", type=float, default=0.75)
    parser.add_argument("--min_accepted_records", type=int, default=30000,
                        help="Minimum post-filter records.  A directional background model is not trained from a small balance-only patch.")
    parser.add_argument("--diagnostics_json", type=Path, default=None,
                        help="Write coverage/distribution evidence even when the strict diversity gate rejects the build.")
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_speed", type=float, default=0.0)
    parser.add_argument("--max_speed", type=float, default=25.0)
    parser.add_argument("--min_length", type=float, default=2.0)
    parser.add_argument("--max_length", type=float, default=8.0)
    parser.add_argument("--min_width", type=float, default=1.4)
    parser.add_argument("--max_width", type=float, default=3.2)
    parser.add_argument("--min_target_pair_distance", type=float, default=2.0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_files", type=int, default=-1)
    parser.add_argument("--source", default="carla_rift_raw_background_directional")
    return parser.parse_args()


def validate_args(args):
    if abs(float(args.fov) - 64.0) > 1e-6:
        raise ValueError("background v1 requires --fov 64 to retain pretrained Scenario Dreamer normalization")
    if abs(float(args.rear_max_x) - float(args.front_min_x)) > 1e-6:
        raise ValueError("directional background v1 requires rear_max_x == front_min_x; no unmodeled gap/overlap")
    half = float(args.fov) / 2.0
    if not -half < float(args.rear_max_x) < half:
        raise ValueError("partition boundary must lie strictly inside the FOV")
    if not 0.0 <= float(args.target_fov_margin) < half:
        raise ValueError("target FOV margin is invalid")
    if int(args.max_target_agents) < 1:
        raise ValueError("max_target_agents must be positive")
    if int(args.max_num_agents) != 61:
        raise ValueError("background v1 freezes Scenario Dreamer graph capacity at 61")
    if int(args.max_target_agents) >= int(args.max_num_agents):
        raise ValueError("max_target_agents leaves no ego/context capacity")
    if not 0.0 < float(args.max_town_fraction) <= 1.0 or not 0.0 < float(args.max_topology_fraction) <= 1.0:
        raise ValueError("maximum Town/topology fractions must be in (0, 1]")
    if int(args.min_accepted_records) < 1:
        raise ValueError("min_accepted_records must be positive")
    if not 0.0 <= float(args.val_ratio) < 1.0 or not 0.0 <= float(args.test_ratio) < 1.0:
        raise ValueError("split ratios must be in [0, 1)")
    if float(args.val_ratio) + float(args.test_ratio) >= 1.0:
        raise ValueError("validation plus test ratio must be below one")


def _valid_raw_snapshot(scene):
    # Labels may exist because the same RIFT snapshots support HV collection,
    # but their existence must never decide background record inclusion.
    return isinstance(scene, dict) and isinstance(scene.get("ego"), dict) and isinstance(scene.get("agents", scene.get("actors")), list)


def normalize_raw_snapshot(scene, actor_radius, fov):
    """Ego-local raw traffic normalization with no label lookup whatsoever."""
    ego = scene["ego"]
    half = float(fov) / 2.0
    agents = []
    for local_id, actor in enumerate(scene.get("agents", scene.get("actors", []))):
        try:
            distance = float(actor.get("ego_distance", math.hypot(float(actor["x"]) - float(ego["x"]), float(actor["y"]) - float(ego["y"]))))
            state = base.state_from_actor(ego, actor)
            actor_id = int(actor.get("actor_id", local_id))
        except (KeyError, TypeError, ValueError):
            continue
        if distance > float(actor_radius) or abs(float(state[0])) > half or abs(float(state[1])) > half:
            continue
        agents.append({"local_id": local_id, "actor_id": actor_id, "ego_distance": distance, "state": state})
    agents.sort(key=lambda row: (row["ego_distance"], row["actor_id"]))
    route = [base.point_local(ego, point) for point in scene.get("ego_route", []) if point is not None]
    if len(route) < 2:
        route = [[0.0, 0.0], [1.0, 0.0]]
    lanes = []
    for lane in scene.get("map_lanes", []) or []:
        points = [base.point_local(ego, point) for point in lane.get("centerline", []) if point is not None]
        if len(points) >= 2:
            lanes.append(points)
    scenario = scene.get("scenario") if isinstance(scene.get("scenario"), dict) else {}
    return {
        "scene_id": scene.get("scene_id") or str(scene.get("route_id")), "episode_key": base.episode_key(scene),
        "town": scene.get("town") or scenario.get("town"),
        "ego_state": [0.0, 0.0, 0.0, float(ego.get("speed_mps", 0.0)), float(ego.get("length", 5.176)), float(ego.get("width", 2.297)), 1.0],
        "ego_route": {"polyline": route}, "map": {"lane_polylines": lanes}, "agents": agents,
    }


def _state_valid(state, args):
    try:
        x, y, heading, speed, length, width = (float(value) for value in state[:6])
    except (TypeError, ValueError, IndexError):
        return False
    return all(math.isfinite(value) for value in (x, y, heading, speed, length, width)) and (
        float(args.min_speed) <= speed <= float(args.max_speed)
        and float(args.min_length) <= length <= float(args.max_length)
        and float(args.min_width) <= width <= float(args.max_width)
    )


def _target_sector(row, args):
    x, y = float(row["state"][0]), float(row["state"][1])
    limit = float(args.fov) / 2.0 - float(args.target_fov_margin)
    return x >= float(args.front_min_x) and x <= limit and abs(y) <= limit


def _rear_sector(row, args):
    # No target margin is applied to the condition: the historical rear
    # observation remains all x < boundary inside the model FOV.
    return float(row["state"][0]) < float(args.rear_max_x)


def _target_pair_clear(targets, args):
    minimum = float(args.min_target_pair_distance)
    if minimum <= 0.0:
        return True
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            dx = float(targets[left]["state"][0]) - float(targets[right]["state"][0])
            dy = float(targets[left]["state"][1]) - float(targets[right]["state"][1])
            if math.hypot(dx, dy) < minimum:
                return False
    return True


def _route_bucket(norm, lanes):
    points = (norm.get("ego_route") or {}).get("polyline") or []
    if len(points) < 3:
        return "route_unknown"
    headings = []
    for left, right in zip(points[:-1], points[1:]):
        dx, dy = float(right[0]) - float(left[0]), float(right[1]) - float(left[1])
        if math.hypot(dx, dy) > 1e-6:
            headings.append(math.atan2(dy, dx))
    if len(headings) < 2:
        return "route_straight"
    changes = [abs(math.atan2(math.sin(b - a), math.cos(b - a))) for a, b in zip(headings[:-1], headings[1:])]
    if max(changes, default=0.0) > math.radians(25.0):
        return "route_turning"
    return "route_straight"


def build_record(scene, lanes, args):
    """Return record/reason.  No RIFT label is accessed after normalization."""
    norm = normalize_raw_snapshot(scene, actor_radius=float(args.actor_radius), fov=float(args.fov))
    agents = list(norm["agents"])
    rear = [row for row in agents if _rear_sector(row, args)]
    front = [row for row in agents if _target_sector(row, args)]
    if not front:
        return None, "no_front_target"
    if len(front) > int(args.max_target_agents):
        return None, "target_count_exceeds_max"
    if any(not _state_valid(row["state"], args) for row in rear + front):
        return None, "raw_state_quality_rejected"
    if 1 + len(rear) + len(front) > int(args.max_num_agents):
        return None, "agent_capacity_rejected"
    if not _target_pair_clear(front, args):
        return None, "target_pair_clearance_rejected"
    states = [row["state"] for row in front]
    associations = [v3.associate(state, lanes, args.default_lane_half_width) for state in states]
    if any(item.get("distance") is None or item["distance"] > float(args.max_target_lane_distance) for item in associations):
        return None, "target_lane_rejected"
    # The only model condition is ego/route/HD-map/rear actors.  Keep
    # directional and count metadata outside condition and remove all actor
    # IDs, labels and interaction scores before serialization.
    record = {
        "schema_version": SCHEMA,
        "scene_id": norm["scene_id"],
        "episode_key": norm["episode_key"],
        "town": norm["town"],
        "condition": {
            "ego_state": norm["ego_state"],
            "fov": float(args.fov),
            "ego_route": norm["ego_route"],
            "map": {
                "lane_polylines": norm["map"]["lane_polylines"],
                "lanes_background_v1": lanes,
                # Alias only for the unchanged V3 geometry-loss implementation;
                # this is lane geometry, not a borrowed HV target task.
                "lanes_v3": lanes,
            },
            "observed_agent_states": [row["state"] for row in rear],
            "observed_agent_types": [1 for _ in rear],
            "num_generate_agents": len(front),
        },
        "target": {
            "generated_agent_states": states,
            "generated_agent_types": [1 for _ in states],
            "map_associations_background_v1": associations,
            "map_associations_v3": associations,
        },
        "background_contract": {
            "task": "current_front_scene_completion_not_future_prediction",
            "condition_partition": "rear_x_lt_boundary",
            "target_partition": "front_x_gte_boundary",
            "partition_boundary_x": float(args.rear_max_x),
            "target_fov_margin": float(args.target_fov_margin),
            "context_actor_count": len(rear),
            "target_actor_count": len(front),
            "agent_capacity_policy": "reject_record_no_truncation",
            "max_num_agents": int(args.max_num_agents),
        },
        "count_features": {
            "map_bucket": v3.topology_bucket(lanes),
            "route_bucket": _route_bucket(norm, lanes),
            "num_before": len(rear),
            "num_after": len(front),
        },
        "topology_background_v1": {"bucket": v3.topology_bucket(lanes), "lane_count": len(lanes)},
        "source": args.source,
    }
    # Existing generic schema check only protects HV fields.  The explicit
    # background audit below also rejects label/ID leakage.
    if not state_generation_schema_is_clean(record):
        raise ValueError("forbidden high-value selector field leaked into background record")
    return record, ""


def _dedup_key(record):
    condition = record["condition"]
    payload = {
        "episode": record["episode_key"],
        "rear": [[round(float(x), 1) for x in row[:3]] for row in condition["observed_agent_states"]],
        "front": [[round(float(x), 1) for x in row[:3]] for row in record["target"]["generated_agent_states"]],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def split_records(records, args):
    """Episode split before count estimation; prevents frame and count leakage."""
    by_episode = defaultdict(list)
    for record in records:
        by_episode[str(record["episode_key"])].append(record)
    keys = sorted(by_episode)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n_test = int(round(len(keys) * float(args.test_ratio)))
    n_val = int(round(len(keys) * float(args.val_ratio)))
    test_keys, val_keys = set(keys[:n_test]), set(keys[n_test:n_test + n_val])
    output = {"train": [], "val": [], "test": []}
    for key, rows in by_episode.items():
        split = "test" if key in test_keys else "val" if key in val_keys else "train"
        output[split].extend(rows)
    index = 0
    for split in ("train", "val", "test"):
        for record in output[split]:
            record["idx"], record["split"] = index, split
            index += 1
    return output


def estimate_count_model(train_records, args):
    """Empirical smoothed count PMFs, estimated only from training records."""
    maximum = int(args.max_target_agents)
    groups = defaultdict(Counter)
    global_counts = Counter()
    for record in train_records:
        feature = record["count_features"]
        key = (str(feature["map_bucket"]), str(feature["route_bucket"]), min(int(feature["num_before"]), 20))
        count = int(feature["num_after"])
        groups[key][count] += 1
        global_counts[count] += 1
    def pmf(counter):
        # Laplace smoothing makes an unseen count possible but never exposes
        # val/test statistics.  Online sampling remains deterministic only if
        # a caller supplies a seed.
        total = sum(counter.values()) + maximum
        return {str(value): (counter[value] + 1.0) / total for value in range(1, maximum + 1)}
    return {
        "schema_version": "scenario_dreamer_background_count_model_v1",
        "estimated_from_split": "train_only",
        "task": "current_front_scene_completion_not_future_prediction",
        "features": ["map_bucket", "route_bucket", "num_before_capped_20"],
        "max_target_agents": maximum,
        "global_pmf": pmf(global_counts),
        "conditional_pmf": {"|".join(map(str, key)): pmf(value) for key, value in sorted(groups.items())},
        "train_record_count": len(train_records),
    }


def distribution_failures(records, args):
    total = max(1, len(records))
    towns = Counter(str(record.get("town") or "unknown") for record in records)
    topology = Counter(str(record["count_features"]["map_bucket"]) for record in records)
    failures = []
    for name, counts, maximum in (("town", towns, args.max_town_fraction), ("topology", topology, args.max_topology_fraction)):
        for key, count in counts.items():
            fraction = count / total
            if fraction > float(maximum) + 1e-12:
                failures.append(f"{name}_fraction_exceeded:{key}:{fraction:.6f}>{float(maximum):.6f}")
    return dict(sorted(towns.items())), dict(sorted(topology.items())), failures


def main():
    args = parse_args()
    validate_args(args)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite background dataset directory: {args.output_dir}")
    extractor = v3.HDMapLaneExtractor(args.rift_root, args.rift_hd_map_dir)
    records, seen = [], set()
    rejection, map_errors = Counter(), Counter()
    processed = valid = duplicates = 0
    for path in sorted(args.input_dir.rglob("*.json")):
        if int(args.max_input_files) > 0 and processed >= int(args.max_input_files):
            break
        processed += 1
        try:
            scene = base.load_snapshot(path)
        except Exception as exc:
            rejection[f"snapshot_load:{type(exc).__name__}"] += 1
            continue
        if not _valid_raw_snapshot(scene):
            rejection["not_raw_rift_carla_snapshot"] += 1
            continue
        valid += 1
        try:
            # HDMapLaneExtractor returns world-coordinate lanes.  Put them
            # into an isolated copy and convert them to the ego-local format
            # used by both the graph builder and the lane association code.
            scene = json.loads(json.dumps(scene))
            scene["map_lanes"] = extractor.lanes(scene, float(args.fov))
            lanes = v3.local_lanes(scene, float(args.fov))
        except Exception as exc:
            map_errors[type(exc).__name__] += 1
            continue
        record, reason = build_record(scene, lanes, args)
        if record is None:
            rejection[reason] += 1
            continue
        key = _dedup_key(record)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        records.append(record)
    if not records:
        raise ValueError("background builder produced no records; inspect raw traffic coverage rather than relaxing the directional contract")
    splits = split_records(records, args)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError("background dataset split is empty; collect enough independent episodes")
    town_counts, topology_counts, diversity_failures = distribution_failures(records, args)
    summary = {
        "schema_version": SCHEMA,
        "processed_files": processed,
        "valid_raw_snapshot_count": valid,
        "records_before_split": len(records),
        "min_accepted_records_required": int(args.min_accepted_records),
        "duplicates_removed": duplicates,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "rejections": dict(sorted(rejection.items())),
        "hd_map_errors": dict(sorted(map_errors.items())),
        "target_count_histogram": dict(sorted(Counter(r["count_features"]["num_after"] for r in records).items())),
        "context_count_histogram": dict(sorted(Counter(r["count_features"]["num_before"] for r in records).items())),
        "town_counts": town_counts, "topology_counts": topology_counts,
        "distribution_failures": diversity_failures,
        "args": vars(args),
    }
    if args.diagnostics_json:
        args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    scale_failures = []
    if len(records) < int(args.min_accepted_records):
        scale_failures.append("accepted_record_count_below_minimum:%d<%d" % (len(records), int(args.min_accepted_records)))
    if diversity_failures or scale_failures:
        raise ValueError("background dataset violates strict scale/Town/topology gates: " + "; ".join(diversity_failures + scale_failures) +
                         ". Collect targeted non-dominant ordinary-traffic routes; do not relax the contract.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    count_model = estimate_count_model(splits["train"], args)
    (args.output_dir / "count_model.json").write_text(json.dumps(count_model, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
