#!/usr/bin/env python
"""Build v2 high-value scene-generation datasets with quality filtering.

This script is intentionally separate from build_high_value_scene_dataset.py so
the Phase 5 dataset path remains reproducible.  It reuses the original record
builder, then applies explicit v2 filters, de-duplication, and record weights.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import build_high_value_scene_dataset as base
from high_value_init.rift_dataset import write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["inpainting", "full_generation"], required=True)
    parser.add_argument("--min_rift_positive_agents", type=int, default=2)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--positive_target_policy", default="route_s_diversity",
                        choices=["route_s_diversity", "top_score"])
    parser.add_argument("--sparse_context_copies", type=int, default=0)
    parser.add_argument("--sparse_context_min_observed", type=int, default=0)
    parser.add_argument("--sparse_context_max_observed", type=int, default=2)
    parser.add_argument("--sparse_context_replace_base", action="store_true", default=False)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default="real_positive_v2")

    parser.add_argument("--quality_filter_v2", action="store_true", default=False)
    parser.add_argument("--min_target_speed", type=float, default=0.5)
    parser.add_argument("--max_target_speed", type=float, default=25.0)
    parser.add_argument("--min_target_ego_distance", type=float, default=3.0)
    parser.add_argument("--max_target_ego_distance", type=float, default=60.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--min_target_pair_distance", type=float, default=2.0)
    parser.add_argument("--max_observed_agents", type=int, default=-1,
                        help="Drop records with more observed agents after context sampling. -1 disables.")
    parser.add_argument("--dedup_grid_xy", type=float, default=1.0,
                        help="Round target xy by this grid for duplicate removal. <=0 disables.")
    parser.add_argument("--base_training_weight", type=float, default=1.0)
    parser.add_argument("--sparse_context_weight_bonus", type=float, default=0.25)
    parser.add_argument("--low_observed_weight_bonus", type=float, default=0.25)
    parser.add_argument("--high_speed_weight_bonus", type=float, default=0.15)
    parser.add_argument("--weight_key", default="training_weight")
    return parser.parse_args()


def validate_args(args) -> None:
    base.validate_args(args)
    if args.min_target_speed > args.max_target_speed:
        raise ValueError("--min_target_speed must be <= --max_target_speed")
    if args.min_target_ego_distance > args.max_target_ego_distance:
        raise ValueError("--min_target_ego_distance must be <= --max_target_ego_distance")
    if args.target_fov_margin < 0:
        raise ValueError("--target_fov_margin must be >= 0")
    if args.min_target_pair_distance < 0:
        raise ValueError("--min_target_pair_distance must be >= 0")
    if args.base_training_weight <= 0:
        raise ValueError("--base_training_weight must be > 0")


def target_states(record: Dict) -> List[List[float]]:
    target = record.get("target") or {}
    return list(target.get("generated_agent_states") or target.get("agent_states") or [])


def observed_states(record: Dict) -> List[List[float]]:
    condition = record.get("condition") or {}
    return list(condition.get("observed_agent_states") or [])


def pair_distances(states: List[List[float]]) -> List[float]:
    distances = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            distances.append(math.hypot(float(states[i][0]) - float(states[j][0]),
                                        float(states[i][1]) - float(states[j][1])))
    return distances


def reject_reasons(record: Dict, args) -> List[str]:
    states = target_states(record)
    reasons = []
    half_fov = float(args.fov) / 2.0
    max_abs_allowed = half_fov - float(args.target_fov_margin)
    if len(states) < int(args.num_generate_agents):
        reasons.append("too_few_targets")
    if args.max_observed_agents >= 0 and len(observed_states(record)) > args.max_observed_agents:
        reasons.append("too_many_observed_agents")

    for state in states:
        x, y, _heading, speed, length, width = [float(v) for v in state[:6]]
        distance = math.hypot(x, y)
        if speed < args.min_target_speed:
            reasons.append("target_speed_too_low")
        if speed > args.max_target_speed:
            reasons.append("target_speed_too_high")
        if distance < args.min_target_ego_distance:
            reasons.append("target_too_close")
        if distance > args.max_target_ego_distance:
            reasons.append("target_too_far")
        if abs(x) > max_abs_allowed or abs(y) > max_abs_allowed:
            reasons.append("target_near_or_outside_fov")
        if length <= 0 or width <= 0:
            reasons.append("target_invalid_size")

    if len(states) >= 2:
        distances = pair_distances(states)
        if distances and min(distances) < args.min_target_pair_distance:
            reasons.append("target_pair_too_close")
    return sorted(set(reasons))


def duplicate_key(record: Dict, args) -> Tuple:
    states = target_states(record)
    if args.dedup_grid_xy <= 0:
        return (record.get("scene_id"), record.get("idx"))
    grid = float(args.dedup_grid_xy)
    rounded = []
    for state in sorted(states, key=lambda row: (float(row[0]), float(row[1]))):
        rounded.append((round(float(state[0]) / grid), round(float(state[1]) / grid)))
    observed = tuple(
        sorted(
            (round(float(state[0]) / grid), round(float(state[1]) / grid))
            for state in observed_states(record)
        )
    )
    sampling = record.get("context_sampling") or {}
    return (
        record.get("episode_key"),
        tuple(rounded),
        sampling.get("policy"),
        sampling.get("sparse_context_copy_index"),
        observed,
    )


def annotate_weight(record: Dict, args) -> Dict:
    record = dict(record)
    weight = float(args.base_training_weight)
    sampling = record.get("context_sampling") or {}
    if sampling.get("policy") in {"sparse_random_observed", "sparse_inpainting"}:
        weight += float(args.sparse_context_weight_bonus)
    if len(observed_states(record)) <= 2:
        weight += float(args.low_observed_weight_bonus)
    speeds = [float(state[3]) for state in target_states(record) if len(state) >= 4]
    if speeds and max(speeds) >= 8.0:
        weight += float(args.high_speed_weight_bonus)
    record[args.weight_key] = round(max(weight, 1e-6), 6)
    record["quality_v2"] = {
        "target_speed": speeds,
        "target_ego_distance": [math.hypot(float(s[0]), float(s[1])) for s in target_states(record)],
        "target_pair_distance_min": min(pair_distances(target_states(record))) if len(target_states(record)) >= 2 else None,
        "observed_agent_count": len(observed_states(record)),
        "quality_filter_v2": bool(args.quality_filter_v2),
    }
    return record


def filter_and_weight(records: Iterable[Dict], args):
    kept = []
    reject_counter = Counter()
    rejected_records = 0
    duplicate_counter = 0
    seen = set()
    for record in records:
        reasons = reject_reasons(record, args) if args.quality_filter_v2 else []
        if reasons:
            rejected_records += 1
            for reason in reasons:
                reject_counter[reason] += 1
            continue
        key = duplicate_key(record, args)
        if key in seen:
            duplicate_counter += 1
            continue
        seen.add(key)
        kept.append(annotate_weight(record, args))
    return kept, reject_counter, rejected_records, duplicate_counter


def numeric(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def quality_summary(
        records: List[Dict],
        args,
        reject_counter: Counter,
        rejected_records: int,
        duplicate_count: int,
        skipped: int) -> Dict:
    weights = [record.get(args.weight_key, 1.0) for record in records]
    observed = [len(observed_states(record)) for record in records]
    target_speeds = [float(state[3]) for record in records for state in target_states(record)]
    target_distances = [math.hypot(float(state[0]), float(state[1])) for record in records for state in target_states(record)]
    sparse = sum(
        1 for record in records
        if (record.get("context_sampling") or {}).get("policy") in {"sparse_random_observed", "sparse_inpainting"}
    )
    return {
        "schema_version": "hv_scene_dataset_quality_v2",
        "num_records": len(records),
        "num_skipped_by_builder": skipped,
        "num_rejected_by_quality": int(rejected_records),
        "num_quality_reject_reason_occurrences": int(sum(reject_counter.values())),
        "reject_reasons": dict(reject_counter.most_common()),
        "num_removed_duplicates": int(duplicate_count),
        "sparse_context_record_count": int(sparse),
        "sparse_context_record_rate": sparse / len(records) if records else 0.0,
        "observed_agent_count": numeric(observed),
        "target_speed": numeric(target_speeds),
        "target_ego_distance": numeric(target_distances),
        "training_weight": numeric(weights),
        "args": json_safe(vars(args)),
    }


def main():
    args = parse_args()
    validate_args(args)
    paths = sorted(args.input_dir.rglob("*.json"))
    raw_records = []
    skipped = 0
    for path in paths:
        scene = base.load_snapshot(path)
        scene_records = base.build_records(scene, args)
        if not scene_records:
            skipped += 1
            continue
        source_file_sha1 = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
        for record in scene_records:
            record["source_file_sha1"] = source_file_sha1
            raw_records.append(record)

    records, reject_counter, rejected_records, duplicate_count = filter_and_weight(raw_records, args)
    train, val, test = base.split_records(records, args.val_ratio, args.test_ratio, args.seed)
    write_jsonl(str(args.output_dir / "train.jsonl"), train)
    write_jsonl(str(args.output_dir / "val.jsonl"), val)
    if args.test_ratio > 0:
        write_jsonl(str(args.output_dir / "test.jsonl"), test)

    summary = quality_summary(records, args, reject_counter, rejected_records, duplicate_count, skipped)
    summary.update({
        "output_dir": str(args.output_dir),
        "num_raw_records": len(raw_records),
        "num_train_records": len(train),
        "num_val_records": len(val),
        "num_test_records": len(test),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
