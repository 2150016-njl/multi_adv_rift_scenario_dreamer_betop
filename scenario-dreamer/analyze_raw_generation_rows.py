#!/usr/bin/env python
"""Summarize joint raw-state feasibility from frozen evaluator JSONL rows.

The existing evaluators report marginal means.  This script adds the more
useful question for deployment diagnostics: how often does one unmodified
sample satisfy every applicable geometric condition at once?  It only reads
rows already emitted by an evaluator and never ranks, filters, or rewrites a
sample.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from high_value_init.evaluation_geometry import geometry_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs=2,
        metavar=("NAME", "GENERATED_JSONL"),
        action="append",
        required=True,
        help="Repeat for V2/V3/V4A/background generated JSONL evidence.",
    )
    parser.add_argument(
        "--source_jsonl",
        nargs=2,
        metavar=("NAME", "JSONL"),
        action="append",
        default=[],
        help=(
            "Optional source dataset for one --input NAME. Enables read-only "
            "target-distance/speed/context buckets and post-hoc OBB measurement."
        ),
    )
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_index_of(row):
    """Support both historical evaluator row schemas without rewriting rows."""
    return row.get("record_index", row.get("record_idx"))


def as_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def is_zero(metrics, key):
    value = as_number(metrics.get(key))
    return None if value is None else value <= 1e-12


def less_or_equal(metrics, key, threshold):
    value = as_number(metrics.get(key))
    return None if value is None else value <= threshold


def all_known_true(values):
    known = [value for value in values if value is not None]
    return None if not known else all(known)


def infer_task(row):
    metrics = row.get("metrics") or {}
    if "generated_forward_violation_count" in metrics:
        return "v4a_forward_hv"
    if "forward_violation_count" in metrics and "target_count" in row:
        return "directional_background"
    return "full_fov_hv"


def flags_for_row(row, default_lane_half_width, source_record=None):
    metrics = row.get("metrics") or {}
    task = infer_task(row)
    physical_valid = is_zero(metrics, "physical_invalid_rate")
    fov_valid = is_zero(metrics, "off_fov_failure_count")
    nearest_lane_valid = is_zero(metrics, "nearest_lane_corridor_failure_count")
    if nearest_lane_valid is None:
        nearest_lane_valid = is_zero(metrics, "lane_corridor_failure_count")

    associated_lane_valid = is_zero(metrics, "associated_lane_corridor_failure_count")
    if associated_lane_valid is None:
        associated_max = as_number(metrics.get("associated_lane_distance_max"))
        # V4-A names this field ``missing`` while the V2/V3 geometry helper
        # calls it ``unavailable``.  Both mean that a generated state could
        # not be measured against its target-associated lane.
        associated_missing = as_number(metrics.get("associated_lane_missing_count"))
        if associated_missing is None:
            associated_missing = as_number(metrics.get("associated_lane_unavailable_count"))
        if associated_max is not None:
            associated_lane_valid = (
                associated_max <= float(default_lane_half_width) and
                (associated_missing is None or associated_missing <= 0.0)
            )

    clearance_valid = all_known_true((
        is_zero(metrics, "ego_clearance_failure_count"),
        is_zero(metrics, "observed_clearance_failure_count"),
        is_zero(metrics, "pair_clearance_failure_count"),
    ))
    obb_valid = is_zero(metrics, "obb_overlap_count")
    if source_record is not None:
        states = ((row.get("generated") or {}).get("generated_agent_states") or [])
        posthoc = geometry_metrics(states, source_record, lane_half_width=default_lane_half_width)
        posthoc_overlap = as_number(posthoc.get("obb_overlap_count"))
        if posthoc_overlap is not None:
            obb_valid = posthoc_overlap <= 0.0
    if task == "v4a_forward_hv":
        sector_valid = is_zero(metrics, "generated_forward_violation_count")
    elif task == "directional_background":
        sector_valid = is_zero(metrics, "forward_violation_count")
    else:
        sector_valid = None

    required = [physical_valid, fov_valid, nearest_lane_valid, clearance_valid]
    if associated_lane_valid is not None:
        required.append(associated_lane_valid)
    if obb_valid is not None:
        required.append(obb_valid)
    if sector_valid is not None:
        required.append(sector_valid)
    return {
        "task": task,
        "physical_valid": physical_valid,
        "fov_valid": fov_valid,
        "nearest_lane_valid": nearest_lane_valid,
        "associated_lane_valid": associated_lane_valid,
        "clearance_valid": clearance_valid,
        "obb_nonoverlap": obb_valid,
        "sector_valid": sector_valid,
        "joint_feasible": all_known_true(required),
    }


FLAG_KEYS = (
    "physical_valid",
    "fov_valid",
    "nearest_lane_valid",
    "associated_lane_valid",
    "clearance_valid",
    "obb_nonoverlap",
    "sector_valid",
    "joint_feasible",
)


def rate(values):
    known = [value for value in values if value is not None]
    return {
        "known_count": len(known),
        "pass_count": sum(bool(value) for value in known),
        "pass_rate": sum(bool(value) for value in known) / len(known) if known else None,
    }


def summary(rows):
    flags = [row["flags"] for row in rows]
    return {
        "sample_count": len(rows),
        "task_counts": dict(sorted(defaultdict_count(row["flags"]["task"] for row in rows).items())),
        "raw_joint_feasibility": {
            key: rate([item[key] for item in flags]) for key in FLAG_KEYS
        },
    }


def defaultdict_count(values):
    counts = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return counts


def group_summary(rows, field):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        groups[str(value) if value is not None else "unknown"].append(row)
    return {name: summary(items) for name, items in sorted(groups.items())}


def source_features(source_record):
    if source_record is None:
        return {}
    condition = source_record.get("condition") or {}
    target = source_record.get("target") or {}
    states = target.get("generated_agent_states") or []
    distances, speeds = [], []
    for state in states:
        if not isinstance(state, (list, tuple)) or len(state) < 4:
            continue
        x, y, speed = as_number(state[0]), as_number(state[1]), as_number(state[3])
        if x is not None and y is not None:
            distances.append(math.hypot(x, y))
        if speed is not None:
            speeds.append(speed)
    return {
        "town": source_record.get("town"),
        "topology_bucket": (
            (source_record.get("topology_v4") or {}).get("bucket") or
            (source_record.get("count_features") or {}).get("map_bucket")
        ),
        "target_count": len(states),
        "observed_actor_count": len(condition.get("observed_agent_states") or []),
        "target_min_ego_distance_m": min(distances) if distances else None,
        "target_mean_speed_mps": sum(speeds) / len(speeds) if speeds else None,
    }


def bucket(value, boundaries):
    value = as_number(value)
    if value is None:
        return "unknown"
    for limit in boundaries:
        if value < limit:
            return "<%g" % limit
    return ">=%g" % boundaries[-1]


def normalize_row(row, default_lane_half_width, source_record=None):
    record_index = record_index_of(row)
    features = source_features(source_record)
    return {
        "record_index": record_index,
        "sample_index": row.get("sample_index", row.get("sample_idx")),
        "scene_id": row.get("scene_id"),
        "town": features.get("town") or row.get("town"),
        "topology_bucket": features.get("topology_bucket") or row.get("topology_bucket"),
        "target_count": features.get("target_count", row.get("target_count")),
        "observed_actor_count": features.get("observed_actor_count", row.get("observed_actor_count")),
        "target_distance_bucket": bucket(features.get("target_min_ego_distance_m"), (10, 20, 30)),
        "target_speed_bucket": bucket(features.get("target_mean_speed_mps"), (2, 5, 10)),
        "flags": flags_for_row(row, default_lane_half_width, source_record),
    }


def main():
    args = parse_args()
    names = [name for name, _ in args.input]
    if len(set(names)) != len(names):
        raise ValueError("--input names must be unique")
    source_inputs = dict(args.source_jsonl)
    if len(source_inputs) != len(args.source_jsonl):
        raise ValueError("--source_jsonl names must be unique")
    unknown_sources = set(source_inputs) - set(names)
    if unknown_sources:
        raise ValueError("--source_jsonl names missing from --input: %s" % sorted(unknown_sources))

    reports = {}
    for name, input_path in args.input:
        generated_rows = read_jsonl(input_path)
        source_rows = read_jsonl(source_inputs[name]) if name in source_inputs else None
        if source_rows is not None:
            invalid_indices = [
                record_index_of(row) for row in generated_rows
                if not isinstance(record_index_of(row), int) or
                record_index_of(row) < 0 or
                record_index_of(row) >= len(source_rows)
            ]
            if invalid_indices:
                raise ValueError("%s has generated rows without valid source record_index" % name)
        rows = [
            normalize_row(
                row,
                args.default_lane_half_width,
                None if source_rows is None else source_rows[record_index_of(row)],
            )
            for row in generated_rows
        ]
        reports[name] = {
            "input_jsonl": str(Path(input_path)),
            "source_jsonl": source_inputs.get(name),
            "summary": summary(rows),
            "by_town": group_summary(rows, "town"),
            "by_topology_bucket": group_summary(rows, "topology_bucket"),
            "by_target_count": group_summary(rows, "target_count"),
            "by_observed_actor_count": group_summary(rows, "observed_actor_count"),
            "by_target_distance_bucket": group_summary(rows, "target_distance_bucket"),
            "by_target_speed_bucket": group_summary(rows, "target_speed_bucket"),
        }

    payload = {
        "schema_version": "raw_generation_joint_feasibility_v1",
        "state_policy": "analysis_only_raw_outputs_no_clamp_no_rerank_no_rejection",
        "default_lane_half_width": args.default_lane_half_width,
        "reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
