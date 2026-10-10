#!/usr/bin/env python
"""Check that serialized ground-truth targets satisfy evaluator geometry.

This is a prerequisite for changing lane-related loss weights.  If a target's
own associated-lane measurement is inconsistent, a model cannot learn a
meaningful associated-lane objective from that record.  The script is read
only and does not modify a dataset split or target state.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from high_value_init.evaluation_geometry import geometry_metrics, nearest_lane
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--kind", choices=("auto", "hv", "v4a", "background"), default="auto")
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def numeric(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    values.sort()

    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = int(index), min(int(index) + 1, len(values) - 1)
        return values[lo] * (1.0 - (index - lo)) + values[hi] * (index - lo)

    return {"mean": sum(values) / len(values), "p50": percentile(0.5), "p90": percentile(0.9), "max": values[-1]}


def kind_for(record, requested):
    if requested != "auto":
        return requested
    schema = str(record.get("schema_version") or "")
    if "background" in schema:
        return "background"
    if "v4a" in schema:
        return "v4a"
    return "hv"


def target_states(record):
    return list((record.get("target") or {}).get("generated_agent_states") or [])


def associated_lane_metrics(record, states, default_half_width):
    condition = record.get("condition") or {}
    map_payload = condition.get("map") or {}
    target = record.get("target") or {}
    # Lane indices are representation-specific.  Never pair V3 associations
    # with V4 lanes merely because both serializations happen to be present.
    if target.get("map_associations_v4"):
        associations = target.get("map_associations_v4") or []
        lanes = map_payload.get("lanes_v4") or []
    else:
        associations = target.get("map_associations_v3") or []
        lanes = map_payload.get("lanes_v3") or []
    if not associations:
        return {
            "associated_lane_available": False,
            "associated_lane_missing_count": len(states),
            "associated_lane_distance_mean": None,
            "associated_lane_corridor_failure_count": None,
        }
    distances = []
    missing = 0
    failures = 0
    for state, association in zip(states, associations):
        try:
            point = np.asarray([float(state[0]), float(state[1])], dtype=np.float64)
            lane_index = int((association or {}).get("lane_index"))
            lane = lanes[lane_index]
        except (IndexError, TypeError, ValueError):
            missing += 1
            continue
        measurement = nearest_lane(point, [lane])
        if measurement is None:
            missing += 1
            continue
        distance = float(measurement[0])
        distances.append(distance)
        half_width = float((association or {}).get("lane_half_width", default_half_width))
        failures += int(distance > half_width)
    missing += max(0, len(states) - len(associations))
    return {
        "associated_lane_available": True,
        "associated_lane_missing_count": missing,
        "associated_lane_distance_mean": sum(distances) / len(distances) if distances else None,
        "associated_lane_corridor_failure_count": failures,
    }


def forward_violation(state, args, limit):
    try:
        x, y = float(state[0]), float(state[1])
    except (IndexError, TypeError, ValueError):
        return True
    return (
        not math.isfinite(x) or not math.isfinite(y) or
        x < args.forward_min_x or x > limit or abs(y) > limit
    )


def target_row(record, kind, args):
    states = target_states(record)
    geometry = geometry_metrics(
        states,
        record,
        lane_half_width=args.default_lane_half_width,
        min_center_clearance=args.min_initial_clearance,
        fov=args.fov,
    )
    associated = associated_lane_metrics(record, states, args.default_lane_half_width)
    limit = args.fov / 2.0 - args.target_fov_margin
    if kind in ("v4a", "background"):
        forward_violations = sum(int(forward_violation(state, args, limit)) for state in states)
    else:
        forward_violations = None
    return {
        "town": str(record.get("town") or "unknown"),
        "topology_bucket": str(
            (record.get("topology_v4") or record.get("topology_v3") or record.get("count_features") or {}).get("bucket") or
            (record.get("count_features") or {}).get("map_bucket") or "unknown"
        ),
        "target_count": len(states),
        "nearest_lane_distance_mean": geometry.get("nearest_lane_distance_mean"),
        "nearest_lane_corridor_failure_count": geometry.get("lane_corridor_failure_count"),
        "off_fov_failure_count": geometry.get("off_fov_failure_count"),
        "ego_clearance_failure_count": geometry.get("ego_clearance_failure_count"),
        "observed_clearance_failure_count": geometry.get("observed_clearance_failure_count"),
        "pair_clearance_failure_count": geometry.get("pair_clearance_failure_count"),
        "obb_overlap_count": geometry.get("obb_overlap_count"),
        "forward_target_violation_count": forward_violations,
        **associated,
    }


METRICS = (
    "nearest_lane_distance_mean",
    "nearest_lane_corridor_failure_count",
    "associated_lane_missing_count",
    "associated_lane_distance_mean",
    "associated_lane_corridor_failure_count",
    "off_fov_failure_count",
    "ego_clearance_failure_count",
    "observed_clearance_failure_count",
    "pair_clearance_failure_count",
    "obb_overlap_count",
    "forward_target_violation_count",
)


def summarize(rows):
    return {
        "record_count": len(rows),
        **{key: numeric(row.get(key) for row in rows) for key in METRICS},
        "associated_lane_available_record_count": sum(bool(row["associated_lane_available"]) for row in rows),
    }


def grouped(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {key: summarize(value) for key, value in sorted(groups.items())}


def main():
    args = parse_args()
    records = read_jsonl(args.jsonl)
    if not records:
        raise ValueError("no records in %s" % args.jsonl)
    kinds = {kind_for(record, args.kind) for record in records}
    if len(kinds) != 1:
        raise ValueError("mixed task contracts; pass one dataset per audit")
    kind = next(iter(kinds))
    rows = [target_row(record, kind, args) for record in records]
    report = {
        "schema_version": "ground_truth_target_geometry_audit_v1",
        "input_jsonl": args.jsonl,
        "task_kind": kind,
        "policy": "measure_ground_truth_only_no_dataset_or_state_mutation",
        "summary": summarize(rows),
        "by_town": grouped(rows, "town"),
        "by_topology_bucket": grouped(rows, "topology_bucket"),
        "by_target_count": grouped(rows, "target_count"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
