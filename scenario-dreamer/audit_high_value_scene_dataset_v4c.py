"""Audit V4C target-only RIFT route-interaction supervision."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from audit_high_value_scene_dataset_v4 import audit_record
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--min_target_obb_axis_clearance", type=float, default=0.5)
    parser.add_argument("--max_sidecar_distance_abs_error_m", type=float, default=2.0)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def numeric(values):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p90": None, "max": None}
    def percentile(q):
        return values[min(len(values) - 1, int(round((len(values) - 1) * q)))]
    return {
        "count": len(values), "min": values[0], "mean": sum(values) / len(values),
        "p50": percentile(.5), "p90": percentile(.9), "max": values[-1],
    }


def condition_has_route_interaction_label(record):
    text = json.dumps(record.get("condition") or {}, ensure_ascii=False)
    markers = (
        "interaction_waypoint", "conflict_point", "route_distance_difference",
        "target_route_distance_to_conflict", "interaction_strength",
    )
    return any(marker in text for marker in markers)


def route_remaining_distance(point, heading, preview_distance, polyline):
    if len(polyline) < 2:
        return None
    px = float(point[0]) + float(preview_distance) * math.cos(float(heading))
    py = float(point[1]) + float(preview_distance) * math.sin(float(heading))
    lengths, total = [], 0.0
    for start, end in zip(polyline[:-1], polyline[1:]):
        length = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
        lengths.append(length)
        total += length
    best, traversed = None, 0.0
    for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:])):
        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        norm_sq = dx * dx + dy * dy
        if norm_sq <= 1e-12:
            traversed += lengths[index]
            continue
        ratio = min(1.0, max(0.0, ((px - float(start[0])) * dx + (py - float(start[1])) * dy) / norm_sq))
        qx, qy = float(start[0]) + ratio * dx, float(start[1]) + ratio * dy
        distance_sq = (px - qx) ** 2 + (py - qy) ** 2
        progress = traversed + ratio * lengths[index]
        if best is None or distance_sq < best[0]:
            best = (distance_sq, progress)
        traversed += lengths[index]
    return None if best is None else total - best[1]


def audit_split(name, rows, args):
    errors, scores, diffs, thresholds, anchor_along_errors, anchor_lateral_errors, route_polyline_counts, route_offsets, planner_distance_errors, route_reconstruction_errors, route_sync_errors = [], [], [], [], [], [], [], [], [], [], []
    for index, record in enumerate(rows):
        record_errors = audit_record(record, name, args)
        if record.get("dataset_version") != "v4c_route_interaction_supervised":
            record_errors.append("dataset_version_mismatch")
        if (record.get("v4b_contract") or {}).get("target_pair_selection") != "high_value_pair":
            record_errors.append("target_pair_policy_mismatch")
        try:
            pair_clearance = float((record.get("v4b_contract") or {}).get("target_pair_obb_axis_clearance"))
        except (TypeError, ValueError):
            pair_clearance = float("-inf")
        if pair_clearance < args.min_target_obb_axis_clearance:
            record_errors.append("target_pair_obb_clearance_below_training_contract")
        contract = record.get("v4c_contract") or {}
        if contract.get("supervision") != "rift_conflict_point_target_only":
            record_errors.append("v4c_contract_missing")
        if condition_has_route_interaction_label(record):
            record_errors.append("route_interaction_label_leaked_to_condition")
        target = record.get("target") or {}
        states = list(target.get("generated_agent_states") or [])
        supervision = list(target.get("route_interaction_supervision") or [])
        if len(states) != 2 or len(supervision) != len(states):
            record_errors.append("route_interaction_target_count_mismatch")
        for state, item in zip(states, supervision):
            item = item or {}
            anchor = item.get("conflict_point_local_xy")
            required = (
                "ego_route_distance_to_conflict_m", "target_route_distance_to_conflict_m",
                "route_distance_difference_m", "route_distance_difference_threshold_m",
                "interaction_strength", "target_to_conflict_along_heading_m",
                "target_to_conflict_lateral_heading_m",
            )
            if not isinstance(anchor, (list, tuple)) or len(anchor) < 2 or any(key not in item for key in required):
                record_errors.append("route_interaction_supervision_missing_field")
                continue
            try:
                values = [float(anchor[0]), float(anchor[1])] + [float(item[key]) for key in required]
                x, y, heading = float(state[0]), float(state[1]), float(state[2])
            except (IndexError, TypeError, ValueError):
                record_errors.append("route_interaction_supervision_non_numeric")
                continue
            if not all(math.isfinite(value) for value in values + [x, y, heading]):
                record_errors.append("route_interaction_supervision_non_finite")
                continue
            ego_distance, target_distance, distance_diff, threshold, score, saved_along, saved_lateral = values[2:]
            if ego_distance < 0 or target_distance < 0 or distance_diff < 0 or threshold <= 0 or not 0 <= score <= 1:
                record_errors.append("route_interaction_supervision_invalid_range")
                continue
            expected_score = max(0.0, 1.0 - distance_diff / threshold)
            if abs(expected_score - score) > 1e-5:
                record_errors.append("route_interaction_strength_inconsistent")
            dx, dy = values[0] - x, values[1] - y
            expected_along = dx * math.cos(heading) + dy * math.sin(heading)
            expected_lateral = -dx * math.sin(heading) + dy * math.cos(heading)
            anchor_along_errors.append(abs(expected_along - saved_along))
            anchor_lateral_errors.append(abs(expected_lateral - saved_lateral))
            if abs(expected_along - saved_along) > 1e-4 or abs(expected_lateral - saved_lateral) > 1e-4:
                record_errors.append("target_anchor_geometry_inconsistent")
            scores.append(score); diffs.append(distance_diff); thresholds.append(threshold)
            polyline = item.get("target_route_polyline_local_xy") or []
            try:
                offset = float(item["target_route_distance_offset_m"])
                preview_distance = float(item["rift_preview_distance_m"])
                planner_distance_error = float(item["planner_distance_abs_error_to_rift_label_m"])
            except (KeyError, TypeError, ValueError):
                record_errors.append("route_progress_supervision_missing")
                continue
            if (len(polyline) < 2 or not math.isfinite(offset) or
                    not math.isfinite(preview_distance) or preview_distance <= 0.0 or
                    not math.isfinite(planner_distance_error) or planner_distance_error < 0.0):
                record_errors.append("route_progress_supervision_invalid")
                continue
            route_polyline_counts.append(len(polyline))
            route_offsets.append(offset)
            planner_distance_errors.append(planner_distance_error)
            if planner_distance_error > args.max_sidecar_distance_abs_error_m:
                record_errors.append("sidecar_distance_error_above_limit")
            reconstructed_remaining = route_remaining_distance(state[:2], heading, preview_distance, polyline)
            if reconstructed_remaining is None:
                record_errors.append("route_progress_ground_truth_projection_missing")
                continue
            reconstructed_distance = reconstructed_remaining + offset
            route_reconstruction_error = abs(reconstructed_distance - target_distance)
            route_sync_error = abs(abs(reconstructed_distance - ego_distance) - distance_diff)
            route_reconstruction_errors.append(route_reconstruction_error)
            route_sync_errors.append(route_sync_error)
            if route_reconstruction_error > 1e-4 or route_sync_error > 1e-4:
                record_errors.append("route_progress_ground_truth_reconstruction_inconsistent")
        if record_errors and len(errors) < 30:
            errors.append({"index": index, "scene_id": record.get("scene_id"), "errors": sorted(set(record_errors))})
    return {
        "records": len(rows), "route_interaction_targets": len(scores),
        "interaction_strength": numeric(scores),
        "route_distance_difference_m": numeric(diffs),
        "route_distance_difference_threshold_m": numeric(thresholds),
        "anchor_along_recompute_abs_error_m": numeric(anchor_along_errors),
        "anchor_lateral_recompute_abs_error_m": numeric(anchor_lateral_errors),
        "target_route_polyline_point_count": numeric(route_polyline_counts),
        "target_route_distance_offset_m": numeric(route_offsets),
        "planner_distance_abs_error_to_rift_label_m": numeric(planner_distance_errors),
        "ground_truth_route_distance_reconstruction_abs_error_m": numeric(route_reconstruction_errors),
        "ground_truth_route_sync_gap_abs_error_m": numeric(route_sync_errors),
        "errors": errors,
    }


def main():
    args = parse_args()
    splits = {
        "train": read_jsonl(args.train_jsonl),
        "val": read_jsonl(args.val_jsonl),
        "test": read_jsonl(args.test_jsonl),
    }
    if any(not rows for rows in splits.values()):
        raise ValueError("V4C audit requires non-empty train, val, and test JSONL files")
    summaries = {name: audit_split(name, rows, args) for name, rows in splits.items()}
    memberships = defaultdict(set)
    for name, rows in splits.items():
        for row in rows:
            memberships[str(row.get("episode_key"))].add(name)
    leakage = sorted(key for key, values in memberships.items() if len(values) > 1)
    failures = {name: summary["errors"] for name, summary in summaries.items() if summary["errors"]}
    report = {
        "schema_version": "hv_scene_dataset_v4c_route_interaction_audit_v1",
        "passed": not failures and not leakage,
        "splits": summaries,
        "episode_split_leakage_examples": leakage[:30],
        "args": vars(args),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
