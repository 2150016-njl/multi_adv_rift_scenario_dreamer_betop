"""Raw V4-A evaluation: full-FOV context and forward-HV target contract.

This evaluator never clips, lane-snaps, reranks, or rejects a sampled state.
Its directional and associated-lane measurements are diagnostics of one raw
sample.  The optional best-of-N block is explicitly an offline oracle and is
never an online acceptance policy.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_high_value_scene_dataset_v4 import audit_record
from high_value_init.evaluation_geometry import seed_everything, sha256_file
from high_value_init.high_value_ldm import match_state_metrics
from high_value_init.high_value_ldm_v3 import RouteConditionedHighValueLDMV3
from high_value_init.rift_dataset import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_generated_jsonl", default=None)
    parser.add_argument("--num_samples_per_condition", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=-1)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; each (record, sample) receives a stable derived seed.")
    parser.add_argument("--overwrite_output", action="store_true", default=False)
    return parser.parse_args()


def contract_args(args):
    return SimpleNamespace(
        fov=args.fov,
        forward_min_x=args.forward_min_x,
        target_fov_margin=args.target_fov_margin,
        max_num_agents=args.max_num_agents,
        max_target_lane_distance=args.max_target_lane_distance,
    )


def require_v4a_test_contract(records, args):
    check_args = contract_args(args)
    failures = []
    for index, record in enumerate(records):
        errors = audit_record(record, "test", check_args)
        if errors:
            failures.append({"index": index, "scene_id": record.get("scene_id"), "errors": errors})
            if len(failures) >= 20:
                break
    if failures:
        raise ValueError(json.dumps({"v4a_test_contract_failures": failures}, indent=2))


def segment_distance_and_heading(point, start, end):
    delta = end - start
    norm_sq = float(np.dot(delta, delta))
    if norm_sq <= 1e-9:
        return float(np.linalg.norm(point - start)), float("nan")
    ratio = float(np.clip(np.dot(point - start, delta) / norm_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * delta))), math.atan2(float(delta[1]), float(delta[0]))


def lane_distance_and_heading(point, lane):
    points = np.asarray((lane or {}).get("centerline") or [], dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    best = None
    for index in range(points.shape[0] - 1):
        distance, heading = segment_distance_and_heading(point, points[index, :2], points[index + 1, :2])
        if best is None or distance < best[0]:
            best = (distance, heading)
    return best


def abs_angle(delta):
    return abs(math.atan2(math.sin(delta), math.cos(delta)))


def raw_geometry(states, record, half_width, clearance):
    """Measure nearest-lane and clearance feasibility without altering states."""
    condition = record.get("condition") or {}
    lanes = (condition.get("map") or {}).get("lanes_v4") or []
    observed = condition.get("observed_agent_states") or []
    half_fov = float(condition.get("fov", 64.0)) / 2.0
    nearest_distances, nearest_heading_errors = [], []
    corridor_failures = ego_failures = observed_failures = pair_failures = off_fov_failures = 0
    for state in states:
        point = np.asarray(state[:2], dtype=np.float64)
        candidates = [lane_distance_and_heading(point, lane) for lane in lanes]
        candidates = [item for item in candidates if item is not None]
        nearest = min(candidates, key=lambda item: item[0]) if candidates else None
        distance = nearest[0] if nearest is not None else float("nan")
        nearest_distances.append(distance)
        if nearest is not None:
            nearest_heading_errors.append(abs_angle(float(state[2]) - nearest[1]))
        corridor_failures += int(math.isfinite(distance) and distance > half_width)
        ego_failures += int(float(np.linalg.norm(point)) < clearance)
        off_fov_failures += int(abs(float(point[0])) > half_fov or abs(float(point[1])) > half_fov)
        if observed:
            observed_failures += int(min(
                float(np.linalg.norm(point - np.asarray(row[:2], dtype=np.float64))) for row in observed
            ) < clearance)
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            pair_failures += int(np.linalg.norm(np.asarray(states[left][:2]) - np.asarray(states[right][:2])) < clearance)
    finite = [value for value in nearest_distances if math.isfinite(value)]
    return {
        "nearest_lane_distance_mean": float(np.mean(finite)) if finite else None,
        "nearest_lane_distance_max": float(np.max(finite)) if finite else None,
        "nearest_lane_heading_abs": float(np.mean(nearest_heading_errors)) if nearest_heading_errors else None,
        "nearest_lane_corridor_failure_count": corridor_failures,
        "ego_clearance_failure_count": ego_failures,
        "observed_clearance_failure_count": observed_failures,
        "pair_clearance_failure_count": pair_failures,
        "off_fov_failure_count": off_fov_failures,
    }


def associated_lane_geometry(states, target, record, default_half_width):
    """Measure against each target's training-time associated lane, not a new nearest lane."""
    if not states or target.numel() == 0:
        return {
            "associated_lane_distance_mean": None,
            "associated_lane_distance_max": None,
            "associated_lane_heading_abs": None,
            "associated_lane_corridor_failure_count": 0,
            "associated_lane_missing_count": 0,
        }
    prediction = torch.tensor(states, dtype=torch.float32)
    prediction = RouteConditionedHighValueLDMV3._match_prediction_order(prediction, target)
    lanes = ((record.get("condition") or {}).get("map") or {}).get("lanes_v4") or []
    associations = (record.get("target") or {}).get("map_associations_v4") or []
    distances, heading_errors = [], []
    corridor_failures = missing = 0
    for index in range(min(int(prediction.shape[0]), len(associations))):
        association = associations[index] or {}
        lane_index = association.get("lane_index")
        if lane_index is None or int(lane_index) < 0 or int(lane_index) >= len(lanes):
            missing += 1
            continue
        state = prediction[index].detach().cpu().tolist()
        measurement = lane_distance_and_heading(np.asarray(state[:2], dtype=np.float64), lanes[int(lane_index)])
        if measurement is None:
            missing += 1
            continue
        distance, heading = measurement
        distances.append(distance)
        heading_errors.append(abs_angle(float(state[2]) - heading))
        lane_half_width = float(association.get("lane_half_width", default_half_width))
        corridor_failures += int(distance > lane_half_width)
    return {
        "associated_lane_distance_mean": float(np.mean(distances)) if distances else None,
        "associated_lane_distance_max": float(np.max(distances)) if distances else None,
        "associated_lane_heading_abs": float(np.mean(heading_errors)) if heading_errors else None,
        "associated_lane_corridor_failure_count": corridor_failures,
        "associated_lane_missing_count": missing,
    }


def forward_geometry(states, args):
    limit = args.fov / 2.0 - args.target_fov_margin
    violations = 0
    for state in states:
        try:
            x, y = float(state[0]), float(state[1])
        except (IndexError, TypeError, ValueError):
            violations += 1
            continue
        violations += int(not math.isfinite(x) or not math.isfinite(y) or x < args.forward_min_x or x > limit or abs(y) > limit)
    count = len(states)
    return {
        "generated_forward_count": count - violations,
        "generated_forward_violation_count": violations,
        "generated_forward_violation_rate": float(violations / count) if count else None,
    }


def position_l1(states, target):
    if not states or target.numel() == 0:
        return float("nan")
    prediction = torch.tensor(states, dtype=torch.float32)
    prediction = RouteConditionedHighValueLDMV3._match_prediction_order(prediction, target)
    count = min(int(prediction.shape[0]), int(target.shape[0]))
    return float(torch.abs(prediction[:count, :2] - target[:count, :2]).mean().item())


def numeric(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


METRIC_KEYS = (
    "position_l1", "position_l2", "heading_abs", "speed_l1", "size_l1",
    "physical_invalid_count", "physical_invalid_rate", "fov_boundary_saturation_rate",
    "generated_forward_count", "generated_forward_violation_count", "generated_forward_violation_rate",
    "nearest_lane_distance_mean", "nearest_lane_distance_max", "nearest_lane_heading_abs",
    "nearest_lane_corridor_failure_count", "associated_lane_distance_mean", "associated_lane_distance_max",
    "associated_lane_heading_abs", "associated_lane_corridor_failure_count", "associated_lane_missing_count",
    "ego_clearance_failure_count", "observed_clearance_failure_count", "pair_clearance_failure_count",
    "off_fov_failure_count",
)


def summarize_rows(rows):
    return {"num_samples": len(rows), **{key: numeric(row["metrics"].get(key) for row in rows) for key in METRIC_KEYS}}


def group_rows(rows, field):
    groups = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        groups.setdefault(key, []).append(row)
    return {key: summarize_rows(group) for key, group in sorted(groups.items())}


def best_of_n_oracle(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["record_index"]), []).append(row)
    selected = []
    for candidates in grouped.values():
        valid = [
            row for row in candidates
            if row["metrics"].get("position_l2") is not None and math.isfinite(float(row["metrics"]["position_l2"]))
        ]
        if valid:
            selected.append(min(valid, key=lambda row: float(row["metrics"]["position_l2"])))
    return summarize_rows(selected)


def main():
    args = parse_args()
    if args.num_generate_agents != 2:
        raise ValueError("V4-A evaluation preserves the validated fixed-two-target task")
    if abs(args.fov - 64.0) > 1e-6:
        raise ValueError("V4-A evaluation requires --fov 64")
    if args.max_num_agents != 61:
        raise ValueError("V4-A evaluation requires --max_num_agents 61")
    if args.num_samples_per_condition < 1:
        raise ValueError("--num_samples_per_condition must be positive")
    output = Path(args.output_json)
    generated_output = Path(args.output_generated_jsonl) if args.output_generated_jsonl else None
    if not args.overwrite_output and (output.exists() or (generated_output is not None and generated_output.exists())):
        raise FileExistsError("refusing to overwrite evaluation evidence; choose new paths or pass --overwrite_output")
    records = read_jsonl(args.jsonl)
    if args.max_records > 0:
        records = records[:args.max_records]
    if not records:
        raise ValueError("no evaluation records")
    require_v4a_test_contract(records, args)
    model = RouteConditionedHighValueLDMV3(
        checkpoint_path=args.checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode="inpainting",
        num_generate_agents=2,
        route_num_points=80,
        default_lane_half_width=args.default_lane_half_width,
        min_initial_clearance=args.min_initial_clearance,
        clamp_generated_states=False,
        builder_cfg={"reject_observed_over_capacity": True},
    )
    rows = []
    for record_index, record in enumerate(records):
        target = torch.tensor((record.get("target") or {}).get("generated_agent_states") or [], dtype=torch.float32)
        for sample_index in range(args.num_samples_per_condition):
            sample_seed = seed_everything(args.seed, record_index, sample_index)
            generated = model.generate(record, 2)
            states = generated["generated_agent_states"]
            metrics = {
                **match_state_metrics(torch.tensor(states, dtype=torch.float32), target, bounds=model.generated_state_bounds),
                **raw_geometry(states, record, args.default_lane_half_width, args.min_initial_clearance),
                **associated_lane_geometry(states, target, record, args.default_lane_half_width),
                **forward_geometry(states, args),
            }
            metrics["position_l1"] = position_l1(states, target)
            rows.append({
                "record_index": record_index,
                "sample_index": sample_index,
                "sample_seed": sample_seed,
                "scene_id": record.get("scene_id"),
                "town": record.get("town"),
                "topology_bucket": (record.get("topology_v4") or {}).get("bucket"),
                "observed_actor_count": len((record.get("condition") or {}).get("observed_agent_states") or []),
                "generated": generated,
                "metrics": metrics,
            })
    summary = {
        "schema_version": "hv_generation_offline_eval_v4a_raw_v1",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "jsonl_sha256": sha256_file(args.jsonl),
        "base_seed": args.seed,
        "sample_seed_scheme": "(base_seed + 1000003*record_index + 9176*sample_index) mod 2^32",
        "data_contract": {
            "condition_region": "full_fov_non_target",
            "target_sector": "forward",
            "forward_min_x": args.forward_min_x,
            "target_fov_margin": args.target_fov_margin,
            "state_postprocessing": "none_raw_model_output",
        },
        "num_records": len(records),
        "num_samples_per_condition": args.num_samples_per_condition,
        "sampled_raw": summarize_rows(rows),
        "by_town": group_rows(rows, "town"),
        "by_topology_bucket": group_rows(rows, "topology_bucket"),
    }
    if args.num_samples_per_condition > 1:
        summary["best_of_n_position_l2_oracle_not_online_policy"] = best_of_n_oracle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if generated_output:
        write_jsonl(str(generated_output), rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
