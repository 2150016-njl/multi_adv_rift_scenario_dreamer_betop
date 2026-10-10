"""Evaluate raw V4C outputs plus target-only RIFT route consistency.

The evaluator projects the raw generated preview point onto the fixed RIFT
trace_route serialized for that target. This is a static offline proxy for the
same route-distance relation used by RIFT, not an online state edit. Closed-
loop RIFT recognition remains the final interaction validation.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

import eval_high_value_scene_generator_v4a as v4a
from high_value_init.evaluation_geometry import seed_everything, sha256_file
from high_value_init.high_value_ldm import match_state_metrics
from high_value_init.high_value_ldm_v3 import RouteConditionedHighValueLDMV3
from high_value_init.rift_dataset import read_jsonl, write_jsonl


EXTRA_METRICS = (
    "interaction_anchor_along_abs_error_m", "interaction_anchor_lateral_abs_error_m",
    "interaction_anchor_relation_l1_m", "interaction_anchor_metadata_missing_count",
    "interaction_route_distance_abs_error_m", "interaction_route_sync_abs_gap_m",
    "interaction_route_sync_excess_over_threshold_m", "interaction_route_sync_within_threshold_rate",
    "interaction_route_preview_lateral_error_m", "interaction_route_heading_tangent_error",
    "interaction_route_progress_metadata_missing_count",
)


def require_v4c_test_contract(records, args):
    failures = []
    check_args = v4a.contract_args(args)
    for index, record in enumerate(records):
        errors = v4a.audit_record(record, "test", check_args)
        target = record.get("target") or {}
        supervision = list(target.get("route_interaction_supervision") or [])
        condition_text = json.dumps(record.get("condition") or {}, ensure_ascii=False)
        if record.get("dataset_version") != "v4c_route_interaction_supervised":
            errors.append("dataset_version_mismatch")
        if len(supervision) != len(target.get("generated_agent_states") or []):
            errors.append("route_interaction_target_count_mismatch")
        for item in supervision:
            if (len((item or {}).get("target_route_polyline_local_xy") or []) < 2 or
                    (item or {}).get("target_route_distance_offset_m") is None or
                    (item or {}).get("rift_preview_distance_m") is None or
                    (item or {}).get("planner_distance_abs_error_to_rift_label_m") is None):
                errors.append("route_progress_supervision_missing")
                break
        if any(marker in condition_text for marker in (
                "interaction_waypoint", "conflict_point", "route_distance_difference",
                "target_route_distance_to_conflict", "interaction_strength")):
            errors.append("route_interaction_label_leaked_to_condition")
        if errors:
            failures.append({"index": index, "scene_id": record.get("scene_id"), "errors": sorted(set(errors))})
            if len(failures) >= 20:
                break
    if failures:
        raise ValueError(json.dumps({"v4c_test_contract_failures": failures}, indent=2))


def anchor_geometry(states, target, record):
    if not states or target.numel() == 0:
        return {key: None for key in EXTRA_METRICS}
    prediction = torch.tensor(states, dtype=torch.float32)
    prediction = RouteConditionedHighValueLDMV3._match_prediction_order(prediction, target)
    supervision = list((record.get("target") or {}).get("route_interaction_supervision") or [])
    along, lateral, missing = [], [], 0
    for index in range(min(int(prediction.shape[0]), len(supervision))):
        item = supervision[index] or {}
        anchor = item.get("conflict_point_local_xy")
        try:
            anchor_x, anchor_y = float(anchor[0]), float(anchor[1])
            expected_along = float(item["target_to_conflict_along_heading_m"])
            expected_lateral = float(item["target_to_conflict_lateral_heading_m"])
            state = prediction[index].detach().cpu().numpy()
        except (IndexError, KeyError, TypeError, ValueError):
            missing += 1
            continue
        dx, dy = anchor_x - float(state[0]), anchor_y - float(state[1])
        heading = float(state[2])
        predicted_along = dx * math.cos(heading) + dy * math.sin(heading)
        predicted_lateral = -dx * math.sin(heading) + dy * math.cos(heading)
        along.append(abs(predicted_along - expected_along))
        lateral.append(abs(predicted_lateral - expected_lateral))
    relation = along + lateral
    return {
        "interaction_anchor_along_abs_error_m": float(np.mean(along)) if along else None,
        "interaction_anchor_lateral_abs_error_m": float(np.mean(lateral)) if lateral else None,
        "interaction_anchor_relation_l1_m": float(np.mean(relation)) if relation else None,
        "interaction_anchor_metadata_missing_count": missing,
    }


def route_remaining_distance(point, polyline, heading_rad, preview_distance):
    if len(polyline) < 2:
        return None
    point = np.asarray(point[:2], dtype=np.float64) + float(preview_distance) * np.asarray(
        [math.cos(float(heading_rad)), math.sin(float(heading_rad))], dtype=np.float64,
    )
    points = np.asarray([row[:2] for row in polyline], dtype=np.float64)
    starts, ends = points[:-1], points[1:]
    delta = ends - starts
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > 1e-5
    if not np.any(valid):
        return None
    ratios = np.zeros_like(lengths)
    ratios[valid] = np.clip(
        np.sum((point - starts[valid]) * delta[valid], axis=1) / np.sum(delta[valid] ** 2, axis=1), 0.0, 1.0,
    )
    projections = starts + ratios[:, None] * delta
    distance_sq = np.sum((point - projections) ** 2, axis=1)
    distance_sq[~valid] = np.inf
    index = int(np.argmin(distance_sq))
    tangent = delta[index] / max(float(lengths[index]), 1e-5)
    heading = np.asarray([math.cos(float(heading_rad)), math.sin(float(heading_rad))], dtype=np.float64)
    return (
        float(np.sum(lengths[index + 1:]) + (1.0 - ratios[index]) * lengths[index]),
        float(math.sqrt(max(float(distance_sq[index]), 0.0))),
        float(1.0 - np.clip(float(np.dot(heading, tangent)), -1.0, 1.0)),
    )


def route_progress_geometry(states, target, record):
    if not states or target.numel() == 0:
        return {key: None for key in EXTRA_METRICS[4:]}
    prediction = torch.tensor(states, dtype=torch.float32)
    prediction = RouteConditionedHighValueLDMV3._match_prediction_order(prediction, target)
    supervision = list((record.get("target") or {}).get("route_interaction_supervision") or [])
    distance_errors, sync_gaps, sync_excess, within, corridor_errors, tangent_errors, missing = [], [], [], [], [], [], 0
    for index in range(min(int(prediction.shape[0]), len(supervision))):
        item = supervision[index] or {}
        try:
            route_projection = route_remaining_distance(
                prediction[index].detach().cpu().tolist(), item["target_route_polyline_local_xy"],
                float(prediction[index, 2].detach().cpu()), float(item.get("rift_preview_distance_m", 2.0)),
            )
            offset = float(item["target_route_distance_offset_m"])
            target_distance = float(item["target_route_distance_to_conflict_m"])
            ego_distance = float(item["ego_route_distance_to_conflict_m"])
            threshold = float(item["route_distance_difference_threshold_m"])
        except (KeyError, TypeError, ValueError):
            missing += 1
            continue
        if route_projection is None or not all(math.isfinite(value) for value in (offset, target_distance, ego_distance, threshold)):
            missing += 1
            continue
        remaining, corridor_error, tangent_error = route_projection
        predicted_distance = remaining + offset
        gap = abs(predicted_distance - ego_distance)
        distance_errors.append(abs(predicted_distance - target_distance))
        sync_gaps.append(gap)
        sync_excess.append(max(0.0, gap - threshold))
        within.append(float(gap <= threshold))
        corridor_errors.append(corridor_error)
        tangent_errors.append(tangent_error)
    return {
        "interaction_route_distance_abs_error_m": float(np.mean(distance_errors)) if distance_errors else None,
        "interaction_route_sync_abs_gap_m": float(np.mean(sync_gaps)) if sync_gaps else None,
        "interaction_route_sync_excess_over_threshold_m": float(np.mean(sync_excess)) if sync_excess else None,
        "interaction_route_sync_within_threshold_rate": float(np.mean(within)) if within else None,
        "interaction_route_preview_lateral_error_m": float(np.mean(corridor_errors)) if corridor_errors else None,
        "interaction_route_heading_tangent_error": float(np.mean(tangent_errors)) if tangent_errors else None,
        "interaction_route_progress_metadata_missing_count": missing,
    }


def numeric(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)), "max": float(np.max(values))}


def summarize_rows(rows):
    keys = tuple(v4a.METRIC_KEYS) + EXTRA_METRICS
    return {"num_samples": len(rows), **{key: numeric(row["metrics"].get(key) for row in rows) for key in keys}}


def group_rows(rows, field):
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get(field) or "unknown"), []).append(row)
    return {key: summarize_rows(value) for key, value in sorted(groups.items())}


def main():
    args = v4a.parse_args()
    if args.num_generate_agents != 2 or abs(args.fov - 64.0) > 1e-6 or args.max_num_agents != 61:
        raise ValueError("V4C evaluation requires the fixed-two 64m V4 contract")
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
    require_v4c_test_contract(records, args)
    model = RouteConditionedHighValueLDMV3(
        checkpoint_path=args.checkpoint, autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint, device=args.device, mode="inpainting",
        num_generate_agents=2, route_num_points=80,
        default_lane_half_width=args.default_lane_half_width,
        min_initial_clearance=args.min_initial_clearance,
        clamp_generated_states=False, builder_cfg={"reject_observed_over_capacity": True},
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
                **v4a.raw_geometry(states, record, args.default_lane_half_width, args.min_initial_clearance),
                **v4a.associated_lane_geometry(states, target, record, args.default_lane_half_width),
                **v4a.forward_geometry(states, args),
                **anchor_geometry(states, target, record),
                **route_progress_geometry(states, target, record),
            }
            metrics["position_l1"] = v4a.position_l1(states, target)
            rows.append({
                "record_index": record_index, "sample_index": sample_index, "sample_seed": sample_seed,
                "scene_id": record.get("scene_id"), "town": record.get("town"),
                "topology_bucket": (record.get("topology_v4") or {}).get("bucket"),
                "observed_actor_count": len((record.get("condition") or {}).get("observed_agent_states") or []),
                "generated": generated, "metrics": metrics,
            })
    summary = {
        "schema_version": "hv_generation_offline_eval_v4c_raw_anchor_geometry_v1",
        "checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(args.checkpoint),
        "jsonl_sha256": sha256_file(args.jsonl), "base_seed": args.seed,
        "sample_seed_scheme": "(base_seed + 1000003*record_index + 9176*sample_index) mod 2^32",
        "data_contract": {
            "condition_region": "full_fov_non_target", "target_sector": "forward",
            "route_interaction_supervision": "target_only_offline", "state_postprocessing": "none_raw_model_output",
        },
        "metric_caveat": (
            "interaction_route_* projects raw output onto the target-specific RIFT trace_route polyline "
            "and compares remaining route length to the original RIFT distance labels. It is static/offline "
            "evidence, not an online state edit or a replacement for normal CBV recognition."
        ),
        "num_records": len(records), "num_samples_per_condition": args.num_samples_per_condition,
        "sampled_raw": summarize_rows(rows), "by_town": group_rows(rows, "town"),
        "by_topology_bucket": group_rows(rows, "topology_bucket"),
    }
    if args.num_samples_per_condition > 1:
        summary["best_of_n_position_l2_oracle_not_online_policy"] = v4a.best_of_n_oracle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if generated_output:
        write_jsonl(str(generated_output), rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
