"""Evaluate v3 raw outputs against v3 lane-association diagnostics."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from high_value_init.evaluation_geometry import geometry_metrics, position_l1, seed_everything, sha256_file
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
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; each (record, sample) receives a stable derived seed.")
    parser.add_argument("--overwrite_output", action="store_true", default=False)
    return parser.parse_args()


def numeric(values):
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not values: return {"mean": None, "p50": None, "p90": None, "max": None}
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)), "p90": float(np.percentile(values, 90)), "max": float(np.max(values))}


METRIC_KEYS = (
    "position_l1", "position_l2", "heading_abs", "speed_l1", "size_l1",
    "physical_invalid_count", "physical_invalid_rate", "fov_boundary_saturation_rate",
    "nearest_lane_distance_mean", "nearest_lane_distance_max", "lane_heading_abs",
    "associated_lane_distance_mean", "associated_lane_distance_max", "associated_lane_heading_abs",
    "associated_lane_unavailable_count", "lane_corridor_failure_count",
    "ego_clearance_failure_count", "observed_clearance_failure_count", "pair_clearance_failure_count",
    "off_fov_failure_count", "obb_axis_clearance_mean", "obb_axis_clearance_min", "obb_overlap_count",
    "ego_obb_overlap_count", "observed_obb_overlap_count", "pair_obb_overlap_count", "obb_invalid_state_count",
)


def summarize_rows(rows):
    return {
        "num_samples": len(rows),
        **{key: numeric(row["metrics"].get(key) for row in rows) for key in METRIC_KEYS},
    }


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
            if row["metrics"].get("position_l2") is not None and
            math.isfinite(float(row["metrics"]["position_l2"]))
        ]
        if valid:
            selected.append(min(valid, key=lambda row: float(row["metrics"]["position_l2"])))
    return summarize_rows(selected)


def main():
    args = parse_args()
    if args.num_generate_agents != 2:
        raise ValueError("v3 evaluation only supports the validated two-target high-value task")
    records = read_jsonl(args.jsonl)
    if args.max_records > 0: records = records[:args.max_records]
    if not records:
        raise ValueError(f"no records in {args.jsonl}")
    outputs = [Path(args.output_json)]
    if args.output_generated_jsonl:
        outputs.append(Path(args.output_generated_jsonl))
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite_output:
        raise FileExistsError(f"refusing to overwrite frozen evaluation outputs: {existing}; pass --overwrite_output explicitly")
    model = RouteConditionedHighValueLDMV3(
        checkpoint_path=args.checkpoint, autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint, device=args.device, mode="inpainting",
        num_generate_agents=args.num_generate_agents, route_num_points=80,
        default_lane_half_width=args.default_lane_half_width, min_initial_clearance=args.min_initial_clearance,
    )
    rows = []
    for record_index, record in enumerate(records):
        for sample_index in range(args.num_samples_per_condition):
            sample_seed = seed_everything(args.seed, record_index, sample_index)
            generated = model.generate(record, args.num_generate_agents)
            states = generated["generated_agent_states"]
            target = torch.tensor(
                (record.get("target") or {}).get("generated_agent_states") or [],
                dtype=torch.float32,
            )
            state_metrics = match_state_metrics(
                torch.tensor(states, dtype=torch.float32),
                target,
                bounds=model.generated_state_bounds,
            )
            metrics = {
                **state_metrics,
                **geometry_metrics(states, record, args.default_lane_half_width, args.min_initial_clearance),
            }
            metrics["position_l1"] = position_l1(states, target.detach().cpu().tolist())
            rows.append({"record_index": record_index, "sample_index": sample_index, "sample_seed": sample_seed,
                         "scene_id": record.get("scene_id"), "town": record.get("town"),
                         "topology_bucket": (record.get("topology_v3") or {}).get("bucket"),
                         "generated": generated, "metrics": metrics})
    summary = {
        "schema_version": "hv_generation_offline_eval_v3",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "jsonl_sha256": sha256_file(args.jsonl),
        "base_seed": args.seed,
        "sample_seed_scheme": "(base_seed + 1000003*record_index + 9176*sample_index) mod 2^32",
        "raw_state_postprocessing": "none",
        "num_records": len(records),
        "num_samples_per_condition": args.num_samples_per_condition,
        "sampled": summarize_rows(rows),
        "by_town": group_rows(rows, "town"),
        "by_topology_bucket": group_rows(rows, "topology_bucket"),
    }
    if args.num_samples_per_condition > 1:
        summary["best_of_n_position_l2_oracle"] = best_of_n_oracle(rows)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_generated_jsonl: write_jsonl(args.output_generated_jsonl, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
