#!/usr/bin/env python
"""Offline generation metrics for high-value initial-state scene generator."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from high_value_init.high_value_ldm import RouteConditionedHighValueLDM
from high_value_init.rift_dataset import read_jsonl, write_jsonl


def nanmean_metric(metrics: List[Dict], key: str) -> float:
    return float(np.nanmean([m.get(key, float("nan")) for m in metrics]))


def sum_metric(metrics: List[Dict], key: str) -> int:
    return int(np.nansum([m.get(key, 0) for m in metrics]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inpainting", "full_generation"], default="inpainting")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_generated_jsonl", default=None)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--route_num_points", type=int, default=80)
    parser.add_argument("--num_samples_per_condition", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_generated_speed", type=float, default=0.0)
    parser.add_argument("--max_generated_speed", type=float, default=30.0)
    parser.add_argument("--min_generated_length", type=float, default=2.0)
    parser.add_argument("--max_generated_length", type=float, default=8.0)
    parser.add_argument("--min_generated_width", type=float, default=1.4)
    parser.add_argument("--max_generated_width", type=float, default=3.2)
    parser.add_argument("--clamp_generated_states", action="store_true", default=False)
    return parser.parse_args()


def validate_args(args) -> None:
    if args.min_generated_speed > args.max_generated_speed:
        raise ValueError("--min_generated_speed must be <= --max_generated_speed")
    if args.min_generated_length > args.max_generated_length:
        raise ValueError("--min_generated_length must be <= --max_generated_length")
    if args.min_generated_width > args.max_generated_width:
        raise ValueError("--min_generated_width must be <= --max_generated_width")


def main():
    args = parse_args()
    validate_args(args)
    records = read_jsonl(args.jsonl)
    if args.max_records and args.max_records > 0:
        records = records[:args.max_records]
    if not records:
        raise ValueError(f"no records in {args.jsonl}")

    model = RouteConditionedHighValueLDM(
        checkpoint_path=args.checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode=args.mode,
        num_generate_agents=args.num_generate_agents,
        route_num_points=args.route_num_points,
        min_generated_speed=args.min_generated_speed,
        max_generated_speed=args.max_generated_speed,
        min_generated_length=args.min_generated_length,
        max_generated_length=args.max_generated_length,
        min_generated_width=args.min_generated_width,
        max_generated_width=args.max_generated_width,
        clamp_generated_states=args.clamp_generated_states,
    )

    rows: List[Dict] = []
    metrics = []
    for record in records:
        best = None
        samples = []
        for sample_idx in range(args.num_samples_per_condition):
            out = model.reconstruct_targets(record)
            sample = {
                "scene_id": record.get("scene_id"),
                "sample_idx": sample_idx,
                "generated": out["generated"],
                "metrics": out["metrics"],
            }
            samples.append(sample)
            pos = out["metrics"]["position_l2"]
            if best is None or pos < best["metrics"]["position_l2"]:
                best = sample
        rows.extend(samples)
        metrics.append(best["metrics"])

    summary = {
        "schema_version": "hv_init_offline_eval_v1",
        "mode": args.mode,
        "num_records": len(records),
        "num_samples_per_condition": args.num_samples_per_condition,
        "generated_state_bounds": {
            "speed": [args.min_generated_speed, args.max_generated_speed],
            "length": [args.min_generated_length, args.max_generated_length],
            "width": [args.min_generated_width, args.max_generated_width],
        },
        "clamp_generated_states": args.clamp_generated_states,
        "best_of_n_position_l2": nanmean_metric(metrics, "position_l2"),
        "best_of_n_heading_abs": nanmean_metric(metrics, "heading_abs"),
        "best_of_n_speed_l1": nanmean_metric(metrics, "speed_l1"),
        "best_of_n_size_l1": nanmean_metric(metrics, "size_l1"),
        "best_of_n_length_l1": nanmean_metric(metrics, "length_l1"),
        "best_of_n_width_l1": nanmean_metric(metrics, "width_l1"),
        "physical_invalid_count": sum_metric(metrics, "physical_invalid_count"),
        "physical_invalid_rate": nanmean_metric(metrics, "physical_invalid_rate"),
        "fov_boundary_saturation_rate": nanmean_metric(metrics, "fov_boundary_saturation_rate"),
        "note": "RIFT labels are not inference inputs; RIFT pass rate must be measured by RIFT/CARLA audit.",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.output_generated_jsonl:
        write_jsonl(args.output_generated_jsonl, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
