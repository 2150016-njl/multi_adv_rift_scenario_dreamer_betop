#!/usr/bin/env python
"""Raw single-sample/oracle evaluation of variable-count ordinary-BV models."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit_scenario_dreamer_background_dataset import audit_record
from high_value_init.background_ldm import RouteConditionedBackgroundLDM
from high_value_init.evaluation_geometry import geometry_metrics, position_l1, seed_everything, sha256_file
from high_value_init.high_value_ldm import match_state_metrics
from high_value_init.rift_dataset import read_jsonl, write_jsonl


def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--jsonl", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--autoencoder_checkpoint", required=True); p.add_argument("--ldm_checkpoint", required=True)
    p.add_argument("--output_json", required=True); p.add_argument("--output_generated_jsonl")
    p.add_argument("--num_samples_per_condition", type=int, default=1); p.add_argument("--max_records", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda"); p.add_argument("--overwrite_output", action="store_true")
    p.add_argument("--max_target_agents", type=int, default=8); p.add_argument("--fov", type=float, default=64.0)
    p.add_argument("--partition_boundary_x", type=float, default=0.0); p.add_argument("--target_fov_margin", type=float, default=2.0)
    p.add_argument("--max_num_agents", type=int, default=61); p.add_argument("--default_lane_half_width", type=float, default=1.75)
    p.add_argument("--min_initial_clearance", type=float, default=2.5); return p.parse_args()


def numeric(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {"mean": float(np.mean(values)) if values else None, "p90": float(np.percentile(values, 90)) if values else None,
            "max": float(np.max(values)) if values else None}


def stratified(rows, field, keys):
    """Report raw metrics by a serialized metadata field without selection."""
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    return {
        group: {"num_samples": len(items), **{
            key: numeric(item["metrics"].get(key) for item in items) for key in keys
        }}
        for group, items in sorted(groups.items())
    }


def main():
    args = parse_args(); records = read_jsonl(args.jsonl)
    if args.max_records > 0: records = records[:args.max_records]
    if not records: raise ValueError("no background evaluation records")
    outputs = [Path(args.output_json)] + ([Path(args.output_generated_jsonl)] if args.output_generated_jsonl else [])
    if any(path.exists() for path in outputs) and not args.overwrite_output: raise FileExistsError("refusing to overwrite raw background evaluation output")
    errors = []
    for index, record in enumerate(records):
        check = audit_record(record, record.get("split", "test"), args)
        if check: errors.append({"index": index, "errors": check})
    if errors: raise ValueError(json.dumps({"background_eval_contract_failures": errors[:20]}, indent=2))
    model = RouteConditionedBackgroundLDM(checkpoint_path=args.checkpoint, autoencoder_checkpoint=args.autoencoder_checkpoint,
                                          ldm_checkpoint=args.ldm_checkpoint, device=args.device, mode="inpainting", num_generate_agents=1,
                                          route_num_points=80, builder_cfg={"reject_observed_over_capacity": True})
    rows = []
    for record_index, record in enumerate(records):
        target_count = int(record["condition"]["num_generate_agents"])
        for sample_index in range(args.num_samples_per_condition):
            sample_seed = seed_everything(args.seed, record_index, sample_index)
            generated = model.generate(record, target_count); states = generated["generated_agent_states"]
            target = torch.tensor(record["target"]["generated_agent_states"], dtype=torch.float32)
            metrics = {**match_state_metrics(torch.tensor(states, dtype=torch.float32), target, bounds=model.generated_state_bounds),
                       **geometry_metrics(states, record, args.default_lane_half_width, args.min_initial_clearance, args.fov)}
            metrics["position_l1"] = position_l1(states, target.tolist())
            metrics["target_count_exact"] = int(len(states) == target_count)
            metrics["forward_violation_count"] = sum(int(float(s[0]) < args.partition_boundary_x or float(s[0]) > args.fov / 2.0 - args.target_fov_margin or abs(float(s[1])) > args.fov / 2.0 - args.target_fov_margin) for s in states)
            rows.append({"record_index": record_index, "sample_index": sample_index, "sample_seed": sample_seed, "scene_id": record.get("scene_id"),
                         "town": record.get("town"), "topology_bucket": record["count_features"]["map_bucket"], "target_count": target_count,
                         "generated": generated, "metrics": metrics})
    keys = ("position_l1", "position_l2", "physical_invalid_rate", "nearest_lane_distance_mean", "associated_lane_distance_mean",
            "lane_corridor_failure_count", "off_fov_failure_count", "forward_violation_count", "obb_overlap_count", "obb_axis_clearance_min")
    summary = {"schema_version": "scenario_dreamer_background_raw_eval_v1", "task": "current_front_scene_completion_not_future_prediction",
               "raw_state_postprocessing": "none", "checkpoint_sha256": sha256_file(args.checkpoint), "jsonl_sha256": sha256_file(args.jsonl),
               "base_seed": args.seed, "num_records": len(records), "num_samples_per_condition": args.num_samples_per_condition,
               "sampled": {"num_samples": len(rows), **{key: numeric(row["metrics"].get(key) for row in rows) for key in keys}},
               "count_exact_rate": float(np.mean([row["metrics"]["target_count_exact"] for row in rows])),
               "by_town": stratified(rows, "town", keys),
               "by_topology_bucket": stratified(rows, "topology_bucket", keys)}
    if args.num_samples_per_condition > 1:
        groups = defaultdict(list)
        for row in rows: groups[row["record_index"]].append(row)
        oracle = [min(group, key=lambda row: float(row["metrics"].get("position_l2", float("inf")))) for group in groups.values()]
        summary["best_of_n_position_l2_oracle"] = {"num_samples": len(oracle), **{key: numeric(row["metrics"].get(key) for row in oracle) for key in keys}}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True); Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_generated_jsonl: write_jsonl(args.output_generated_jsonl, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
