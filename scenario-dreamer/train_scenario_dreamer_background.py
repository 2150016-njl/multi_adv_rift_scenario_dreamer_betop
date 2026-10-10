#!/usr/bin/env python
"""Train the independent, variable-count ordinary-BV directional model."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from audit_scenario_dreamer_background_dataset import audit_record
from high_value_init.background_ldm import RouteConditionedBackgroundLDM
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True); p.add_argument("--val_jsonl", required=True)
    p.add_argument("--output_dir", required=True); p.add_argument("--autoencoder_checkpoint", required=True)
    p.add_argument("--ldm_checkpoint", required=True); p.add_argument("--resume_checkpoint", default=None)
    p.add_argument("--max_target_agents", type=int, default=8); p.add_argument("--route_num_points", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16); p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps_per_epoch", type=int, default=-1); p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validation_seed", type=int, default=173); p.add_argument("--device", default="cuda")
    p.add_argument("--decoded_state_loss_weight", type=float, default=0.4)
    p.add_argument("--decoded_position_loss_weight", type=float, default=2.0)
    p.add_argument("--decoded_heading_loss_weight", type=float, default=0.7)
    p.add_argument("--decoded_speed_loss_weight", type=float, default=1.0)
    p.add_argument("--decoded_size_loss_weight", type=float, default=0.5)
    p.add_argument("--physical_bounds_loss_weight", type=float, default=0.3)
    p.add_argument("--lane_corridor_loss_weight", type=float, default=0.10)
    p.add_argument("--lane_heading_loss_weight", type=float, default=0.0)
    p.add_argument("--initial_clearance_loss_weight", type=float, default=0.10)
    p.add_argument("--forward_support_loss_weight", type=float, default=0.10)
    p.add_argument("--obb_clearance_loss_weight", type=float, default=0.10)
    p.add_argument("--heading_unit_loss_weight", type=float, default=0.02)
    p.add_argument("--speed_limit_loss_weight", type=float, default=0.05)
    p.add_argument("--min_obb_axis_clearance", type=float, default=0.5)
    p.add_argument("--min_initial_clearance", type=float, default=2.5)
    p.add_argument("--default_lane_half_width", type=float, default=1.75)
    p.add_argument("--min_generated_speed", type=float, default=0.0); p.add_argument("--max_generated_speed", type=float, default=25.0)
    p.add_argument("--fov", type=float, default=64.0); p.add_argument("--partition_boundary_x", type=float, default=0.0)
    p.add_argument("--target_fov_margin", type=float, default=2.0); p.add_argument("--max_num_agents", type=int, default=61)
    p.add_argument("--best_metric", choices=["val_decoded_position_l1", "val_loss"], default="val_decoded_position_l1")
    p.add_argument("--grad_clip_norm", type=float, default=10.0)
    return p.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_contract(rows, split, args):
    errors = []
    for index, row in enumerate(rows):
        result = audit_record(row, split, args)
        if result:
            errors.append({"index": index, "scene_id": row.get("scene_id"), "errors": result})
            if len(errors) == 20: break
    if errors:
        raise ValueError(json.dumps({"background_contract_failures": errors}, indent=2))


def batches(rows, batch_size, rng):
    indices = list(range(len(rows))); rng.shuffle(indices)
    for begin in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[begin:begin + batch_size]]


def make_model(args):
    return RouteConditionedBackgroundLDM(
        checkpoint_path=args.resume_checkpoint, autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint, device=args.device, mode="inpainting",
        num_generate_agents=1, route_num_points=args.route_num_points,
        decoded_state_loss_weight=args.decoded_state_loss_weight, decoded_position_loss_weight=args.decoded_position_loss_weight,
        decoded_heading_loss_weight=args.decoded_heading_loss_weight, decoded_speed_loss_weight=args.decoded_speed_loss_weight,
        decoded_size_loss_weight=args.decoded_size_loss_weight, physical_bounds_loss_weight=args.physical_bounds_loss_weight,
        lane_corridor_loss_weight=args.lane_corridor_loss_weight, lane_heading_loss_weight=args.lane_heading_loss_weight,
        initial_clearance_loss_weight=args.initial_clearance_loss_weight, default_lane_half_width=args.default_lane_half_width,
        min_initial_clearance=args.min_initial_clearance, forward_support_loss_weight=args.forward_support_loss_weight,
        obb_clearance_loss_weight=args.obb_clearance_loss_weight, heading_unit_loss_weight=args.heading_unit_loss_weight,
        speed_limit_loss_weight=args.speed_limit_loss_weight, min_obb_axis_clearance=args.min_obb_axis_clearance,
        min_generated_speed=args.min_generated_speed, max_generated_speed=args.max_generated_speed,
        forward_min_x=args.partition_boundary_x, target_fov_margin=args.target_fov_margin,
        builder_cfg={"reject_observed_over_capacity": True},
    )


@torch.no_grad()
def evaluate(model, rows, args):
    model.eval(); output = []; rng = random.Random(0)
    devices = [model.device.index] if model.device.type == "cuda" and model.device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.validation_seed)
        for batch in batches(rows, args.batch_size, rng):
            metrics = model.training_loss(batch)
            output.append({k: float(v.detach().cpu()) for k, v in metrics.items() if torch.is_tensor(v) and v.ndim == 0})
    return {f"val_{key}": float(np.mean([row[key] for row in output if key in row]))
            for key in {key for row in output for key in row}}


def main():
    args = parse_args(); output = Path(args.output_dir)
    if output.exists(): raise FileExistsError(f"refusing to overwrite background checkpoint output: {output}")
    if abs(args.fov - 64.0) > 1e-6 or args.max_num_agents != 61 or args.max_target_agents < 1:
        raise ValueError("background v1 freezes fov=64, capacity=61, and a positive variable max target count")
    train, val = read_jsonl(args.train_jsonl), read_jsonl(args.val_jsonl)
    if not train or not val: raise ValueError("background train and validation JSONL files must be non-empty")
    require_contract(train, "train", args); require_contract(val, "val", args)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    model = make_model(args); output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed); steps = args.steps_per_epoch if args.steps_per_epoch > 0 else int(np.ceil(len(train) / args.batch_size))
    config = {**vars(args), "trainer": "scenario_dreamer_background_directional_v1", "architecture": "unchanged_v3_scenario_dreamer_ldm_losses_only",
              "task": "current_front_scene_completion_not_future_prediction", "train_jsonl_sha256": sha256(args.train_jsonl),
              "val_jsonl_sha256": sha256(args.val_jsonl), "num_train_records": len(train), "num_val_records": len(val),
              "steps_per_epoch_resolved": steps, "ae_load_info": model.ae_load_info, "ldm_load_info": model.ldm_load_info}
    (output / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    keys = ("loss", "decoded_state_loss", "decoded_position_l1", "decoded_heading_l1", "decoded_speed_l1", "decoded_size_l1",
            "decoded_physical_bounds_loss", "lane_corridor_loss", "lane_heading_loss", "initial_clearance_loss",
            "forward_support_loss", "obb_clearance_loss", "heading_unit_loss", "speed_limit_loss")
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        model.train(); started = time.time(); values = {key: [] for key in keys}
        for batch in batches(train, args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True); losses = model.training_loss(batch); losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm); optimizer.step()
            for key in keys:
                if key in losses: values[key].append(float(losses[key].detach().cpu()))
        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update({f"train_{key}": float(np.mean(value)) for key, value in values.items() if value}); row.update(evaluate(model, val, args))
        history.append(row); print(json.dumps(row, indent=2))
        metadata = {"trainer": "scenario_dreamer_background", "last_epoch": epoch, "config": config}
        model.save_checkpoint(str(output / "last.ckpt"), metadata=metadata)
        current = float(row["val_decoded_position_l1"] if args.best_metric == "val_decoded_position_l1" else row["val_loss"])
        if current < best:
            best = current; model.save_checkpoint(str(output / "best.ckpt"), metadata={**metadata, "best_epoch": epoch, "best_metric": args.best_metric, "best_metric_value": best})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "best_metric": args.best_metric, "best_metric_value": best}, indent=2))


if __name__ == "__main__": main()
