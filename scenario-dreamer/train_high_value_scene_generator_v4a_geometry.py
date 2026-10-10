#!/usr/bin/env python
"""Train isolated V4-A1/A2 geometry-loss checkpoints.

V4-A0 remains ``train_high_value_scene_generator_v4a.py``.  This trainer
requires an A0 parent for A1, and an A1 parent for A2, so data-task, forward
support, and OBB effects cannot be silently mixed in one checkpoint.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_high_value_scene_dataset_v4 import audit_record
from high_value_init.high_value_ldm_v4a import RouteConditionedHighValueLDMV4A
from high_value_init.rift_dataset import read_jsonl
from train_high_value_scene_generator_v4a import batches, evaluate, weight


STAGES = {
    "a1_forward_support": {
        "parent": "a0_data_only",
        "requires_forward": True,
        "requires_obb": False,
    },
    "a2_obb_clearance": {
        "parent": "a1_forward_support",
        "requires_forward": True,
        "requires_obb": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--resume_checkpoint", required=True)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--route_num_points", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation_seed", type=int, default=173)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoded_state_loss_weight", type=float, default=0.4)
    parser.add_argument("--decoded_position_loss_weight", type=float, default=2.0)
    parser.add_argument("--decoded_heading_loss_weight", type=float, default=0.7)
    parser.add_argument("--decoded_speed_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoded_size_loss_weight", type=float, default=0.5)
    parser.add_argument("--physical_bounds_loss_weight", type=float, default=0.3)
    parser.add_argument("--lane_corridor_loss_weight", type=float, default=0.10)
    parser.add_argument("--lane_heading_loss_weight", type=float, default=0.0)
    parser.add_argument("--initial_clearance_loss_weight", type=float, default=0.10)
    parser.add_argument("--forward_support_loss_weight", type=float, default=0.10)
    parser.add_argument("--obb_clearance_loss_weight", type=float, default=0.10)
    parser.add_argument("--min_obb_axis_clearance", type=float, default=0.5)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--record_weight_key", default="training_weight")
    parser.add_argument("--no_weighted_sampling", action="store_true", default=False)
    parser.add_argument("--best_metric", choices=["val_decoded_position_l1", "val_loss"], default="val_decoded_position_l1")
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_stage(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    config = metadata.get("config") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or metadata.get("trainer") != "v4a" or not isinstance(config, dict):
        raise ValueError("geometry V4-A stages require a checkpoint written by an audited V4-A trainer")
    stage = config.get("v4a_stage")
    if not isinstance(stage, str):
        raise ValueError("parent V4-A checkpoint has no v4a_stage metadata")
    return stage


def contract_args(args):
    return SimpleNamespace(fov=64.0, forward_min_x=args.forward_min_x,
                           target_fov_margin=args.target_fov_margin,
                           max_num_agents=args.max_num_agents,
                           max_target_lane_distance=args.max_target_lane_distance)


def require_contract(records, split, args):
    failures, checked = [], contract_args(args)
    for index, record in enumerate(records):
        errors = audit_record(record, split, checked)
        if errors:
            failures.append({"index": index, "scene_id": record.get("scene_id"), "errors": errors})
            if len(failures) >= 20:
                break
    if failures:
        raise ValueError(json.dumps({"split": split, "v4a_contract_failures": failures}, indent=2))


def validate_stage(args):
    spec = STAGES[args.stage]
    if args.num_generate_agents != 2 or args.max_num_agents != 61:
        raise ValueError("V4-A geometry stages require two targets and graph capacity 61")
    if spec["requires_forward"] and args.forward_support_loss_weight <= 0.0:
        raise ValueError("forward-support stage requires --forward_support_loss_weight > 0")
    if spec["requires_obb"] and args.obb_clearance_loss_weight <= 0.0:
        raise ValueError("OBB stage requires --obb_clearance_loss_weight > 0")
    if not spec["requires_obb"] and args.obb_clearance_loss_weight != 0.0:
        raise ValueError("A1 must use --obb_clearance_loss_weight 0; OBB is isolated to A2")
    actual_parent = parent_stage(args.resume_checkpoint)
    if actual_parent != spec["parent"]:
        raise ValueError(f"{args.stage} requires parent stage {spec['parent']!r}, got {actual_parent!r}")


def make_model(args):
    return RouteConditionedHighValueLDMV4A(
        checkpoint_path=args.resume_checkpoint, autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint, device=args.device, mode="inpainting",
        num_generate_agents=2, route_num_points=args.route_num_points,
        decoded_state_loss_weight=args.decoded_state_loss_weight,
        decoded_position_loss_weight=args.decoded_position_loss_weight,
        decoded_heading_loss_weight=args.decoded_heading_loss_weight,
        decoded_speed_loss_weight=args.decoded_speed_loss_weight,
        decoded_size_loss_weight=args.decoded_size_loss_weight,
        physical_bounds_loss_weight=args.physical_bounds_loss_weight,
        lane_corridor_loss_weight=args.lane_corridor_loss_weight,
        lane_heading_loss_weight=args.lane_heading_loss_weight,
        initial_clearance_loss_weight=args.initial_clearance_loss_weight,
        default_lane_half_width=args.default_lane_half_width,
        min_initial_clearance=args.min_initial_clearance,
        forward_support_loss_weight=args.forward_support_loss_weight,
        obb_clearance_loss_weight=args.obb_clearance_loss_weight,
        forward_min_x=args.forward_min_x, target_fov_margin=args.target_fov_margin,
        min_obb_axis_clearance=args.min_obb_axis_clearance,
        builder_cfg={"reject_observed_over_capacity": True},
    )


def main():
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing V4-A stage output: {output}")
    validate_stage(args)
    train, val = read_jsonl(args.train_jsonl), read_jsonl(args.val_jsonl)
    if not train or not val:
        raise ValueError("train and val JSONL files must be non-empty")
    require_contract(train, "train", args)
    require_contract(val, "val", args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = make_model(args)
    output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    steps = args.steps_per_epoch if args.steps_per_epoch > 0 else int(np.ceil(len(train) / args.batch_size))
    config = {
        **vars(args), "v4a_stage": args.stage, "parent_checkpoint_sha256": sha256_file(args.resume_checkpoint),
        "num_train_records": len(train), "num_val_records": len(val), "steps_per_epoch_resolved": steps,
        "ae_load_info": model.ae_load_info, "ldm_load_info": model.ldm_load_info,
        "architecture": "unchanged_v3_scenario_dreamer_ldm_losses_only",
    }
    (output / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    keys = ("loss", "decoded_state_loss", "decoded_position_l1", "decoded_heading_l1", "decoded_speed_l1",
            "decoded_size_l1", "decoded_physical_bounds_loss", "lane_corridor_loss", "lane_heading_loss",
            "initial_clearance_loss", "forward_support_loss", "obb_clearance_loss")
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        model.train()
        started, values = time.time(), {key: [] for key in keys}
        for batch in batches(train, args.batch_size, rng, not args.no_weighted_sampling, args.record_weight_key, steps):
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_loss(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            for key in keys:
                if key in losses:
                    values[key].append(float(losses[key].detach().cpu()))
        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update({f"train_{key}": float(np.mean(value)) for key, value in values.items() if value})
        row.update(evaluate(model, val, args.batch_size, args.validation_seed))
        history.append(row)
        print(json.dumps(row, indent=2))
        metadata = {"trainer": "v4a", "v4a_stage": args.stage, "last_epoch": epoch, "config": config}
        model.save_checkpoint(str(output / "last.ckpt"), metadata=metadata)
        current = float(row["val_decoded_position_l1"] if args.best_metric == "val_decoded_position_l1" else row["val_loss"])
        if current < best:
            best = current
            model.save_checkpoint(str(output / "best.ckpt"), metadata={
                "trainer": "v4a", "v4a_stage": args.stage, "best_epoch": epoch,
                "best_metric": args.best_metric, "best_metric_value": best, "config": config,
            })
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "stage": args.stage, "best_metric": args.best_metric,
                      "best_metric_value": best}, indent=2))


if __name__ == "__main__":
    main()
