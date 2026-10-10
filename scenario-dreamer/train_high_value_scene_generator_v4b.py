"""Train V4B with the actual V4-A weak-geometry loss implementation.

V4A's historical trainer intentionally used the V3 class for its A0
data-contract baseline.  This separate trainer leaves that baseline intact and
instantiates ``RouteConditionedHighValueLDMV4A`` so forward-support and OBB
clearance ablations are real, checkpointed experiments.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from audit_high_value_scene_dataset_v4 import audit_record
from high_value_init.high_value_ldm_v4a import RouteConditionedHighValueLDMV4A
from high_value_init.rift_dataset import read_jsonl


def build_parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument(
        "--v4b_stage",
        choices=("b0_data_only", "b1_forward_support", "b2_obb_clearance", "b3_joint_geometry"),
        required=True,
    )
    parser.add_argument("--forward_support_loss_weight", type=float, default=None)
    parser.add_argument("--obb_clearance_loss_weight", type=float, default=None)
    parser.add_argument("--heading_unit_loss_weight", type=float, default=0.0)
    parser.add_argument("--speed_limit_loss_weight", type=float, default=0.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--min_obb_axis_clearance", type=float, default=0.5)
    parser.add_argument(
        "--expected_target_pair_policy",
        choices=("high_value_pair", "route_diversity"),
        default="high_value_pair",
        help="Refuse a dataset not built with this explicit V4B offline pair policy.",
    )
    parser.add_argument("--min_generated_speed", type=float, default=0.0)
    parser.add_argument("--max_generated_speed", type=float, default=30.0)
    parser.add_argument("--context_dropout_prob", type=float, default=0.0)
    parser.add_argument("--context_dropout_min_keep", type=int, default=0)
    parser.add_argument("--record_weight_key", default="training_weight")
    parser.add_argument("--no_weighted_sampling", action="store_true", default=False)
    parser.add_argument("--best_metric", choices=("val_decoded_position_l1", "val_loss"), default="val_decoded_position_l1")
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    return parser


def parse_args():
    return build_parser().parse_args()


def resolve_geometry_weights(args):
    presets = {
        "b0_data_only": (0.0, 0.0),
        "b1_forward_support": (0.05, 0.0),
        "b2_obb_clearance": (0.0, 0.05),
        "b3_joint_geometry": (0.05, 0.05),
    }
    default_forward, default_obb = presets[args.v4b_stage]
    forward = default_forward if args.forward_support_loss_weight is None else args.forward_support_loss_weight
    obb = default_obb if args.obb_clearance_loss_weight is None else args.obb_clearance_loss_weight
    if forward < 0.0 or obb < 0.0:
        raise ValueError("V4B geometry loss weights must be non-negative")
    expected = {
        "b0_data_only": (forward == 0.0 and obb == 0.0),
        "b1_forward_support": (forward > 0.0 and obb == 0.0),
        "b2_obb_clearance": (forward == 0.0 and obb > 0.0),
        "b3_joint_geometry": (forward > 0.0 and obb > 0.0),
    }
    if not expected[args.v4b_stage]:
        raise ValueError("--v4b_stage does not match the resolved forward/OBB loss weights")
    return float(forward), float(obb)


def require_contract(records, split, args):
    failures = []
    audit_args = argparse.Namespace(
        fov=64.0, forward_min_x=args.forward_min_x,
        target_fov_margin=args.target_fov_margin, max_num_agents=61,
        max_target_lane_distance=5.0,
    )
    for index, record in enumerate(records):
        errors = audit_record(record, split, audit_args)
        contract = record.get("v4b_contract") or {}
        if record.get("dataset_version") != "v4b_group_stratified":
            errors.append("dataset_version_mismatch")
        if contract.get("target_pair_selection") != args.expected_target_pair_policy:
            errors.append("target_pair_policy_mismatch")
        try:
            pair_clearance = float(contract.get("target_pair_obb_axis_clearance"))
        except (TypeError, ValueError):
            pair_clearance = float("-inf")
        if pair_clearance < args.min_obb_axis_clearance:
            errors.append("target_pair_obb_clearance_below_training_contract")
        if errors:
            failures.append({"index": index, "scene_id": record.get("scene_id"), "errors": errors})
            if len(failures) == 20:
                break
    if failures:
        raise ValueError(json.dumps({"split": split, "v4b_contract_failures": failures}, indent=2))


def record_weight(record, key):
    try:
        return min(max(float(record.get(key, 1.0)), 0.25), 5.0)
    except (TypeError, ValueError):
        return 1.0


def batches(records, batch_size, rng, weighted, key, steps):
    indices = list(range(len(records)))
    if weighted:
        weights = [record_weight(record, key) for record in records]
        for _ in range(steps):
            yield [records[index] for index in rng.choices(indices, weights=weights, k=batch_size)]
        return
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [records[index] for index in indices[start:start + batch_size]]


def apply_context_dropout(records, rng, probability, min_keep):
    """Drop only observed BV rows for training; target/map/ego/route stay exact."""
    if probability <= 0.0:
        return records, {"before": sum(len((row.get("condition") or {}).get("observed_agent_states") or []) for row in records),
                         "after": sum(len((row.get("condition") or {}).get("observed_agent_states") or []) for row in records)}
    transformed, before, after = [], 0, 0
    for record in records:
        item = copy.deepcopy(record)
        condition = item["condition"]
        states = list(condition.get("observed_agent_states") or [])
        types = list(condition.get("observed_agent_types") or [])
        before += len(states)
        kept = [index for index in range(len(states)) if rng.random() >= probability]
        if len(kept) < min_keep:
            missing = [index for index in range(len(states)) if index not in kept]
            rng.shuffle(missing)
            kept.extend(missing[:max(0, min_keep - len(kept))])
        kept = sorted(set(kept))
        condition["observed_agent_states"] = [states[index] for index in kept]
        condition["observed_agent_types"] = [types[index] if index < len(types) else 1 for index in kept]
        after += len(kept)
        transformed.append(item)
    return transformed, {"before": before, "after": after}


def make_model(args, forward_weight, obb_weight):
    return RouteConditionedHighValueLDMV4A(
        checkpoint_path=args.resume_checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device, mode="inpainting", num_generate_agents=2,
        route_num_points=args.route_num_points,
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
        forward_support_loss_weight=forward_weight,
        obb_clearance_loss_weight=obb_weight,
        forward_min_x=args.forward_min_x,
        target_fov_margin=args.target_fov_margin,
        min_obb_axis_clearance=args.min_obb_axis_clearance,
        heading_unit_loss_weight=args.heading_unit_loss_weight,
        speed_limit_loss_weight=args.speed_limit_loss_weight,
        min_generated_speed=args.min_generated_speed,
        max_generated_speed=args.max_generated_speed,
        builder_cfg={"reject_observed_over_capacity": True},
    )


@torch.no_grad()
def evaluate(model, records, batch_size, validation_seed):
    model.eval()
    rows = []
    rng = random.Random(0)
    devices = [model.device.index] if model.device.type == "cuda" and model.device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(validation_seed)
        for batch in batches(records, batch_size, rng, False, "", 0):
            losses = model.training_loss(batch)
            rows.append({key: float(value.detach().cpu()) for key, value in losses.items()
                         if torch.is_tensor(value) and value.ndim == 0})
    keys = {key for row in rows for key in row}
    return {"val_" + key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def main():
    args = parse_args()
    if args.num_generate_agents != 2 or args.context_dropout_min_keep < 0:
        raise ValueError("V4B requires exactly two targets and a non-negative context minimum")
    if args.min_obb_axis_clearance < 0.0:
        raise ValueError("--min_obb_axis_clearance must be non-negative")
    if not 0.0 <= args.context_dropout_prob < 1.0:
        raise ValueError("--context_dropout_prob must be in [0, 1)")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite existing V4B output directory: %s" % output)
    train, val = read_jsonl(args.train_jsonl), read_jsonl(args.val_jsonl)
    if not train or not val:
        raise ValueError("train and val JSONL files must be non-empty")
    require_contract(train, "train", args)
    require_contract(val, "val", args)
    forward_weight, obb_weight = resolve_geometry_weights(args)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    model = make_model(args, forward_weight, obb_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    steps = args.steps_per_epoch if args.steps_per_epoch > 0 else int(np.ceil(len(train) / args.batch_size))
    config = {
        **vars(args), "trainer": "v4b", "v4b_stage": args.v4b_stage,
        "architecture": "RouteConditionedHighValueLDMV4A_with_frozen_scenario_dreamer_backbone",
        "resolved_forward_support_loss_weight": forward_weight,
        "resolved_obb_clearance_loss_weight": obb_weight,
        "num_train_records": len(train), "num_val_records": len(val),
        "steps_per_epoch_resolved": steps, "ae_load_info": model.ae_load_info,
        "ldm_load_info": model.ldm_load_info,
    }
    (output / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metric_keys = (
        "loss", "agent_loss", "decoded_state_loss", "decoded_position_l1", "decoded_heading_l1",
        "decoded_speed_l1", "decoded_size_l1", "decoded_physical_bounds_loss", "lane_corridor_loss",
        "lane_heading_loss", "initial_clearance_loss", "forward_support_loss", "obb_clearance_loss",
        "heading_unit_loss", "speed_limit_loss",
    )
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        model.train(); started = time.time(); values = {key: [] for key in metric_keys}
        dropout_before = dropout_after = 0
        for clean_batch in batches(train, args.batch_size, rng, not args.no_weighted_sampling, args.record_weight_key, steps):
            batch, context_counts = apply_context_dropout(
                clean_batch, rng, args.context_dropout_prob, args.context_dropout_min_keep,
            )
            dropout_before += context_counts["before"]; dropout_after += context_counts["after"]
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_loss(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            for key in metric_keys:
                if key in losses:
                    values[key].append(float(losses[key].detach().cpu()))
        row = {"epoch": epoch, "seconds": time.time() - started,
               "train_context_agents_before_dropout": dropout_before,
               "train_context_agents_after_dropout": dropout_after}
        row.update({"train_" + key: float(np.mean(value)) for key, value in values.items() if value})
        row.update(evaluate(model, val, args.batch_size, args.validation_seed))
        history.append(row)
        print(json.dumps(row, indent=2))
        metadata = {"trainer": "v4b", "v4b_stage": args.v4b_stage, "config": config}
        model.save_checkpoint(str(output / "last.ckpt"), metadata={**metadata, "last_epoch": epoch})
        current = float(row[args.best_metric])
        if current < best:
            best = current
            model.save_checkpoint(str(output / "best.ckpt"), metadata={
                **metadata, "best_epoch": epoch, "best_metric": args.best_metric,
                "best_metric_value": best,
            })
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "best_metric": args.best_metric,
                      "best_metric_value": best, "v4b_stage": args.v4b_stage}, indent=2))


if __name__ == "__main__":
    main()
