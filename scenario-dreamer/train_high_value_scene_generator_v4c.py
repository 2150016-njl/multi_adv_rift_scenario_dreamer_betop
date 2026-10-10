"""Train V4C with target-only RIFT route-interaction supervision.

This is an ablation trainer, not an online-RIFT trainer.  Its input graph is
identical to V4B; the only new values are target labels used after decoding.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from audit_high_value_scene_dataset_v4 import audit_record
from high_value_init.high_value_ldm_v4c import RouteConditionedHighValueLDMV4C
from high_value_init.rift_dataset import read_jsonl
import train_high_value_scene_generator_v4b as v4b


def parse_args():
    parser = v4b.build_parser()
    parser.prog = "train_high_value_scene_generator_v4c.py"
    parser.description = __doc__
    parser.add_argument(
        "--v4c_stage", choices=("c0_target_metadata_only", "c1_conflict_anchor", "c2_route_progress_sync"), required=True,
        help=("c0 proves the target-label path; c1 enables anchor geometry; "
              "c2 directly supervises RIFT trace_route remaining distance and synchronization."),
    )
    parser.add_argument("--interaction_anchor_along_loss_weight", type=float, default=None)
    parser.add_argument("--interaction_anchor_lateral_loss_weight", type=float, default=None)
    parser.add_argument("--interaction_route_distance_loss_weight", type=float, default=None)
    parser.add_argument("--interaction_route_sync_loss_weight", type=float, default=None)
    parser.add_argument("--interaction_route_corridor_loss_weight", type=float, default=None)
    parser.add_argument("--interaction_route_tangent_loss_weight", type=float, default=None)
    parser.add_argument("--interaction_strength_weight_power", type=float, default=0.0)
    return parser.parse_args()


def resolve_interaction_weights(args):
    defaults = {
        "c0_target_metadata_only": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        # Kept deliberately smaller than the existing decoded-state branch.
        # It is an ablation starting point, not a claim that a large anchor
        # loss creates better CBVs by itself.
        "c1_conflict_anchor": (0.05, 0.05, 0.0, 0.0, 0.0, 0.0),
        # Directly uses RIFT trace_route target labels reconstructed in the
        # offline sidecar. It remains smaller than the decoded-state branch.
        # This is the actual strong-route-interaction ablation: RIFT's
        # preview-point route distance, ego synchronization, path corridor,
        # and travel direction. Conflict-anchor loss is deliberately zero.
        "c2_route_progress_sync": (0.0, 0.0, 0.05, 0.05, 0.05, 0.025),
    }
    (default_along, default_lateral, default_route_distance, default_route_sync,
     default_route_corridor, default_route_tangent) = defaults[args.v4c_stage]
    along = default_along if args.interaction_anchor_along_loss_weight is None else args.interaction_anchor_along_loss_weight
    lateral = default_lateral if args.interaction_anchor_lateral_loss_weight is None else args.interaction_anchor_lateral_loss_weight
    route_distance = default_route_distance if args.interaction_route_distance_loss_weight is None else args.interaction_route_distance_loss_weight
    route_sync = default_route_sync if args.interaction_route_sync_loss_weight is None else args.interaction_route_sync_loss_weight
    route_corridor = default_route_corridor if args.interaction_route_corridor_loss_weight is None else args.interaction_route_corridor_loss_weight
    route_tangent = default_route_tangent if args.interaction_route_tangent_loss_weight is None else args.interaction_route_tangent_loss_weight
    if min(along, lateral, route_distance, route_sync, route_corridor, route_tangent) < 0.0:
        raise ValueError("V4C interaction weights must be non-negative")
    if args.v4c_stage == "c0_target_metadata_only" and any(value != 0.0 for value in (along, lateral, route_distance, route_sync, route_corridor, route_tangent)):
        raise ValueError("c0_target_metadata_only requires zero V4C interaction weights")
    if args.v4c_stage == "c1_conflict_anchor" and (along <= 0.0 or lateral <= 0.0):
        raise ValueError("c1_conflict_anchor requires positive along and lateral weights")
    if args.v4c_stage == "c1_conflict_anchor" and any(value != 0.0 for value in (route_distance, route_sync, route_corridor, route_tangent)):
        raise ValueError("c1_conflict_anchor requires zero route-progress/synchronization weights")
    if args.v4c_stage == "c2_route_progress_sync" and min(route_distance, route_sync, route_corridor, route_tangent) <= 0.0:
        raise ValueError("c2_route_progress_sync requires positive route distance, sync, corridor, and tangent weights")
    return tuple(float(value) for value in (along, lateral, route_distance, route_sync, route_corridor, route_tangent))


def require_v4c_contract(records, split, args):
    failures = []
    audit_args = argparse.Namespace(
        fov=64.0, forward_min_x=args.forward_min_x,
        target_fov_margin=args.target_fov_margin, max_num_agents=61,
        max_target_lane_distance=5.0,
    )
    for index, record in enumerate(records):
        errors = audit_record(record, split, audit_args)
        target = record.get("target") or {}
        supervision = list(target.get("route_interaction_supervision") or [])
        condition_text = json.dumps(record.get("condition") or {}, ensure_ascii=False)
        if record.get("dataset_version") != "v4c_route_interaction_supervised":
            errors.append("dataset_version_mismatch")
        if (record.get("v4b_contract") or {}).get("target_pair_selection") != args.expected_target_pair_policy:
            errors.append("target_pair_policy_mismatch")
        try:
            pair_clearance = float((record.get("v4b_contract") or {}).get("target_pair_obb_axis_clearance"))
        except (TypeError, ValueError):
            pair_clearance = float("-inf")
        if pair_clearance < args.min_obb_axis_clearance:
            errors.append("target_pair_obb_clearance_below_training_contract")
        if (record.get("v4c_contract") or {}).get("supervision") != "rift_conflict_point_target_only":
            errors.append("v4c_contract_missing")
        if len(supervision) != 2:
            errors.append("route_interaction_target_count_mismatch")
        if any(marker in condition_text for marker in (
                "interaction_waypoint", "conflict_point", "route_distance_difference",
                "target_route_distance_to_conflict", "interaction_strength")):
            errors.append("route_interaction_label_leaked_to_condition")
        for item in supervision:
            anchor = (item or {}).get("conflict_point_local_xy")
            if not isinstance(anchor, (list, tuple)) or len(anchor) < 2:
                errors.append("route_interaction_anchor_missing")
                break
            polyline = (item or {}).get("target_route_polyline_local_xy") or []
            if (len(polyline) < 2 or
                    (item or {}).get("target_route_distance_offset_m") is None or
                    (item or {}).get("rift_preview_distance_m") is None or
                    (item or {}).get("planner_distance_abs_error_to_rift_label_m") is None):
                errors.append("route_progress_supervision_missing")
                break
        if errors:
            failures.append({"index": index, "scene_id": record.get("scene_id"), "errors": sorted(set(errors))})
            if len(failures) >= 20:
                break
    if failures:
        raise ValueError(json.dumps({"split": split, "v4c_contract_failures": failures}, indent=2))


def make_model(args, forward_weight, obb_weight, along_weight, lateral_weight, route_distance_weight, route_sync_weight, route_corridor_weight, route_tangent_weight):
    return RouteConditionedHighValueLDMV4C(
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
        interaction_anchor_along_loss_weight=along_weight,
        interaction_anchor_lateral_loss_weight=lateral_weight,
        interaction_route_distance_loss_weight=route_distance_weight,
        interaction_route_sync_loss_weight=route_sync_weight,
        interaction_route_corridor_loss_weight=route_corridor_weight,
        interaction_route_tangent_loss_weight=route_tangent_weight,
        interaction_strength_weight_power=args.interaction_strength_weight_power,
        builder_cfg={"reject_observed_over_capacity": True},
    )


def main():
    args = parse_args()
    if args.num_generate_agents != 2 or args.context_dropout_min_keep < 0:
        raise ValueError("V4C requires exactly two targets and a non-negative context minimum")
    if not 0.0 <= args.context_dropout_prob < 1.0:
        raise ValueError("--context_dropout_prob must be in [0, 1)")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite existing V4C output directory: %s" % output)
    train, val = read_jsonl(args.train_jsonl), read_jsonl(args.val_jsonl)
    if not train or not val:
        raise ValueError("train and val JSONL files must be non-empty")
    require_v4c_contract(train, "train", args)
    require_v4c_contract(val, "val", args)
    forward_weight, obb_weight = v4b.resolve_geometry_weights(args)
    (along_weight, lateral_weight, route_distance_weight, route_sync_weight,
     route_corridor_weight, route_tangent_weight) = resolve_interaction_weights(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    model = make_model(
        args, forward_weight, obb_weight, along_weight, lateral_weight,
        route_distance_weight, route_sync_weight, route_corridor_weight, route_tangent_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    steps = args.steps_per_epoch if args.steps_per_epoch > 0 else int(np.ceil(len(train) / args.batch_size))
    config = {
        **vars(args), "trainer": "v4c", "architecture": "RouteConditionedHighValueLDMV4A_with_target_only_rift_conflict_loss",
        "resolved_forward_support_loss_weight": forward_weight,
        "resolved_obb_clearance_loss_weight": obb_weight,
        "resolved_interaction_anchor_along_loss_weight": along_weight,
        "resolved_interaction_anchor_lateral_loss_weight": lateral_weight,
        "resolved_interaction_route_distance_loss_weight": route_distance_weight,
        "resolved_interaction_route_sync_loss_weight": route_sync_weight,
        "resolved_interaction_route_corridor_loss_weight": route_corridor_weight,
        "resolved_interaction_route_tangent_loss_weight": route_tangent_weight,
        "num_train_records": len(train), "num_val_records": len(val),
        "steps_per_epoch_resolved": steps, "ae_load_info": model.ae_load_info,
        "ldm_load_info": model.ldm_load_info,
    }
    (output / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metric_keys = (
        "loss", "agent_loss", "decoded_state_loss", "decoded_position_l1", "decoded_heading_l1",
        "decoded_speed_l1", "decoded_size_l1", "decoded_physical_bounds_loss", "lane_corridor_loss",
        "lane_heading_loss", "initial_clearance_loss", "forward_support_loss", "obb_clearance_loss",
        "heading_unit_loss", "speed_limit_loss", "interaction_anchor_along_loss",
        "interaction_anchor_lateral_loss", "interaction_anchor_along_abs_error_m",
        "interaction_anchor_lateral_abs_error_m",
        "interaction_route_distance_loss", "interaction_route_sync_loss",
        "interaction_route_distance_abs_error_m", "interaction_route_sync_gap_m",
        "interaction_route_sync_abs_error_m", "interaction_route_sync_abs_gap_m",
        "interaction_route_corridor_loss", "interaction_route_tangent_loss",
        "interaction_route_preview_lateral_error_m", "interaction_route_heading_tangent_error",
    )
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        model.train()
        started = time.time()
        values = {key: [] for key in metric_keys}
        dropout_before = dropout_after = 0
        for clean_batch in v4b.batches(
                train, args.batch_size, rng, not args.no_weighted_sampling, args.record_weight_key, steps):
            batch, context_counts = v4b.apply_context_dropout(
                clean_batch, rng, args.context_dropout_prob, args.context_dropout_min_keep,
            )
            dropout_before += context_counts["before"]
            dropout_after += context_counts["after"]
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_loss(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            for key in metric_keys:
                if key in losses:
                    values[key].append(float(losses[key].detach().cpu()))
        row = {
            "epoch": epoch, "seconds": time.time() - started,
            "train_context_agents_before_dropout": dropout_before,
            "train_context_agents_after_dropout": dropout_after,
        }
        row.update({"train_" + key: float(np.mean(value)) for key, value in values.items() if value})
        row.update(v4b.evaluate(model, val, args.batch_size, args.validation_seed))
        history.append(row)
        print(json.dumps(row, indent=2))
        metadata = {"trainer": "v4c", "v4c_stage": args.v4c_stage, "config": config}
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
                      "best_metric_value": best, "v4c_stage": args.v4c_stage}, indent=2))


if __name__ == "__main__":
    main()
