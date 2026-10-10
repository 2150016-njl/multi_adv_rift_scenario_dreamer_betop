"""Fail-fast one-batch backward test for V4C direct route-interaction loss.

This test is intentionally read-only: it loads a checkpoint and V4C records,
verifies that target-only route labels are present, and proves that the direct
RIFT route-distance/synchronization branch has a finite non-zero gradient.
It creates no checkpoint directory and never calls CARLA.
"""

import argparse
import json
import math

import torch

from high_value_init.high_value_ldm_v4c import RouteConditionedHighValueLDMV4C
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    records = read_jsonl(args.jsonl)[:args.batch_size]
    if len(records) != args.batch_size:
        raise ValueError("insufficient records for requested batch size")
    model = RouteConditionedHighValueLDMV4C(
        checkpoint_path=args.checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode="inpainting",
        num_generate_agents=2,
        decoded_state_loss_weight=0.4,
        decoded_position_loss_weight=2.0,
        decoded_heading_loss_weight=0.7,
        decoded_speed_loss_weight=1.0,
        decoded_size_loss_weight=0.5,
        physical_bounds_loss_weight=0.3,
        lane_corridor_loss_weight=0.10,
        lane_heading_loss_weight=0.0,
        initial_clearance_loss_weight=0.10,
        forward_support_loss_weight=0.0,
        obb_clearance_loss_weight=0.0,
        forward_min_x=0.0,
        target_fov_margin=2.0,
        min_obb_axis_clearance=0.5,
        interaction_anchor_along_loss_weight=0.0,
        interaction_anchor_lateral_loss_weight=0.0,
        interaction_route_distance_loss_weight=0.05,
        interaction_route_sync_loss_weight=0.05,
        interaction_route_corridor_loss_weight=0.05,
        interaction_route_tangent_loss_weight=0.025,
        builder_cfg={"reject_observed_over_capacity": True},
    )
    model.train()
    losses = model.training_loss(records)
    direct_keys = (
        "interaction_route_distance_loss", "interaction_route_sync_loss",
        "interaction_route_corridor_loss", "interaction_route_tangent_loss",
    )
    if any(key not in losses for key in direct_keys):
        raise RuntimeError("V4C direct interaction metrics were not produced")
    direct_loss = (
        0.05 * losses["interaction_route_distance_loss"] +
        0.05 * losses["interaction_route_sync_loss"] +
        0.05 * losses["interaction_route_corridor_loss"] +
        0.025 * losses["interaction_route_tangent_loss"]
    )
    if not torch.isfinite(losses["loss"]) or not torch.isfinite(direct_loss):
        raise RuntimeError("V4C total or direct loss is non-finite")
    if float(direct_loss.detach().cpu()) <= 0.0:
        raise RuntimeError("V4C direct route-interaction loss is zero; reject this dataset/checkpoint")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(direct_loss, parameters, retain_graph=True, allow_unused=True)
    direct_gradient_norm = math.sqrt(sum(
        float(torch.sum(gradient.detach() ** 2).cpu())
        for gradient in gradients if gradient is not None
    ))
    if not math.isfinite(direct_gradient_norm) or direct_gradient_norm <= 1e-10:
        raise RuntimeError("V4C direct route-interaction branch has no usable parameter gradient")
    losses["loss"].backward()
    report = {
        "passed": True,
        "batch_size": args.batch_size,
        "direct_route_interaction_gradient_l2": direct_gradient_norm,
        "metrics": {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if torch.is_tensor(value) and value.ndim == 0
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
