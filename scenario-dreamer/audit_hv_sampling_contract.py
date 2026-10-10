#!/usr/bin/env python
"""Diagnose the gap between target-latent reconstruction and LDM sampling.

This script is read-only.  It compares three different operations on the
same supervised records:

* AE oracle: encode the *true full scene* and immediately decode it;
* training-time x0: add noise to the true target latent and predict x0 once;
* reverse diffusion: sample missing targets from the condition, as online
  generation does.

The first two operations have access to target latents.  The third does not.
Consequently this is a sampling-contract diagnostic, not an online policy and
not evidence that the oracle path is deployable.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from cfgs.config import BEFORE_PARTITION
from torch_geometric.data import Batch

import eval_high_value_scene_generator_v4a as v4a
import eval_high_value_scene_generator_v4c as v4c
from high_value_init.evaluation_geometry import seed_everything, sha256_file
from high_value_init.high_value_ldm import match_state_metrics
from high_value_init.high_value_ldm_v3 import RouteConditionedHighValueLDMV3
from high_value_init.rift_dataset import decoded_sd_agents_to_hv_states, read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_rows_jsonl", default=None)
    parser.add_argument("--max_records", type=int, default=-1)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite_output", action="store_true", default=False)
    return parser.parse_args()


def decoded_states(model, agent_latents, lane_latents, data, context_count, target_count):
    agent_latents, lane_latents = model._unnormalize_latents(agent_latents, lane_latents)
    agents, lanes, _, _, _ = model.autoencoder.forward_decoder(agent_latents, lane_latents, data)
    agents, _ = model._unnormalize_scene(agents, lanes)
    return decoded_sd_agents_to_hv_states(agents, context_count, target_count).detach().cpu().tolist()


@torch.no_grad()
def ae_oracle_states(model, record):
    full_data, meta = model.graph_builder.build_ae_data(record, include_targets=True)
    full_batch = Batch.from_data_list([full_data]).to(model.device)
    agent_latents, lane_latents = model._encode_normalized(full_batch)
    return decoded_states(
        model, agent_latents, lane_latents, full_batch,
        int(meta["context_agent_count"]), int(meta["target_agent_count"]),
    )


@torch.no_grad()
def one_step_x0_states(model, record):
    """Mirror V3 decoded-x0 training logic, returning its raw decoded state."""
    data = model.build_training_batch([record])
    x_agent = data["agent"].latents.unsqueeze(1)
    x_lane = data["lane"].latents.unsqueeze(1)
    agent_batch = data["agent"].batch
    lane_batch = data["lane"].batch
    timesteps = torch.randint(0, model.ldm.n_timesteps, (data.batch_size,), device=x_agent.device).long()
    t_agent, t_lane = timesteps[agent_batch], timesteps[lane_batch]
    x_agent_noisy = model.ldm.q_sample(x_start=x_agent, t=t_agent, noise=torch.randn_like(x_agent))
    x_lane_noisy = model.ldm.q_sample(x_start=x_lane, t=t_lane, noise=torch.randn_like(x_lane))
    agent_condition = data["agent"].partition_mask == BEFORE_PARTITION
    lane_condition = data["lane"].partition_mask == BEFORE_PARTITION
    x_agent_noisy = torch.where(agent_condition.view(-1, 1, 1), x_agent, x_agent_noisy)
    x_lane_noisy = torch.where(lane_condition.view(-1, 1, 1), x_lane, x_lane_noisy)
    pred_agent_noise, pred_lane_noise = model.ldm.model(x_agent_noisy, x_lane_noisy, data, t_agent, t_lane)
    pred_agent = model.ldm.predict_start_from_noise(x_agent_noisy, t=t_agent, noise=pred_agent_noise)[:, 0]
    pred_lane = model.ldm.predict_start_from_noise(x_lane_noisy, t=t_lane, noise=pred_lane_noise)[:, 0]
    pred_agent = torch.where(agent_condition.view(-1, 1), data["agent"].latents, pred_agent)
    pred_lane = torch.where(lane_condition.view(-1, 1), data["lane"].latents, pred_lane)
    meta = model.graph_builder.build_ldm_seed(record, model.agent_latent_dim, model.lane_latent_dim)[1]
    return decoded_states(
        model, pred_agent, pred_lane, data,
        int(meta["context_agent_count"]), int(meta["target_agent_count"]),
    ), int(timesteps[0].detach().cpu())


def metrics(model, states, record, args):
    target = torch.tensor((record.get("target") or {}).get("generated_agent_states") or [], dtype=torch.float32)
    return {
        **match_state_metrics(torch.tensor(states, dtype=torch.float32), target, bounds=model.generated_state_bounds),
        **v4a.raw_geometry(states, record, args.default_lane_half_width, args.min_initial_clearance),
        **v4a.associated_lane_geometry(states, target, record, args.default_lane_half_width),
        **v4a.forward_geometry(states, args),
        **v4c.anchor_geometry(states, target, record),
        **v4c.route_progress_geometry(states, target, record),
        "position_l1": v4a.position_l1(states, target),
    }


def summarize(rows):
    return v4c.summarize_rows(rows)


def main():
    args = parse_args()
    output = Path(args.output_json)
    rows_output = Path(args.output_rows_jsonl) if args.output_rows_jsonl else None
    if not args.overwrite_output and (output.exists() or (rows_output and rows_output.exists())):
        raise FileExistsError("refusing to overwrite sampling-contract evidence")
    records = read_jsonl(args.jsonl)
    if args.max_records > 0:
        records = records[:args.max_records]
    if not records:
        raise ValueError("no records")
    model = RouteConditionedHighValueLDMV3(
        checkpoint_path=args.checkpoint, autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint, device=args.device, mode="inpainting",
        num_generate_agents=2, route_num_points=80,
        default_lane_half_width=args.default_lane_half_width,
        min_initial_clearance=args.min_initial_clearance, clamp_generated_states=False,
        builder_cfg={"reject_observed_over_capacity": True},
    )
    rows = []
    for index, record in enumerate(records):
        seed = seed_everything(args.seed, index, 0)
        for method, states, timestep in (
            ("ae_oracle_true_target_latent", ae_oracle_states(model, record), None),
            ("training_time_noisy_true_target_x0", *one_step_x0_states(model, record)),
            ("reverse_diffusion_condition_only_s1", model.generate(record, 2)["generated_agent_states"], None),
        ):
            rows.append({
                "record_index": index, "sample_seed": seed, "method": method,
                "training_timestep": timestep, "scene_id": record.get("scene_id"),
                "town": record.get("town"),
                "topology_bucket": (record.get("topology_v4") or {}).get("bucket"),
                "states": states, "metrics": metrics(model, states, record, args),
            })
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        by_method[method] = summarize([row for row in rows if row["method"] == method])
    report = {
        "schema_version": "hv_sampling_contract_audit_v1",
        "checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(args.checkpoint),
        "jsonl": args.jsonl, "jsonl_sha256": sha256_file(args.jsonl),
        "num_records": len(records), "base_seed": args.seed,
        "policy": {
            "online_equivalent_method": "reverse_diffusion_condition_only_s1",
            "oracle_methods_use_true_target_latents": True,
            "state_postprocessing": "none",
            "interpretation": (
                "A large reverse-diffusion gap after good AE/x0 results identifies a conditional sampling/training "
                "contract failure. It does not prove a CARLA, decoder, or RIFT failure."
            ),
        },
        "by_method": by_method,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if rows_output:
        write_jsonl(str(rows_output), rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
