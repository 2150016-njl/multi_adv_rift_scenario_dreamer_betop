#!/usr/bin/env python
"""Offline evaluation for the high-value actor scoring head."""

import argparse
import json
from pathlib import Path

import torch

from high_value_init.rift_generator import ScenarioDreamerHighValueInitGenerator
from train_high_value_init import build_feature_batch, iter_batches, topk_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--hv_init_checkpoint", type=str, required=True)
    parser.add_argument("--hv_model_backbone", type=str, default="autoencoder", choices=["autoencoder", "adapter_only"])
    parser.add_argument("--hv_model_autoencoder_checkpoint", type=str, default=None)
    parser.add_argument("--hv_model_cfg_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument(
        "--mask_pos_weight",
        type=float,
        default=None,
        help="Positive weight for mask BCE. Default None computes negatives/positives per batch.",
    )
    parser.add_argument("--hv_model_hidden_dim", type=int, default=256)
    parser.add_argument("--hv_model_num_layers", type=int, default=3)
    parser.add_argument("--hv_model_num_heads", type=int, default=8)
    parser.add_argument("--hv_model_dropout", type=float, default=0.1)
    parser.add_argument("--hv_init_target_count", type=int, default=2)
    parser.add_argument("--hv_sd_fov", type=float, default=64.0)
    parser.add_argument("--hv_sd_max_num_lanes", type=int, default=100)
    parser.add_argument("--hv_sd_num_points_per_lane", type=int, default=20)
    parser.add_argument("--hv_sd_map_id", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_generator(args):
    config = {
        "hv_init_target_count": args.hv_init_target_count,
        "hv_model_state_source": "identity",
        "hv_model_backbone": args.hv_model_backbone,
        "hv_model_autoencoder_checkpoint": args.hv_model_autoencoder_checkpoint,
        "hv_model_cfg_path": args.hv_model_cfg_path,
        "hv_model_hidden_dim": args.hv_model_hidden_dim,
        "hv_model_num_layers": args.hv_model_num_layers,
        "hv_model_num_heads": args.hv_model_num_heads,
        "hv_model_dropout": args.hv_model_dropout,
        "hv_sd_fov": args.hv_sd_fov,
        "hv_sd_max_num_lanes": args.hv_sd_max_num_lanes,
        "hv_sd_num_points_per_lane": args.hv_sd_num_points_per_lane,
        "hv_sd_map_id": args.hv_sd_map_id,
    }
    return ScenarioDreamerHighValueInitGenerator(
        checkpoint_path=args.hv_init_checkpoint,
        device=args.device,
        config=config,
    )


def main():
    args = parse_args()
    records = load_jsonl(args.jsonl)
    if not records:
        raise ValueError(f"no records found in {args.jsonl}")
    generator = make_generator(args)
    generator.model.eval()

    losses = []
    mask_losses = []
    score_losses = []
    exact = []
    recall = []
    precision = []
    score_mae = []

    with torch.no_grad():
        for batch in iter_batches(records, args.batch_size):
            af, cf, target_mask, target_score, valid_mask = build_feature_batch(batch, generator, generator.device)
            outputs = generator.model(af, cf, valid_mask=valid_mask)
            loss_dict = generator.model.loss(
                outputs,
                target_mask,
                target_score,
                valid_mask=valid_mask,
                score_weight=args.score_weight,
                mask_pos_weight=args.mask_pos_weight,
            )
            losses.append(float(loss_dict["loss"].detach().cpu()))
            mask_losses.append(float(loss_dict["mask_loss"].detach().cpu()))
            score_losses.append(float(loss_dict["score_loss"].detach().cpu()))
            batch_exact, batch_recall, batch_precision, batch_score_mae = topk_metrics(
                outputs,
                target_mask,
                target_score,
                valid_mask,
                args.hv_init_target_count,
            )
            exact.extend(batch_exact)
            recall.extend(batch_recall)
            precision.extend(batch_precision)
            score_mae.extend(batch_score_mae)

    metrics = {
        "num_records": len(records),
        "loss": sum(losses) / len(losses),
        "mask_loss": sum(mask_losses) / len(mask_losses),
        "score_loss": sum(score_losses) / len(score_losses),
        "exact_topk": sum(exact) / len(exact) if exact else None,
        "recall_at_k": sum(recall) / len(recall) if recall else None,
        "precision_at_k": sum(precision) / len(precision) if precision else None,
        "score_mae": sum(score_mae) / len(score_mae) if score_mae else None,
    }
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
