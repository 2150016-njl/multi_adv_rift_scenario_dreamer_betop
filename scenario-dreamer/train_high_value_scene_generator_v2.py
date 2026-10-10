#!/usr/bin/env python
"""Train v2 high-value scene generator with weighted high-quality records.

This is a side-by-side trainer.  It does not change the Phase 5 trainer or the
model class defaults.  The main v2 improvement is explicit high-quality record
weighting via weighted sampling, combined with stronger decoded-state losses.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from high_value_init.high_value_ldm import RouteConditionedHighValueLDM
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inpainting", "full_generation"], default="inpainting")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--route_num_points", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=-1,
                        help="Weighted batches per epoch. -1 uses ceil(num_train/batch_size).")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--decoded_state_loss_weight", type=float, default=0.4)
    parser.add_argument("--decoded_position_loss_weight", type=float, default=2.0)
    parser.add_argument("--decoded_heading_loss_weight", type=float, default=0.7)
    parser.add_argument("--decoded_speed_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoded_size_loss_weight", type=float, default=0.5)
    parser.add_argument("--physical_bounds_loss_weight", type=float, default=0.3)
    parser.add_argument("--min_generated_speed", type=float, default=0.0)
    parser.add_argument("--max_generated_speed", type=float, default=30.0)
    parser.add_argument("--min_generated_length", type=float, default=2.0)
    parser.add_argument("--max_generated_length", type=float, default=8.0)
    parser.add_argument("--min_generated_width", type=float, default=1.4)
    parser.add_argument("--max_generated_width", type=float, default=3.2)
    parser.add_argument("--clamp_generated_states", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--route_dropout", type=float, default=0.0)
    parser.add_argument("--sample_encoder_latents", action="store_true", default=False)
    parser.add_argument("--max_train_records", type=int, default=-1)
    parser.add_argument("--max_val_records", type=int, default=-1)
    parser.add_argument("--record_weight_key", default="training_weight")
    parser.add_argument("--no_weighted_sampling", action="store_true", default=False)
    parser.add_argument("--min_record_weight", type=float, default=0.25)
    parser.add_argument("--max_record_weight", type=float, default=5.0)
    parser.add_argument("--best_metric", default="val_decoded_position_l1",
                        choices=["val_loss", "val_decoded_position_l1", "val_decoded_state_loss"])
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    return parser.parse_args()


def validate_args(args) -> None:
    if not str(args.output_dir).strip():
        raise ValueError("--output_dir must be a non-empty path")
    for name in (
        "decoded_state_loss_weight",
        "decoded_position_loss_weight",
        "decoded_heading_loss_weight",
        "decoded_speed_loss_weight",
        "decoded_size_loss_weight",
        "physical_bounds_loss_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name} must be >= 0")
    if args.physical_bounds_loss_weight > 0 and args.decoded_state_loss_weight <= 0:
        raise ValueError("--physical_bounds_loss_weight requires --decoded_state_loss_weight > 0")
    if args.min_record_weight <= 0 or args.max_record_weight < args.min_record_weight:
        raise ValueError("record weight bounds are invalid")
    if args.min_generated_speed > args.max_generated_speed:
        raise ValueError("--min_generated_speed must be <= --max_generated_speed")
    if args.min_generated_length > args.max_generated_length:
        raise ValueError("--min_generated_length must be <= --max_generated_length")
    if args.min_generated_width > args.max_generated_width:
        raise ValueError("--min_generated_width must be <= --max_generated_width")


def load_records(path: str, limit: int) -> List[Dict]:
    records = read_jsonl(path)
    if limit and limit > 0:
        records = records[:limit]
    return records


def record_weight(record: Dict, args) -> float:
    value = record.get(args.record_weight_key, 1.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 1.0
    return min(max(value, args.min_record_weight), args.max_record_weight)


def iter_batches(records: List[Dict], batch_size: int, rng: random.Random = None):
    order = list(range(len(records)))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[idx] for idx in order[start:start + batch_size]]


def iter_weighted_batches(records: List[Dict], batch_size: int, steps: int, rng: random.Random, args):
    weights = [record_weight(record, args) for record in records]
    indices = list(range(len(records)))
    for _ in range(steps):
        chosen = rng.choices(indices, weights=weights, k=batch_size)
        yield [records[idx] for idx in chosen]


@torch.no_grad()
def evaluate(model: RouteConditionedHighValueLDM, records: List[Dict], batch_size: int) -> Dict[str, float]:
    if not records:
        return {key: float("nan") for key in (
            "loss", "agent_loss", "lane_loss", "decoded_state_loss", "decoded_position_l1",
            "decoded_heading_l1", "decoded_speed_l1", "decoded_size_l1",
            "decoded_physical_bounds_loss",
        )}
    model.eval()
    buckets = {
        "loss": [],
        "agent_loss": [],
        "lane_loss": [],
        "decoded_state_loss": [],
        "decoded_position_l1": [],
        "decoded_heading_l1": [],
        "decoded_speed_l1": [],
        "decoded_size_l1": [],
        "decoded_physical_bounds_loss": [],
    }
    for batch in iter_batches(records, batch_size):
        loss_dict = model.training_loss(batch)
        for key in buckets:
            buckets[key].append(float(loss_dict[key].detach().cpu()))
    return {key: float(np.mean(values)) for key, values in buckets.items()}


def make_model(args) -> RouteConditionedHighValueLDM:
    return RouteConditionedHighValueLDM(
        checkpoint_path=args.resume_checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode=args.mode,
        num_generate_agents=args.num_generate_agents,
        route_num_points=args.route_num_points,
        route_dropout=args.route_dropout,
        decoded_state_loss_weight=args.decoded_state_loss_weight,
        decoded_position_loss_weight=args.decoded_position_loss_weight,
        decoded_heading_loss_weight=args.decoded_heading_loss_weight,
        decoded_speed_loss_weight=args.decoded_speed_loss_weight,
        decoded_size_loss_weight=args.decoded_size_loss_weight,
        physical_bounds_loss_weight=args.physical_bounds_loss_weight,
        min_generated_speed=args.min_generated_speed,
        max_generated_speed=args.max_generated_speed,
        min_generated_length=args.min_generated_length,
        max_generated_length=args.max_generated_length,
        min_generated_width=args.min_generated_width,
        max_generated_width=args.max_generated_width,
        clamp_generated_states=args.clamp_generated_states,
        use_encoder_mean=not args.sample_encoder_latents,
    )


def metadata(args, train_records, val_records):
    weights = [record_weight(record, args) for record in train_records]
    return {
        "trainer": "train_high_value_scene_generator_v2",
        "mode": args.mode,
        "resume_checkpoint": args.resume_checkpoint,
        "base_ldm_checkpoint": args.ldm_checkpoint,
        "autoencoder_checkpoint": args.autoencoder_checkpoint,
        "num_generate_agents": args.num_generate_agents,
        "route_num_points": args.route_num_points,
        "loss_weights": {
            "decoded_state": args.decoded_state_loss_weight,
            "decoded_position": args.decoded_position_loss_weight,
            "decoded_heading": args.decoded_heading_loss_weight,
            "decoded_speed": args.decoded_speed_loss_weight,
            "decoded_size": args.decoded_size_loss_weight,
            "physical_bounds": args.physical_bounds_loss_weight,
        },
        "weighted_sampling": not args.no_weighted_sampling,
        "record_weight_key": args.record_weight_key,
        "record_weight_stats": {
            "min": min(weights) if weights else None,
            "mean": sum(weights) / len(weights) if weights else None,
            "max": max(weights) if weights else None,
        },
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
    }


def metric_value(row: Dict, name: str) -> float:
    if name == "val_loss":
        return float(row["val_loss"])
    if name == "val_decoded_position_l1":
        return float(row["val_decoded_position_l1"])
    if name == "val_decoded_state_loss":
        return float(row["val_decoded_state_loss"])
    raise KeyError(name)


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_records = load_records(args.train_jsonl, args.max_train_records)
    val_records = load_records(args.val_jsonl, args.max_val_records)
    if not train_records:
        raise ValueError(f"no training records in {args.train_jsonl}")
    if not val_records:
        raise ValueError(f"no validation records in {args.val_jsonl}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    steps = args.steps_per_epoch if args.steps_per_epoch > 0 else int(np.ceil(len(train_records) / args.batch_size))
    best_value = float("inf")
    history = []
    meta = metadata(args, train_records, val_records)
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "steps_per_epoch_resolved": steps, "metadata": meta}, f, indent=2)

    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        buckets = {
            "loss": [],
            "decoded_state_loss": [],
            "decoded_position_l1": [],
            "decoded_heading_l1": [],
            "decoded_speed_l1": [],
            "decoded_size_l1": [],
            "decoded_physical_bounds_loss": [],
        }
        if args.no_weighted_sampling:
            train_iter = iter_batches(train_records, args.batch_size, rng=rng)
        else:
            train_iter = iter_weighted_batches(train_records, args.batch_size, steps, rng, args)
        for batch_records in train_iter:
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.training_loss(batch_records)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.ldm.parameters(), args.grad_clip_norm)
            optimizer.step()
            for key in buckets:
                buckets[key].append(float(loss_dict[key].detach().cpu()))

        val_metrics = evaluate(model, val_records, args.batch_size)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(buckets["loss"])),
            "train_decoded_state_loss": float(np.mean(buckets["decoded_state_loss"])),
            "train_decoded_position_l1": float(np.mean(buckets["decoded_position_l1"])),
            "train_decoded_heading_l1": float(np.mean(buckets["decoded_heading_l1"])),
            "train_decoded_speed_l1": float(np.mean(buckets["decoded_speed_l1"])),
            "train_decoded_size_l1": float(np.mean(buckets["decoded_size_l1"])),
            "train_decoded_physical_bounds_loss": float(np.mean(buckets["decoded_physical_bounds_loss"])),
            "val_loss": val_metrics["loss"],
            "val_agent_loss": val_metrics["agent_loss"],
            "val_lane_loss": val_metrics["lane_loss"],
            "val_decoded_state_loss": val_metrics["decoded_state_loss"],
            "val_decoded_position_l1": val_metrics["decoded_position_l1"],
            "val_decoded_heading_l1": val_metrics["decoded_heading_l1"],
            "val_decoded_speed_l1": val_metrics["decoded_speed_l1"],
            "val_decoded_size_l1": val_metrics["decoded_size_l1"],
            "val_decoded_physical_bounds_loss": val_metrics["decoded_physical_bounds_loss"],
            "seconds": time.time() - start,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        model.save_checkpoint(str(output_dir / "last.ckpt"), metadata={**meta, "last_epoch": epoch})
        current = metric_value(row, args.best_metric)
        if current < best_value:
            best_value = current
            model.save_checkpoint(str(output_dir / "best.ckpt"), metadata={
                **meta,
                "best_epoch": epoch,
                "best_metric": args.best_metric,
                "best_metric_value": best_value,
            })
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(json.dumps({
        "output_dir": str(output_dir),
        "best_metric": args.best_metric,
        "best_metric_value": best_value,
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
    }, indent=2))


if __name__ == "__main__":
    main()
