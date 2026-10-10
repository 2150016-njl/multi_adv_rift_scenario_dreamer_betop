"""Train the v3 high-value generator without modifying the v2 trainer."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from high_value_init.high_value_ldm_v3 import RouteConditionedHighValueLDMV3
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--route_num_points", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps_per_epoch", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoded_state_loss_weight", type=float, default=0.4)
    parser.add_argument("--decoded_position_loss_weight", type=float, default=2.0)
    parser.add_argument("--decoded_heading_loss_weight", type=float, default=0.7)
    parser.add_argument("--decoded_speed_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoded_size_loss_weight", type=float, default=0.5)
    parser.add_argument("--physical_bounds_loss_weight", type=float, default=0.3)
    # First V3 ablation stays reconstruction-led. Heading-to-lane is kept as
    # an opt-in experiment because connector and crossing interactions can
    # have locally ambiguous lane tangents.
    parser.add_argument("--lane_corridor_loss_weight", type=float, default=0.10)
    parser.add_argument("--lane_heading_loss_weight", type=float, default=0.0)
    parser.add_argument("--initial_clearance_loss_weight", type=float, default=0.10)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument("--record_weight_key", default="training_weight")
    parser.add_argument("--no_weighted_sampling", action="store_true", default=False)
    parser.add_argument("--best_metric", choices=["val_decoded_position_l1", "val_loss"], default="val_decoded_position_l1")
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    return parser.parse_args()


def weight(record, key):
    try:
        return min(max(float(record.get(key, 1.0)), 0.25), 5.0)
    except (TypeError, ValueError):
        return 1.0


def batches(records, batch_size, rng, weighted, key, steps):
    indices = list(range(len(records)))
    if weighted:
        weights = [weight(record, key) for record in records]
        for _ in range(steps):
            yield [records[idx] for idx in rng.choices(indices, weights=weights, k=batch_size)]
        return
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [records[idx] for idx in indices[start:start + batch_size]]


def make_model(args):
    return RouteConditionedHighValueLDMV3(
        checkpoint_path=args.resume_checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode="inpainting",
        num_generate_agents=args.num_generate_agents,
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
    )


@torch.no_grad()
def evaluate(model, records, batch_size):
    model.eval()
    rows = []
    rng = random.Random(0)
    for batch in batches(records, batch_size, rng, False, "", 0):
        metrics = model.training_loss(batch)
        rows.append({key: float(value.detach().cpu()) for key, value in metrics.items() if torch.is_tensor(value) and value.ndim == 0})
    keys = {key for row in rows for key in row}
    return {f"val_{key}": float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def main():
    args = parse_args()
    if not args.output_dir.strip() or args.num_generate_agents != 2:
        raise ValueError("v3 currently trains the validated two-target task; --output_dir must be non-empty")
    train, val = read_jsonl(args.train_jsonl), read_jsonl(args.val_jsonl)
    if not train or not val:
        raise ValueError("train and val JSONL files must be non-empty")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    model = make_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    steps = args.steps_per_epoch if args.steps_per_epoch > 0 else int(np.ceil(len(train) / args.batch_size))
    config = {**vars(args), "num_train_records": len(train), "num_val_records": len(val), "steps_per_epoch_resolved": steps}
    (output / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, history = float("inf"), []
    metric_keys = ("loss", "decoded_state_loss", "decoded_position_l1", "decoded_heading_l1", "decoded_speed_l1", "decoded_size_l1", "decoded_physical_bounds_loss", "lane_corridor_loss", "lane_heading_loss", "initial_clearance_loss")
    for epoch in range(args.epochs):
        model.train(); started = time.time(); values = {key: [] for key in metric_keys}
        for batch in batches(train, args.batch_size, rng, not args.no_weighted_sampling, args.record_weight_key, steps):
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_loss(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            for key in metric_keys:
                if key in losses: values[key].append(float(losses[key].detach().cpu()))
        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update({f"train_{key}": float(np.mean(value)) for key, value in values.items() if value})
        row.update(evaluate(model, val, args.batch_size))
        history.append(row); print(json.dumps(row, indent=2))
        model.save_checkpoint(str(output / "last.ckpt"), metadata={"trainer": "v3", "last_epoch": epoch, "config": config})
        current = float(row["val_decoded_position_l1"] if args.best_metric == "val_decoded_position_l1" else row["val_loss"])
        if current < best:
            best = current
            model.save_checkpoint(str(output / "best.ckpt"), metadata={"trainer": "v3", "best_epoch": epoch, "best_metric": args.best_metric, "best_metric_value": best, "config": config})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "best_metric": args.best_metric, "best_metric_value": best}, indent=2))


if __name__ == "__main__":
    main()
