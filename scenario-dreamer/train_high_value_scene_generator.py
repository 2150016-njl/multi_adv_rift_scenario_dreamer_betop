#!/usr/bin/env python
"""Train route-conditioned high-value initial-state scene generator."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--decoded_state_loss_weight", type=float, default=0.1)
    parser.add_argument("--decoded_position_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoded_heading_loss_weight", type=float, default=0.5)
    parser.add_argument("--decoded_speed_loss_weight", type=float, default=0.5)
    parser.add_argument("--decoded_size_loss_weight", type=float, default=0.2)
    parser.add_argument("--physical_bounds_loss_weight", type=float, default=0.0)
    parser.add_argument("--min_generated_speed", type=float, default=0.0)
    parser.add_argument("--max_generated_speed", type=float, default=30.0)
    parser.add_argument("--min_generated_length", type=float, default=2.0)
    parser.add_argument("--max_generated_length", type=float, default=8.0)
    parser.add_argument("--min_generated_width", type=float, default=1.4)
    parser.add_argument("--max_generated_width", type=float, default=3.2)
    parser.add_argument("--clamp_generated_states", action="store_true", default=False,
                        help="Clamp generated speed/length/width during model.generate(). Does not affect training loss.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--route_dropout", type=float, default=0.0)
    parser.add_argument("--sample_encoder_latents", action="store_true", default=False)
    parser.add_argument("--freeze_autoencoder", action="store_true", default=True,
                        help="Kept for command compatibility; the Scenario Dreamer AE is always frozen.")
    parser.add_argument("--max_train_records", type=int, default=-1)
    parser.add_argument("--max_val_records", type=int, default=-1)
    parser.add_argument("--no_rift_score_aux", action="store_true", default=False)
    parser.add_argument("--rift_score_aux_weight", type=float, default=0.0)
    return parser.parse_args()


def validate_args(args) -> None:
    weighted_args = [
        "decoded_state_loss_weight",
        "decoded_position_loss_weight",
        "decoded_heading_loss_weight",
        "decoded_speed_loss_weight",
        "decoded_size_loss_weight",
        "physical_bounds_loss_weight",
    ]
    for name in weighted_args:
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name} must be >= 0")
    if args.min_generated_speed > args.max_generated_speed:
        raise ValueError("--min_generated_speed must be <= --max_generated_speed")
    if args.min_generated_length > args.max_generated_length:
        raise ValueError("--min_generated_length must be <= --max_generated_length")
    if args.min_generated_width > args.max_generated_width:
        raise ValueError("--min_generated_width must be <= --max_generated_width")
    if args.physical_bounds_loss_weight > 0.0 and args.decoded_state_loss_weight <= 0.0:
        raise ValueError("--physical_bounds_loss_weight requires --decoded_state_loss_weight > 0")


def iter_batches(records: List[Dict], batch_size: int, rng: random.Random = None):
    order = list(range(len(records)))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[idx] for idx in order[start:start + batch_size]]


def load_records(path: str, limit: int) -> List[Dict]:
    records = read_jsonl(path)
    if limit and limit > 0:
        records = records[:limit]
    return records


@torch.no_grad()
def evaluate(model: RouteConditionedHighValueLDM, records: List[Dict], batch_size: int) -> Dict[str, float]:
    if not records:
        return {
            "loss": float("nan"),
            "agent_loss": float("nan"),
            "lane_loss": float("nan"),
            "decoded_state_loss": float("nan"),
            "decoded_position_l1": float("nan"),
            "decoded_heading_l1": float("nan"),
            "decoded_speed_l1": float("nan"),
            "decoded_size_l1": float("nan"),
            "decoded_physical_bounds_loss": float("nan"),
        }
    model.eval()
    losses = []
    agent_losses = []
    lane_losses = []
    decoded_state_losses = []
    decoded_position_losses = []
    decoded_heading_losses = []
    decoded_speed_losses = []
    decoded_size_losses = []
    decoded_physical_bounds_losses = []
    for batch in iter_batches(records, batch_size):
        loss_dict = model.training_loss(batch)
        losses.append(float(loss_dict["loss"].detach().cpu()))
        agent_losses.append(float(loss_dict["agent_loss"].detach().cpu()))
        lane_losses.append(float(loss_dict["lane_loss"].detach().cpu()))
        decoded_state_losses.append(float(loss_dict["decoded_state_loss"].detach().cpu()))
        decoded_position_losses.append(float(loss_dict["decoded_position_l1"].detach().cpu()))
        decoded_heading_losses.append(float(loss_dict["decoded_heading_l1"].detach().cpu()))
        decoded_speed_losses.append(float(loss_dict["decoded_speed_l1"].detach().cpu()))
        decoded_size_losses.append(float(loss_dict["decoded_size_l1"].detach().cpu()))
        decoded_physical_bounds_losses.append(float(loss_dict["decoded_physical_bounds_loss"].detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "agent_loss": float(np.mean(agent_losses)),
        "lane_loss": float(np.mean(lane_losses)),
        "decoded_state_loss": float(np.mean(decoded_state_losses)),
        "decoded_position_l1": float(np.mean(decoded_position_losses)),
        "decoded_heading_l1": float(np.mean(decoded_heading_losses)),
        "decoded_speed_l1": float(np.mean(decoded_speed_losses)),
        "decoded_size_l1": float(np.mean(decoded_size_losses)),
        "decoded_physical_bounds_loss": float(np.mean(decoded_physical_bounds_losses)),
    }


def main():
    args = parse_args()
    validate_args(args)
    if args.rift_score_aux_weight != 0.0:
        raise NotImplementedError("RIFT score auxiliary loss is intentionally not implemented in the MVP.")

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
    model = RouteConditionedHighValueLDM(
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = float("inf")
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_decoded_state_losses = []
        epoch_decoded_position_losses = []
        epoch_decoded_heading_losses = []
        epoch_decoded_speed_losses = []
        epoch_decoded_size_losses = []
        epoch_decoded_physical_bounds_losses = []
        start = time.time()
        for batch_records in iter_batches(train_records, args.batch_size, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.training_loss(batch_records)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.ldm.parameters(), 10.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_decoded_state_losses.append(float(loss_dict["decoded_state_loss"].detach().cpu()))
            epoch_decoded_position_losses.append(float(loss_dict["decoded_position_l1"].detach().cpu()))
            epoch_decoded_heading_losses.append(float(loss_dict["decoded_heading_l1"].detach().cpu()))
            epoch_decoded_speed_losses.append(float(loss_dict["decoded_speed_l1"].detach().cpu()))
            epoch_decoded_size_losses.append(float(loss_dict["decoded_size_l1"].detach().cpu()))
            epoch_decoded_physical_bounds_losses.append(float(loss_dict["decoded_physical_bounds_loss"].detach().cpu()))

        val_metrics = evaluate(model, val_records, args.batch_size)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_decoded_state_loss": float(np.mean(epoch_decoded_state_losses)) if epoch_decoded_state_losses else float("nan"),
            "train_decoded_position_l1": float(np.mean(epoch_decoded_position_losses)) if epoch_decoded_position_losses else float("nan"),
            "train_decoded_heading_l1": float(np.mean(epoch_decoded_heading_losses)) if epoch_decoded_heading_losses else float("nan"),
            "train_decoded_speed_l1": float(np.mean(epoch_decoded_speed_losses)) if epoch_decoded_speed_losses else float("nan"),
            "train_decoded_size_l1": float(np.mean(epoch_decoded_size_losses)) if epoch_decoded_size_losses else float("nan"),
            "train_decoded_physical_bounds_loss": (
                float(np.mean(epoch_decoded_physical_bounds_losses))
                if epoch_decoded_physical_bounds_losses else float("nan")
            ),
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

        metadata = {
            "mode": args.mode,
            "base_ldm_checkpoint": args.ldm_checkpoint,
            "autoencoder_checkpoint": args.autoencoder_checkpoint,
            "num_generate_agents": args.num_generate_agents,
            "route_num_points": args.route_num_points,
            "decoded_state_loss_weight": args.decoded_state_loss_weight,
            "decoded_position_loss_weight": args.decoded_position_loss_weight,
            "decoded_heading_loss_weight": args.decoded_heading_loss_weight,
            "decoded_speed_loss_weight": args.decoded_speed_loss_weight,
            "decoded_size_loss_weight": args.decoded_size_loss_weight,
            "physical_bounds_loss_weight": args.physical_bounds_loss_weight,
            "generated_state_bounds": {
                "speed": [args.min_generated_speed, args.max_generated_speed],
                "length": [args.min_generated_length, args.max_generated_length],
                "width": [args.min_generated_width, args.max_generated_width],
            },
            "clamp_generated_states": args.clamp_generated_states,
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
        }
        model.save_checkpoint(str(output_dir / "last.ckpt"), metadata=metadata)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            model.save_checkpoint(str(output_dir / "best.ckpt"), metadata={**metadata, "best_epoch": epoch})

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(json.dumps({
        "output_dir": str(output_dir),
        "best_val_loss": best_val,
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
    }, indent=2))


if __name__ == "__main__":
    main()
