#!/usr/bin/env python
"""Train the high-value actor scoring head.

Training uses RIFT collect-only labels as supervision. It does not train a
delta-restoration task. State generation is handled by Scenario Dreamer LDM at
online inference time.
"""

import argparse
import json
from pathlib import Path
from random import Random

import torch

from high_value_init.rift_generator import ScenarioDreamerHighValueInitGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--val_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hv_model_backbone", type=str, default="autoencoder", choices=["autoencoder", "adapter_only"])
    parser.add_argument("--hv_model_autoencoder_checkpoint", type=str, default=None)
    parser.add_argument("--hv_model_cfg_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument(
        "--mask_pos_weight",
        type=float,
        default=None,
        help="Positive weight for mask BCE. Default None computes negatives/positives per batch.",
    )
    parser.add_argument("--seed", type=int, default=0)
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


def iter_batches(records, batch_size, rng=None):
    order = list(range(len(records)))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[idx] for idx in order[start:start + batch_size]]


@torch.no_grad()
def build_feature_batch(records, generator, device):
    actor_features = []
    context_features = []
    target_masks = []
    target_scores = []
    valid_masks = []
    max_agents = 0

    for record in records:
        af, cf, actor_ids = generator._build_features(record["scene"])
        af = af.squeeze(0).detach()
        cf = cf.squeeze(0).detach()
        target_mask = torch.tensor(record["target_mask"], dtype=torch.bool)
        target_score = torch.tensor(
            record.get("target_scores", [float(x) for x in record["target_mask"]]),
            dtype=torch.float32,
        )
        if af.shape[0] != len(actor_ids):
            raise ValueError("feature actor count does not match actor_ids")
        if af.shape[0] != cf.shape[0]:
            raise ValueError("actor/context feature counts are not aligned")
        if af.shape[0] != target_mask.shape[0] or af.shape[0] != target_score.shape[0]:
            raise ValueError("record target tensors are not aligned with scene actors")

        actor_features.append(af)
        context_features.append(cf)
        target_masks.append(target_mask)
        target_scores.append(target_score)
        max_agents = max(max_agents, af.shape[0])

    padded_af = []
    padded_cf = []
    padded_mask = []
    padded_score = []
    for af, cf, target_mask, target_score in zip(actor_features, context_features, target_masks, target_scores):
        num_agents = af.shape[0]
        pad_agents = max_agents - num_agents
        valid = torch.zeros(max_agents, dtype=torch.bool)
        valid[:num_agents] = True
        if pad_agents:
            af = torch.cat([af, torch.zeros(pad_agents, af.shape[-1], dtype=af.dtype, device=af.device)], dim=0)
            cf = torch.cat([cf, torch.zeros(pad_agents, cf.shape[-1], dtype=cf.dtype, device=cf.device)], dim=0)
            target_mask = torch.cat([target_mask, torch.zeros(pad_agents, dtype=torch.bool)], dim=0)
            target_score = torch.cat([target_score, torch.zeros(pad_agents, dtype=torch.float32)], dim=0)
        padded_af.append(af)
        padded_cf.append(cf)
        padded_mask.append(target_mask)
        padded_score.append(target_score)
        valid_masks.append(valid)

    return (
        torch.stack(padded_af, dim=0).to(device),
        torch.stack(padded_cf, dim=0).to(device),
        torch.stack(padded_mask, dim=0).to(device),
        torch.stack(padded_score, dim=0).to(device),
        torch.stack(valid_masks, dim=0).to(device),
    )


def make_generator(args):
    config = {
        "hv_init_target_count": args.hv_init_target_count,
        "hv_init_allow_untrained_generator": True,
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
        checkpoint_path=None,
        device=args.device,
        config=config,
    )


def topk_metrics(outputs, target_mask, target_scores, valid_mask, target_count):
    exact_matches = []
    recalls = []
    precisions = []
    score_mae = []
    for logits, pred_score, expected, expected_score, valid in zip(
            outputs["target_mask_logits"],
            outputs["actor_scores"],
            target_mask,
            target_scores,
            valid_mask):
        num_valid = int(valid.sum().item())
        if num_valid == 0:
            continue
        k = min(target_count, num_valid)
        selected = torch.topk(logits[:num_valid], k=k).indices
        pred = torch.zeros(num_valid, dtype=torch.bool, device=logits.device)
        pred[selected] = True
        gt = expected[:num_valid].bool()
        gt_count = int(gt.sum().item())
        if gt_count > 0:
            hits = int((pred & gt).sum().item())
            exact_matches.append(float(torch.equal(pred, gt)))
            recalls.append(hits / gt_count)
            precisions.append(hits / max(k, 1))
        score_mae.append(
            float(torch.abs(pred_score[:num_valid] - expected_score[:num_valid]).mean().detach().cpu())
        )
    return exact_matches, recalls, precisions, score_mae


def evaluate(records, generator, batch_size, score_weight, mask_pos_weight, target_count):
    if not records:
        return {
            "loss": None,
            "mask_loss": None,
            "score_loss": None,
            "exact_topk": None,
            "recall_at_k": None,
            "precision_at_k": None,
            "score_mae": None,
        }
    model = generator.model
    model.eval()
    losses = []
    mask_losses = []
    score_losses = []
    exact_matches = []
    recalls = []
    precisions = []
    score_mae = []
    with torch.no_grad():
        for batch in iter_batches(records, batch_size):
            af, cf, target_mask, target_score, valid_mask = build_feature_batch(batch, generator, generator.device)
            outputs = model(af, cf, valid_mask=valid_mask)
            loss_dict = model.loss(
                outputs,
                target_mask,
                target_score,
                valid_mask=valid_mask,
                score_weight=score_weight,
                mask_pos_weight=mask_pos_weight,
            )
            losses.append(float(loss_dict["loss"].detach().cpu()))
            mask_losses.append(float(loss_dict["mask_loss"].detach().cpu()))
            score_losses.append(float(loss_dict["score_loss"].detach().cpu()))
            batch_exact, batch_recall, batch_precision, batch_score_mae = topk_metrics(
                outputs,
                target_mask,
                target_score,
                valid_mask,
                target_count,
            )
            exact_matches.extend(batch_exact)
            recalls.extend(batch_recall)
            precisions.extend(batch_precision)
            score_mae.extend(batch_score_mae)
    return {
        "loss": sum(losses) / len(losses),
        "mask_loss": sum(mask_losses) / len(mask_losses),
        "score_loss": sum(score_losses) / len(score_losses),
        "exact_topk": sum(exact_matches) / len(exact_matches) if exact_matches else None,
        "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "precision_at_k": sum(precisions) / len(precisions) if precisions else None,
        "score_mae": sum(score_mae) / len(score_mae) if score_mae else None,
    }


def save_checkpoint(path, generator, args, epoch, metrics):
    payload = {
        "hv_head_state_dict": generator.model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": vars(args),
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = Random(args.seed)

    train_records = load_jsonl(args.train_jsonl)
    val_records = load_jsonl(args.val_jsonl)
    if not train_records:
        raise ValueError(f"no training records found in {args.train_jsonl}")

    generator = make_generator(args)
    optimizer = torch.optim.AdamW(
        generator.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -1.0
    history = []
    for epoch in range(args.epochs):
        generator.model.train()
        train_losses = []
        for batch in iter_batches(train_records, args.batch_size, rng=rng):
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
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            optimizer.step()
            train_losses.append(float(loss_dict["loss"].detach().cpu()))

        metrics = evaluate(
            val_records,
            generator,
            args.batch_size,
            args.score_weight,
            args.mask_pos_weight,
            args.hv_init_target_count,
        )
        metrics["train_loss"] = sum(train_losses) / len(train_losses)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        score = metrics["recall_at_k"]
        if score is None:
            score = -metrics["train_loss"]
        if score > best_metric:
            best_metric = score
            save_checkpoint(args.output_dir / "best_hv_head.ckpt", generator, args, epoch, metrics)
        save_checkpoint(args.output_dir / "last_hv_head.ckpt", generator, args, epoch, metrics)

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(json.dumps({"best_metric": best_metric, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
