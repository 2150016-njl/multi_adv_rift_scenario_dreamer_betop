#!/usr/bin/env python
"""Build high-value actor-score records from RIFT collect-only dumps.

The script keeps RIFT as the interaction-label authority. It does not perturb
or regenerate actor states. Each record stores the original RIFT scene plus
actor-aligned mask/score supervision derived from RIFT interaction labels.
"""

import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_count", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--min_positive_score", type=float, default=0.0)
    parser.add_argument(
        "--drop_negative_scenes",
        action="store_true",
        default=False,
        help="Drop scenes with fewer than target_count RIFT-positive actors.",
    )
    parser.add_argument(
        "--augmentations_per_scene",
        type=int,
        default=1,
        help="Deprecated compatibility option; this builder no longer perturbs scenes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def scenario_key(scene):
    scenario = scene.get("scenario", {})
    return ":".join(str(scenario.get(key)) for key in ("town", "name", "index", "repetition_index"))


def split_name(key, val_ratio):
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(16 ** 8)
    return "val" if value < val_ratio else "train"


def rank_positive_labels(scene, min_positive_score):
    labels = scene.get("interaction_labels", [])
    ranked = []
    for index, label in enumerate(labels):
        if not label.get("is_interactive", False):
            continue
        score = float(label.get("score") or 0.0)
        if score < min_positive_score:
            continue
        distance_diff = label.get("distance_diff")
        if distance_diff is None:
            distance_diff = float("inf")
        ranked.append((index, label, score, float(distance_diff)))
    ranked.sort(key=lambda item: (-item[2], item[3], item[1].get("actor_id")))
    return ranked


def actor_aligned_scores(scene, min_positive_score):
    labels = scene.get("interaction_labels", [])
    scores = []
    for label in labels:
        score = float(label.get("score") or 0.0)
        if not label.get("is_interactive", False) or score < min_positive_score:
            score = 0.0
        scores.append(score)
    return scores


def build_record(scene, ranked_labels, args, source_path):
    if len(scene.get("actors", [])) != len(scene.get("interaction_labels", [])):
        raise ValueError(f"actor/label count mismatch in {source_path}")

    target_mask = [False for _ in scene["actors"]]
    selected = ranked_labels[:args.target_count]
    selected_actor_ids = []
    for actor_index, label, _score, _distance_diff in selected:
        target_mask[actor_index] = True
        selected_actor_ids.append(int(label["actor_id"]))

    target_scores = actor_aligned_scores(scene, args.min_positive_score)
    key = scenario_key(scene)
    return {
        "schema_version": "rift_hv_init_actor_score_v2",
        "source_path": str(source_path),
        "scenario_id": key,
        "scene": scene,
        "target_mask": target_mask,
        "target_scores": target_scores,
        "target_actor_ids": selected_actor_ids,
        "supervision": {
            "source": "rift_actor_score_labels",
            "positive_count": len(ranked_labels),
            "target_count": args.target_count,
            "min_positive_score": args.min_positive_score,
        },
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {"train": [], "val": []}
    stats = {
        "schema_version": "rift_hv_init_actor_score_v2",
        "num_dumps": 0,
        "num_valid_scenes": 0,
        "num_positive_scenes": 0,
        "num_kept_negative_scenes": 0,
        "num_skipped_negative_scenes": 0,
        "num_train_records": 0,
        "num_val_records": 0,
        "target_count": args.target_count,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "augmentations_per_scene_ignored": args.augmentations_per_scene,
    }

    for path in sorted(args.dump_dir.glob("*.json")):
        stats["num_dumps"] += 1
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        scene = payload.get("scene")
        if not scene or not scene.get("actors"):
            continue

        stats["num_valid_scenes"] += 1
        ranked = rank_positive_labels(scene, args.min_positive_score)
        enough_positive = len(ranked) >= args.target_count
        if enough_positive:
            stats["num_positive_scenes"] += 1
        elif args.drop_negative_scenes:
            stats["num_skipped_negative_scenes"] += 1
            continue
        else:
            stats["num_kept_negative_scenes"] += 1

        record = build_record(scene, ranked, args, path)
        split = split_name(scenario_key(scene), args.val_ratio)
        records[split].append(record)

    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    write_jsonl(train_path, records["train"])
    write_jsonl(val_path, records["val"])
    stats["num_train_records"] = len(records["train"])
    stats["num_val_records"] = len(records["val"])

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
