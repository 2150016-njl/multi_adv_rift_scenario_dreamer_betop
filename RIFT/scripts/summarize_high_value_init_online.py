#!/usr/bin/env python
"""Summarize online high-value initialization dumps and optional RIFT metrics."""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=Path, required=True)
    parser.add_argument(
        "--statistics_json",
        type=Path,
        default=None,
        help="Optional RIFT simulation_results.json to merge route/collision metrics.",
    )
    parser.add_argument(
        "--statistics_dir",
        type=Path,
        default=None,
        help="Optional directory searched recursively for the latest simulation_results.json.",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def load_payloads(dump_dir):
    payloads = []
    for path in sorted(dump_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def numeric_summary(values):
    values = [float(value) for value in values if value is not None]
    return {
        "min": min(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def summarize(payloads):
    failure_reasons = Counter()
    selected_counts = []
    cbv_counts = []
    positive_counts = []
    post_positive_counts = []
    target_counts = []
    apply_top_ks = []
    generation_sources = Counter()
    generation_candidate_counts = []
    validation_passed = 0
    successes = 0
    collect_only = 0

    for payload in payloads:
        scene = payload.get("scene", {})
        result = payload.get("result") or {}
        proposal = payload.get("proposal") or {}
        labels = scene.get("interaction_labels", [])
        positive_count = sum(1 for label in labels if label.get("is_interactive", False))
        positive_counts.append(positive_count)
        if result.get("num_post_interactive_labels") is not None:
            post_positive_counts.append(int(result["num_post_interactive_labels"]))
        target_counts.append(int(scene.get("target_count", result.get("target_count", 2))))
        apply_top_ks.append(int(scene.get("apply_top_k", result.get("apply_top_k", 2))))
        selected_counts.append(len(result.get("selected_actor_ids") or []))
        cbv_counts.append(len(result.get("cbv_actor_ids") or []))
        successes += int(result.get("success", False))
        collect_only += int(result.get("collect_only", False))
        validation_passed += int(result.get("rift_validation_passed", False))
        if proposal.get("state_generation_source"):
            generation_sources[str(proposal["state_generation_source"])] += 1
        if proposal.get("num_generation_candidates") is not None:
            generation_candidate_counts.append(int(proposal["num_generation_candidates"]))
        if result.get("failure_reason"):
            failure_reasons[str(result["failure_reason"])] += 1

    total = len(payloads)
    two_positive = sum(
        int(pos_count >= target_count)
        for pos_count, target_count in zip(positive_counts, target_counts)
    )
    return {
        "num_dumps": total,
        "num_collect_only": collect_only,
        "num_success": successes,
        "success_ratio": successes / total if total else 0.0,
        "num_rift_validation_passed": validation_passed,
        "rift_validation_passed_ratio": validation_passed / total if total else 0.0,
        "num_two_positive_prelabels": two_positive,
        "two_positive_prelabel_ratio": two_positive / total if total else 0.0,
        "positive_count": numeric_summary(positive_counts),
        "post_positive_count": numeric_summary(post_positive_counts),
        "selected_count": numeric_summary(selected_counts),
        "cbv_count": numeric_summary(cbv_counts),
        "apply_top_k": numeric_summary(apply_top_ks),
        "generation_sources": dict(generation_sources.most_common()),
        "generation_candidate_count": numeric_summary(generation_candidate_counts),
        "failure_reasons": dict(failure_reasons.most_common()),
    }


def resolve_statistics_json(args):
    if args.statistics_json:
        return args.statistics_json
    if not args.statistics_dir:
        return None
    candidates = sorted(
        args.statistics_dir.rglob("simulation_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def count_record_collisions(record):
    infractions = record.get("infractions", {})
    keys = ("collisions_pedestrian", "collisions_vehicle", "collisions_layout")
    return sum(len(infractions.get(key, []) or []) for key in keys)


def summarize_rift_statistics(path):
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    checkpoint = data.get("_checkpoint", {})
    global_record = checkpoint.get("global_record", {})
    records = checkpoint.get("records", [])
    scores_mean = global_record.get("scores_mean", {})
    scores_std = global_record.get("scores_std_dev", {})
    infractions = global_record.get("infractions", {})
    meta = global_record.get("meta", {})

    route_completion = [
        float(record.get("scores", {}).get("score_route", 0.0))
        for record in records
    ]
    driving_score = [
        float(record.get("scores", {}).get("score_composed", 0.0))
        for record in records
    ]
    route_collision_counts = [count_record_collisions(record) for record in records]
    status_counts = Counter(str(record.get("status", "unknown")) for record in records)
    num_collision_routes = sum(1 for count in route_collision_counts if count > 0)
    num_records = len(records)

    return {
        "statistics_json": str(path),
        "entry_status": data.get("entry_status"),
        "eligible": data.get("eligible"),
        "num_routes": num_records,
        "status_counts": dict(status_counts.most_common()),
        "score_mean": {
            "driving_score": scores_mean.get("score_composed"),
            "route_completion": scores_mean.get("score_route"),
            "infraction_penalty": scores_mean.get("score_penalty"),
        },
        "score_std": {
            "driving_score": scores_std.get("score_composed"),
            "route_completion": scores_std.get("score_route"),
            "infraction_penalty": scores_std.get("score_penalty"),
        },
        "route_completion": numeric_summary(route_completion),
        "driving_score": numeric_summary(driving_score),
        "collision_routes": {
            "num_routes_with_collision": num_collision_routes,
            "route_collision_ratio": num_collision_routes / num_records if num_records else 0.0,
            "collision_count_per_route": numeric_summary(route_collision_counts),
        },
        "collision_per_km": {
            "pedestrian": infractions.get("collisions_pedestrian"),
            "vehicle": infractions.get("collisions_vehicle"),
            "layout": infractions.get("collisions_layout"),
        },
        "cbv_metrics": {
            "cbv_count": meta.get("cbv_count"),
            "cbv_collision_count": meta.get("cbv_collision_count"),
            "cbv_reach_goal_count": meta.get("cbv_reach_goal_count"),
            "cbv_progress": meta.get("cbv_progress"),
            "cbv_off_road_game_time": meta.get("cbv_off_road_game_time"),
            "cbv_uncomfortable_game_time": meta.get("cbv_uncomfortable_game_time"),
        },
    }


def main():
    args = parse_args()
    payloads = load_payloads(args.dump_dir)
    summary = summarize(payloads)
    rift_statistics_path = resolve_statistics_json(args)
    if rift_statistics_path is not None:
        summary["rift_statistics"] = summarize_rift_statistics(rift_statistics_path)
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
