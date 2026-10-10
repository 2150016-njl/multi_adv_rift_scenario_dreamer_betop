#!/usr/bin/env python
"""Summarize RIFT-labeled high-value scene snapshots."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--min_positive_agents", type=int, default=2)
    parser.add_argument("--min_forward_positive_agents", type=int, default=2)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    return parser.parse_args()


def percentile(values: List[float], q: float):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def numeric(values: List[float]) -> Dict:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"min": None, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def route_length(route: List) -> float:
    points = []
    for point in route or []:
        if isinstance(point, dict):
            points.append((float(point["x"]), float(point["y"])))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(points[idx][0] - points[idx - 1][0], points[idx][1] - points[idx - 1][1])
        for idx in range(1, len(points))
    )


def load_scene(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("scene") if isinstance(payload.get("scene"), dict) else payload


def is_rift_labeled_snapshot(scene: Dict) -> bool:
    if not isinstance(scene, dict) or not isinstance(scene.get("ego"), dict):
        return False
    actors = scene.get("agents", scene.get("actors"))
    return isinstance(actors, list) and isinstance(scene.get("rift_labels"), list)


def local_xy(ego: Dict, actor: Dict):
    dx = float(actor["x"]) - float(ego["x"])
    dy = float(actor["y"]) - float(ego["y"])
    yaw = math.radians(float(ego.get("yaw_deg", 0.0)))
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def forward_positive_count_and_regions(scene: Dict, args):
    actor_by_id = {
        int(actor["actor_id"]): actor
        for actor in (scene.get("agents") or scene.get("actors") or [])
        if actor.get("actor_id") is not None
    }
    ego = scene.get("ego") or {}
    half_fov = args.fov / 2.0
    forward_limit = half_fov - args.target_fov_margin
    regions = Counter()
    forward = 0
    for label in scene.get("rift_labels") or []:
        if not label.get("is_interactive", False):
            continue
        try:
            actor_id = int(label.get("actor_id"))
        except (TypeError, ValueError):
            actor_id = None
        actor = actor_by_id.get(actor_id)
        if actor is None:
            regions["unmatched_label_actor"] += 1
            continue
        try:
            x, y = local_xy(ego, actor)
        except (KeyError, TypeError, ValueError):
            regions["invalid_actor_or_ego_state"] += 1
            continue
        if abs(x) > half_fov or abs(y) > half_fov:
            regions["outside_candidate_fov"] += 1
        elif args.forward_min_x <= x <= forward_limit and abs(y) <= forward_limit:
            regions["forward_target_sector"] += 1
            forward += 1
        elif x < 0.0:
            regions["rear_half"] += 1
        else:
            regions["forward_or_side_excluded_by_margin_or_min_x"] += 1
    return forward, regions


def main():
    args = parse_args()
    paths = sorted(args.dump_dir.rglob("*.json"))
    positive_counts = []
    actor_counts = []
    route_lengths = []
    town_counts = Counter()
    schema_counts = Counter()
    forward_positive_counts = []
    positive_region_counts = Counter()
    examples_positive = []
    examples_no_labels = []
    invalid_input_examples = []
    valid_labeled_snapshot_count = 0

    for path in paths:
        scene = load_scene(path)
        if not is_rift_labeled_snapshot(scene):
            if len(invalid_input_examples) < 20:
                invalid_input_examples.append({
                    "path": str(path),
                    "schema_version": scene.get("schema_version") if isinstance(scene, dict) else None,
                    "top_level_keys": sorted(scene.keys())[:20] if isinstance(scene, dict) else [],
                })
            continue
        valid_labeled_snapshot_count += 1
        schema_counts[str(scene.get("schema_version", "unknown"))] += 1
        labels = scene.get("rift_labels") or []
        positives = [label for label in labels if label.get("is_interactive", False)]
        forward_positive, regions = forward_positive_count_and_regions(scene, args)
        positive_counts.append(len(positives))
        forward_positive_counts.append(forward_positive)
        positive_region_counts.update(regions)
        actor_counts.append(len(scene.get("agents") or scene.get("actors") or []))
        route_lengths.append(route_length(scene.get("ego_route") or []))
        if scene.get("town"):
            town_counts[str(scene["town"])] += 1
        if len(positives) >= args.min_positive_agents and len(examples_positive) < 20:
            examples_positive.append({
                "path": str(path),
                "scene_id": scene.get("scene_id"),
                "num_positive_agents": len(positives),
            })
        if not labels and len(examples_no_labels) < 20:
            examples_no_labels.append({
                "path": str(path),
                "scene_id": scene.get("scene_id"),
            })

    num_positive_scenes = sum(1 for count in positive_counts if count >= args.min_positive_agents)
    num_forward_positive_scenes = sum(
        1 for count in forward_positive_counts if count >= args.min_forward_positive_agents
    )
    summary = {
        "schema_version": "hv_init_rift_labeled_snapshot_summary_v2_forward_coverage",
        "dump_dir": str(args.dump_dir),
        "num_json": len(paths),
        "valid_rift_labeled_snapshot_count": valid_labeled_snapshot_count,
        "invalid_input_examples": invalid_input_examples,
        "min_positive_agents": args.min_positive_agents,
        "min_forward_positive_agents": args.min_forward_positive_agents,
        "candidate_fov": args.fov,
        "forward_min_x": args.forward_min_x,
        "target_fov_margin": args.target_fov_margin,
        "num_positive_scenes": num_positive_scenes,
        "positive_scene_ratio": num_positive_scenes / len(paths) if paths else 0.0,
        "num_forward_positive_scenes": num_forward_positive_scenes,
        "forward_positive_scene_ratio": num_forward_positive_scenes / len(paths) if paths else 0.0,
        "positive_agents_per_scene": numeric(positive_counts),
        "forward_positive_agents_per_scene": numeric(forward_positive_counts),
        "actor_count": numeric(actor_counts),
        "route_length": numeric(route_lengths),
        "positive_count_hist": dict(sorted(Counter(positive_counts).items())),
        "forward_positive_count_hist": dict(sorted(Counter(forward_positive_counts).items())),
        "rift_positive_region_counts": dict(sorted(positive_region_counts.items())),
        "schema_counts": dict(schema_counts),
        "town_counts": dict(town_counts),
        "examples_positive": examples_positive,
        "examples_missing_labels": examples_no_labels,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if valid_labeled_snapshot_count == 0:
        raise SystemExit(
            "no RIFT-labeled snapshot JSON found: verify --dump_dir/RAW_SNAPSHOTS; "
            "the repository root is not a valid snapshot dataset"
        )


if __name__ == "__main__":
    main()
