#!/usr/bin/env python
"""Build high-value scene-generation datasets from RIFT-labeled snapshots."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from high_value_init.rift_dataset import state_generation_schema_is_clean, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["inpainting", "full_generation"], required=True)
    parser.add_argument("--min_rift_positive_agents", type=int, default=2)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--fov", type=float, default=64.0,
                        help="Scenario Dreamer local square FOV in meters; agents outside +/-fov/2 are not representable.")
    parser.add_argument("--positive_target_policy", default="route_s_diversity",
                        choices=["route_s_diversity", "top_score"])
    parser.add_argument("--sparse_context_copies", type=int, default=0,
                        help="For inpainting records, add this many extra low-observed-context variants per scene.")
    parser.add_argument("--sparse_context_min_observed", type=int, default=0,
                        help="Minimum observed non-target agents kept in sparse context variants.")
    parser.add_argument("--sparse_context_max_observed", type=int, default=2,
                        help="Maximum observed non-target agents kept in sparse context variants.")
    parser.add_argument("--sparse_context_replace_base", action="store_true", default=False,
                        help="Write only sparse context variants instead of also keeping the strict inpainting record.")
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default="real_positive")
    return parser.parse_args()


def load_snapshot(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "scene" in payload and isinstance(payload["scene"], dict):
        return payload["scene"]
    return payload


def local_xy(ego: Dict, x: float, y: float) -> Tuple[float, float]:
    dx = float(x) - float(ego["x"])
    dy = float(y) - float(ego["y"])
    yaw = math.radians(float(ego.get("yaw_deg", 0.0)))
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    return cos_yaw * dx - sin_yaw * dy, sin_yaw * dx + cos_yaw * dy


def local_heading(ego: Dict, yaw_deg: float) -> float:
    return math.radians(float(yaw_deg) - float(ego.get("yaw_deg", 0.0)))


def state_from_actor(ego: Dict, actor: Dict) -> List[float]:
    x, y = local_xy(ego, actor["x"], actor["y"])
    return [
        float(x),
        float(y),
        float(local_heading(ego, actor.get("yaw_deg", 0.0))),
        float(actor.get("speed_mps", 0.0)),
        float(actor.get("length", 4.5)),
        float(actor.get("width", 2.0)),
        1.0,
    ]


def point_local(ego: Dict, point) -> List[float]:
    if isinstance(point, dict):
        return list(local_xy(ego, point["x"], point["y"]))
    return list(local_xy(ego, point[0], point[1]))


def normalize_snapshot(scene: Dict, actor_radius: float, fov: float) -> Dict:
    ego = scene["ego"]
    actors = scene.get("agents", scene.get("actors", []))
    labels = scene.get("rift_labels", scene.get("interaction_labels", []))
    labels_by_id = {int(label["actor_id"]): dict(label) for label in labels if "actor_id" in label}
    half_fov = float(fov) / 2.0

    agent_rows = []
    for local_id, actor in enumerate(actors):
        distance = float(actor.get("ego_distance", math.hypot(actor["x"] - ego["x"], actor["y"] - ego["y"])))
        if distance > actor_radius:
            continue
        state = state_from_actor(ego, actor)
        if abs(state[0]) > half_fov or abs(state[1]) > half_fov:
            continue
        label = labels_by_id.get(int(actor["actor_id"]), {})
        agent_rows.append({
            "local_id": int(actor.get("local_id", local_id)),
            "actor_id": int(actor["actor_id"]),
            "ego_distance": distance,
            "state": state,
            "label": label,
        })
    agent_rows.sort(key=lambda item: (item["ego_distance"], item["actor_id"]))
    for new_id, row in enumerate(agent_rows):
        row["local_id"] = new_id

    route_polyline = [point_local(ego, point) for point in scene.get("ego_route", []) if point is not None]
    if len(route_polyline) < 2:
        route_polyline = [[0.0, 0.0], [1.0, 0.0]]

    lane_polylines = []
    for lane in scene.get("map_lanes", []):
        points = lane.get("centerline", [])
        local_points = [point_local(ego, point) for point in points if point is not None]
        if len(local_points) >= 2:
            lane_polylines.append(local_points)

    route_id = scene.get("route_id")
    scenario = scene.get("scenario") or {}
    if route_id is None and isinstance(scenario, dict):
        route_id = scenario.get("name")
    scene_id = scene.get("scene_id") or str(route_id)
    positive = [
        row for row in agent_rows
        if bool(row["label"].get("is_interactive", False))
    ]
    return {
        "scene_id": scene_id,
        "episode_key": episode_key(scene),
        "town": scene.get("town") or scenario.get("town"),
        "ego_state": [
            0.0,
            0.0,
            0.0,
            float(ego.get("speed_mps", 0.0)),
            float(ego.get("length", 5.176)),
            float(ego.get("width", 2.297)),
            1.0,
        ],
        "ego_route": {"polyline": route_polyline},
        "map": {"lane_polylines": lane_polylines},
        "agents": agent_rows,
        "positive_agents": positive,
    }


def episode_key(scene: Dict) -> str:
    if isinstance(scene.get("scenario"), dict):
        scenario = scene["scenario"]
        return f"{scenario.get('town')}:{scenario.get('name')}:{scenario.get('index')}"
    return f"{scene.get('town')}:{scene.get('route_id')}:{scene.get('route_index')}:{scene.get('repetition_index')}"


def choose_targets(positive: List[Dict], count: int, policy: str) -> List[Dict]:
    if len(positive) < count:
        return []
    if policy == "top_score":
        return sorted(
            positive,
            key=lambda row: (float(row["label"].get("score", 0.0)), -float(row.get("ego_distance", 0.0))),
            reverse=True,
        )[:count]
    if count == 1:
        return [max(positive, key=lambda row: float(row["label"].get("score", 0.0)))]
    best = None
    best_key = None
    for i in range(len(positive)):
        for j in range(i + 1, len(positive)):
            a, b = positive[i], positive[j]
            da = a["label"].get("ego_distance", a.get("ego_distance", 0.0))
            db = b["label"].get("ego_distance", b.get("ego_distance", 0.0))
            score = float(a["label"].get("score", 0.0)) + float(b["label"].get("score", 0.0))
            key = (abs(float(da or 0.0) - float(db or 0.0)), score)
            if best_key is None or key > best_key:
                best_key = key
                best = [a, b]
    return best or []


def build_records(scene: Dict, args) -> List[Dict]:
    record = build_record(scene, args)
    if not record:
        return []
    if args.mode != "inpainting" or int(args.sparse_context_copies) <= 0:
        return [record]

    records = [] if args.sparse_context_replace_base else [record]
    for copy_index in range(int(args.sparse_context_copies)):
        records.append(build_sparse_context_record(record, args, copy_index))
    return records


def build_record(scene: Dict, args) -> Dict:
    norm = normalize_snapshot(scene, actor_radius=float(args.actor_radius), fov=float(args.fov))
    positive = norm["positive_agents"]
    if len(positive) < args.min_rift_positive_agents:
        return {}
    if args.mode == "inpainting":
        targets = choose_targets(positive, args.num_generate_agents, args.positive_target_policy)
        if len(targets) != args.num_generate_agents:
            return {}
        target_ids = {row["actor_id"] for row in targets}
        observed = [row for row in norm["agents"] if row["actor_id"] not in target_ids]
        record = {
            "schema_version": "hv_init_inpainting_v1",
            "scene_id": norm["scene_id"],
            "episode_key": norm["episode_key"],
            "town": norm["town"],
            "condition": {
                "ego_state": norm["ego_state"],
                "ego_route": norm["ego_route"],
                "map": norm["map"],
                "observed_agent_states": [row["state"] for row in observed],
                "observed_agent_types": [1 for _ in observed],
                "num_generate_agents": args.num_generate_agents,
            },
            "target": {
                "generated_agent_states": [row["state"] for row in targets],
                "generated_agent_types": [1 for _ in targets],
            },
            "rift_audit": {
                "num_rift_positive_agents_original": len(positive),
                "removed_positive_local_ids": [int(row["local_id"]) for row in targets],
                "observed_agent_count_before_context_sampling": len(observed),
                "observed_agent_count_after_context_sampling": len(observed),
            },
            "context_sampling": {
                "policy": "strict_inpainting",
                "sparse_context_copy_index": None,
            },
            "source": args.source,
        }
    else:
        agents = select_full_generation_agents(norm["agents"], positive, args.num_generate_agents)
        if len(agents) != args.num_generate_agents:
            return {}
        record = {
            "schema_version": "hv_init_full_generation_v1",
            "scene_id": norm["scene_id"],
            "episode_key": norm["episode_key"],
            "town": norm["town"],
            "condition": {
                "ego_state": norm["ego_state"],
                "ego_route": norm["ego_route"],
                "map": norm["map"],
                "num_generate_agents": args.num_generate_agents,
            },
            "target": {
                "agent_states": [row["state"] for row in agents],
                "agent_types": [1 for _ in agents],
            },
            "rift_audit": {
                "num_rift_positive_agents_original": len(positive),
                "positive_local_ids": [int(row["local_id"]) for row in positive],
            },
            "source": args.source,
        }
    if not state_generation_schema_is_clean(record):
        raise ValueError(f"forbidden selector/delta field leaked into {record['scene_id']}")
    return record


def build_sparse_context_record(record: Dict, args, copy_index: int) -> Dict:
    sparse = json.loads(json.dumps(record))
    observed = list((record.get("condition") or {}).get("observed_agent_states") or [])
    observed_types = list((record.get("condition") or {}).get("observed_agent_types") or [])
    max_count = max(0, int(args.sparse_context_max_observed))
    min_count = max(0, int(args.sparse_context_min_observed))
    if min_count > max_count:
        raise ValueError("--sparse_context_min_observed must be <= --sparse_context_max_observed")
    upper = min(max_count, len(observed))
    lower = min(min_count, upper)
    rng = random.Random(stable_int_seed(
        str(record.get("scene_id")),
        str(record.get("episode_key")),
        str(copy_index),
        str(args.seed),
    ))
    keep_count = rng.randint(lower, upper) if upper >= lower else 0
    indices = sorted(rng.sample(range(len(observed)), keep_count)) if keep_count > 0 else []

    sparse["condition"]["observed_agent_states"] = [observed[index] for index in indices]
    if observed_types:
        sparse["condition"]["observed_agent_types"] = [observed_types[index] for index in indices]
    sparse["scene_id"] = f"{record.get('scene_id')}:sparse{copy_index}"
    sparse["source"] = f"{record.get('source', args.source)}:sparse_context"
    sparse["context_sampling"] = {
        "policy": "sparse_random_observed",
        "sparse_context_copy_index": int(copy_index),
        "min_observed": int(min_count),
        "max_observed": int(max_count),
        "kept_observed_indices": [int(index) for index in indices],
    }
    sparse.setdefault("rift_audit", {})
    sparse["rift_audit"]["observed_agent_count_before_context_sampling"] = len(observed)
    sparse["rift_audit"]["observed_agent_count_after_context_sampling"] = keep_count
    if not state_generation_schema_is_clean(sparse):
        raise ValueError(f"forbidden selector/delta field leaked into {sparse['scene_id']}")
    return sparse


def stable_int_seed(*parts: str) -> int:
    text = "|".join(parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def validate_args(args) -> None:
    if int(args.sparse_context_copies) < 0:
        raise ValueError("--sparse_context_copies must be >= 0")
    if int(args.sparse_context_min_observed) < 0:
        raise ValueError("--sparse_context_min_observed must be >= 0")
    if int(args.sparse_context_max_observed) < 0:
        raise ValueError("--sparse_context_max_observed must be >= 0")
    if int(args.sparse_context_min_observed) > int(args.sparse_context_max_observed):
        raise ValueError("--sparse_context_min_observed must be <= --sparse_context_max_observed")


def select_full_generation_agents(agents: List[Dict], positive: List[Dict], count: int) -> List[Dict]:
    positive_ids = {row["actor_id"] for row in positive}
    keep = list(positive)
    keep_ids = set(positive_ids)
    for row in agents:
        if len(keep) >= count:
            break
        if row["actor_id"] in keep_ids:
            continue
        keep.append(row)
        keep_ids.add(row["actor_id"])
    if len(keep) < count:
        return []
    return sorted(keep[:count], key=lambda row: (row["ego_distance"], row["actor_id"]))


def split_records(records: List[Dict], val_ratio: float, test_ratio: float, seed: int):
    episode_keys = sorted({record["episode_key"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(episode_keys)
    n = len(episode_keys)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_keys = set(episode_keys[:n_test])
    val_keys = set(episode_keys[n_test:n_test + n_val])
    train, val, test = [], [], []
    for idx, record in enumerate(records):
        record = dict(record)
        record["idx"] = idx
        if record["episode_key"] in test_keys:
            record["split"] = "test"
            test.append(record)
        elif record["episode_key"] in val_keys:
            record["split"] = "val"
            val.append(record)
        else:
            record["split"] = "train"
            train.append(record)
    return train, val, test


def main():
    args = parse_args()
    validate_args(args)
    paths = sorted(args.input_dir.rglob("*.json"))
    records = []
    skipped = 0
    for path in paths:
        scene = load_snapshot(path)
        scene_records = build_records(scene, args)
        if scene_records:
            source_file_sha1 = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
            for record in scene_records:
                record["source_file_sha1"] = source_file_sha1
                records.append(record)
        else:
            skipped += 1

    summary = {
        "mode": args.mode,
        "num_input_json": len(paths),
        "num_records": len(records),
        "num_skipped": skipped,
        "min_rift_positive_agents": args.min_rift_positive_agents,
        "num_generate_agents": args.num_generate_agents,
        "fov": args.fov,
        "sparse_context_copies": int(args.sparse_context_copies),
        "sparse_context_min_observed": int(args.sparse_context_min_observed),
        "sparse_context_max_observed": int(args.sparse_context_max_observed),
        "sparse_context_replace_base": bool(args.sparse_context_replace_base),
    }
    if args.output_dir:
        train, val, test = split_records(records, args.val_ratio, args.test_ratio, args.seed)
        write_jsonl(str(args.output_dir / "train.jsonl"), train)
        write_jsonl(str(args.output_dir / "val.jsonl"), val)
        if args.test_ratio > 0:
            write_jsonl(str(args.output_dir / "test.jsonl"), test)
        summary.update({
            "output_dir": str(args.output_dir),
            "num_train_records": len(train),
            "num_val_records": len(val),
            "num_test_records": len(test),
        })
    if args.output_jsonl:
        write_jsonl(str(args.output_jsonl), records)
        summary["output_jsonl"] = str(args.output_jsonl)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
