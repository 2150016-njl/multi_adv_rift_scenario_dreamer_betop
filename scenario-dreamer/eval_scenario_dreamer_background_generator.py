"""Offline feasibility audit for the ordinary-BV stage of the composed generator."""

import argparse
import json
from pathlib import Path

import numpy as np

from high_value_init.high_value_ldm import RouteConditionedHighValueLDM
from high_value_init.rift_dataset import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--background_checkpoint", default=None)
    parser.add_argument("--num_generate_agents", type=int, default=1)
    parser.add_argument("--num_samples_per_condition", type=int, default=4)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--fov_margin", type=float, default=2.0)
    parser.add_argument("--max_lane_distance", type=float, default=5.0)
    parser.add_argument("--min_speed", type=float, default=0.0)
    parser.add_argument("--max_speed", type=float, default=30.0)
    parser.add_argument("--min_length", type=float, default=2.0)
    parser.add_argument("--max_length", type=float, default=8.0)
    parser.add_argument("--min_width", type=float, default=1.4)
    parser.add_argument("--max_width", type=float, default=3.2)
    parser.add_argument("--min_clearance", type=float, default=2.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_generated_jsonl", type=Path, required=True)
    return parser.parse_args()


def composed_background_condition(record, count):
    record = dict(record)
    condition = dict(record.get("condition") or {})
    condition["observed_agent_states"] = []
    condition["observed_agent_types"] = []
    condition["num_generate_agents"] = int(count)
    record["condition"] = condition
    return record


def point_to_segment_distance(point, start, end):
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def nearest_lane_distance(state, record):
    lanes = ((record.get("condition") or {}).get("map") or {}).get("lanes_v3") or []
    point = np.asarray(state[:2], dtype=np.float64)
    best = None
    for lane in lanes:
        centerline = np.asarray((lane or {}).get("centerline") or [], dtype=np.float64)
        if centerline.ndim != 2 or centerline.shape[0] < 2:
            continue
        for start, end in zip(centerline[:-1, :2], centerline[1:, :2]):
            distance = point_to_segment_distance(point, start, end)
            best = distance if best is None else min(best, distance)
    return best


def numeric(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return {"min": None, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "min": float(np.min(values)), "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)), "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main():
    args = parse_args()
    if args.num_generate_agents <= 0 or args.num_samples_per_condition <= 0:
        raise ValueError("generation counts must be positive")
    records = read_jsonl(args.jsonl)
    if args.max_records > 0:
        records = records[:args.max_records]
    if not records:
        raise ValueError("no input records")
    model = RouteConditionedHighValueLDM(
        checkpoint_path=args.background_checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode="inpainting",
        num_generate_agents=args.num_generate_agents,
        use_route_condition=bool(args.background_checkpoint),
        clamp_generated_states=False,
    )
    rows, lane_distances, clearances = [], [], []
    counts = {key: 0 for key in ("agents", "fov_invalid", "speed_invalid", "size_invalid", "lane_missing", "lane_invalid", "ego_clearance_invalid", "pair_clearance_invalid", "fully_feasible_scenes")}
    for record_index, record in enumerate(records):
        condition = composed_background_condition(record, args.num_generate_agents)
        for sample_index in range(args.num_samples_per_condition):
            generated = model.generate(condition, args.num_generate_agents)
            states = generated["generated_agent_states"]
            scene_feasible = True
            xy = [np.asarray(state[:2], dtype=np.float64) for state in states]
            pair_min = min((float(np.linalg.norm(left - right)) for index, left in enumerate(xy) for right in xy[index + 1:]), default=None)
            for agent_index, state in enumerate(states):
                x, y, _, speed, length, width, _ = [float(value) for value in state]
                lane_distance = nearest_lane_distance(state, record)
                ego_clearance = float(np.linalg.norm(np.asarray([x, y], dtype=np.float64)))
                fov_valid = abs(x) <= args.fov / 2.0 - args.fov_margin and abs(y) <= args.fov / 2.0 - args.fov_margin
                speed_valid = args.min_speed <= speed <= args.max_speed
                size_valid = args.min_length <= length <= args.max_length and args.min_width <= width <= args.max_width
                lane_valid = lane_distance is not None and lane_distance <= args.max_lane_distance
                ego_clearance_valid = ego_clearance >= args.min_clearance
                pair_clearance_valid = pair_min is None or pair_min >= args.min_clearance
                counts["agents"] += 1
                counts["fov_invalid"] += int(not fov_valid)
                counts["speed_invalid"] += int(not speed_valid)
                counts["size_invalid"] += int(not size_valid)
                counts["lane_missing"] += int(lane_distance is None)
                counts["lane_invalid"] += int(not lane_valid)
                counts["ego_clearance_invalid"] += int(not ego_clearance_valid)
                counts["pair_clearance_invalid"] += int(not pair_clearance_valid)
                lane_distances.append(lane_distance)
                clearances.append(ego_clearance)
                scene_feasible &= fov_valid and speed_valid and size_valid and lane_valid and ego_clearance_valid and pair_clearance_valid
                rows.append({
                    "schema_version": "scenario_dreamer_background_audit_v1",
                    "scene_id": record.get("scene_id"), "episode_key": record.get("episode_key"),
                    "town": record.get("town"), "record_index": record_index,
                    "sample_index": sample_index, "agent_index": agent_index,
                    "generator_source": "scenario_dreamer_background", "generated_agent_state": state,
                    "nearest_lane_distance_m": lane_distance, "ego_distance_m": ego_clearance,
                    "min_pair_distance_m": pair_min, "fov_valid": fov_valid, "speed_valid": speed_valid,
                    "size_valid": size_valid, "lane_valid": lane_valid,
                    "ego_clearance_valid": ego_clearance_valid, "pair_clearance_valid": pair_clearance_valid,
                })
            counts["fully_feasible_scenes"] += int(scene_feasible)
    scene_count = len(records) * args.num_samples_per_condition
    agent_count = counts["agents"]
    summary = {
        "schema_version": "scenario_dreamer_background_audit_summary_v1",
        "background_checkpoint": args.background_checkpoint or args.ldm_checkpoint,
        "uses_generic_ldm_as_background": args.background_checkpoint is None,
        "num_records": len(records), "num_samples_per_condition": args.num_samples_per_condition,
        "num_generate_agents": args.num_generate_agents, "num_scenes": scene_count,
        "num_generated_agents": agent_count, "counts": counts,
        "rates": {key.replace("_invalid", "_invalid_rate"): value / agent_count for key, value in counts.items() if key.endswith("_invalid")},
        "fully_feasible_scene_rate": counts["fully_feasible_scenes"] / scene_count,
        "nearest_lane_distance_m": numeric(lane_distances), "ego_distance_m": numeric(clearances),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_generated_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(str(args.output_generated_jsonl), rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
