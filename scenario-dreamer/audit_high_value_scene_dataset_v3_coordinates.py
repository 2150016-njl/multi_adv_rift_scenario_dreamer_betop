"""Audit raw snapshot lane coordinates before building the V3 dataset."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import build_high_value_scene_dataset as base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--max_files", type=int, default=2000)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return math.hypot(point[0] - (start[0] + ratio * dx), point[1] - (start[1] + ratio * dy))


def local_lanes(scene, fov, mirror_map_y):
    ego = scene["ego"]
    half = float(fov) / 2.0
    lanes = []
    for lane in scene.get("map_lanes", []):
        points = []
        for point in lane.get("centerline", []):
            if point is None or len(point) < 2:
                continue
            world_point = [point[0], -point[1] if mirror_map_y else point[1]]
            points.append(base.point_local(ego, world_point))
        if len(points) >= 2 and any(abs(point[0]) <= half and abs(point[1]) <= half for point in points):
            lanes.append(points)
    return lanes


def nearest_lane_distance(point, lanes):
    distances = [
        point_segment_distance(point, lane[index], lane[index + 1])
        for lane in lanes
        for index in range(len(lane) - 1)
    ]
    return min(distances) if distances else None


def numeric(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
    def percentile(q):
        index = int(round((len(values) - 1) * q))
        return values[index]
    return {"count": len(values), "min": values[0], "p50": percentile(0.5), "p90": percentile(0.9), "max": values[-1]}


def main():
    args = parse_args()
    build_args = SimpleNamespace(
        actor_radius=args.actor_radius,
        fov=args.fov,
        min_rift_positive_agents=2,
        num_generate_agents=2,
        positive_target_policy="route_s_diversity",
    )
    raw_distances, mirrored_distances, examples = [], [], []
    scanned = candidates = no_raw_lanes = no_mirrored_lanes = 0
    for path in sorted(args.input_dir.rglob("*.json")):
        if args.max_files > 0 and scanned >= args.max_files:
            break
        scanned += 1
        scene = base.load_snapshot(path)
        norm = base.normalize_snapshot(scene, args.actor_radius, args.fov)
        targets = base.choose_targets(
            norm["positive_agents"], build_args.num_generate_agents, build_args.positive_target_policy
        )
        if len(targets) != 2:
            continue
        candidates += 1
        raw_lanes = local_lanes(scene, args.fov, mirror_map_y=False)
        mirrored_lanes = local_lanes(scene, args.fov, mirror_map_y=True)
        no_raw_lanes += int(not raw_lanes)
        no_mirrored_lanes += int(not mirrored_lanes)
        for target in targets:
            point = target["state"][:2]
            raw_distance = nearest_lane_distance(point, raw_lanes)
            mirrored_distance = nearest_lane_distance(point, mirrored_lanes)
            raw_distances.append(raw_distance)
            mirrored_distances.append(mirrored_distance)
            if len(examples) < 8:
                examples.append({
                    "path": str(path),
                    "target_local_xy": point,
                    "raw_map_lane_distance_m": raw_distance,
                    "mirrored_map_lane_distance_m": mirrored_distance,
                    "ego_world_xy": [scene["ego"]["x"], scene["ego"]["y"]],
                })
    threshold = float(args.max_target_lane_distance)
    report = {
        "schema_version": "hv_v3_coordinate_audit_v1",
        "scanned_files": scanned,
        "candidate_scenes": candidates,
        "candidate_targets": len(raw_distances),
        "raw_map": {
            "distance_m": numeric(raw_distances),
            "within_threshold_rate": sum(value <= threshold for value in raw_distances if value is not None) / len(raw_distances) if raw_distances else None,
            "scenes_without_local_lanes": no_raw_lanes,
        },
        "mirrored_y_map": {
            "distance_m": numeric(mirrored_distances),
            "within_threshold_rate": sum(value <= threshold for value in mirrored_distances if value is not None) / len(mirrored_distances) if mirrored_distances else None,
            "scenes_without_local_lanes": no_mirrored_lanes,
        },
        "examples": examples,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
