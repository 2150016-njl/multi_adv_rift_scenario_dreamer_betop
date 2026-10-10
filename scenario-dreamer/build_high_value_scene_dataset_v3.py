"""Build a v3 dataset with map-topology metadata while retaining v2 untouched."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import build_high_value_scene_dataset as base
import build_high_value_scene_dataset_v2 as v2
from high_value_init.rift_dataset import write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["inpainting"], default="inpainting")
    parser.add_argument("--min_rift_positive_agents", type=int, default=2)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--positive_target_policy", choices=["route_s_diversity", "top_score"], default="route_s_diversity")
    parser.add_argument("--sparse_context_copies", type=int, default=4)
    parser.add_argument("--sparse_context_min_observed", type=int, default=0)
    parser.add_argument("--sparse_context_max_observed", type=int, default=4)
    parser.add_argument("--sparse_context_replace_base", action="store_true", default=False)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="real_positive_v3")
    parser.add_argument("--quality_filter_v2", action="store_true", default=False)
    # A stopped/creeping RIFT-positive vehicle can still be a valuable interaction.
    parser.add_argument("--min_target_speed", type=float, default=0.0)
    parser.add_argument("--max_target_speed", type=float, default=25.0)
    parser.add_argument("--min_target_ego_distance", type=float, default=3.0)
    parser.add_argument("--max_target_ego_distance", type=float, default=60.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--min_target_pair_distance", type=float, default=2.0)
    parser.add_argument("--max_observed_agents", type=int, default=-1)
    parser.add_argument("--dedup_grid_xy", type=float, default=1.0)
    parser.add_argument("--base_training_weight", type=float, default=1.0)
    parser.add_argument("--sparse_context_weight_bonus", type=float, default=0.25)
    parser.add_argument("--low_observed_weight_bonus", type=float, default=0.25)
    parser.add_argument("--high_speed_weight_bonus", type=float, default=0.15)
    parser.add_argument("--weight_key", default="training_weight")
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--map_source", choices=["snapshot", "hd_map", "hd_map_fallback"], default="snapshot",
                        help="Use snapshot lanes, RIFT HD-map enrichment, or enrichment only when snapshot lanes are empty.")
    parser.add_argument("--rift_root", type=Path, default=None,
                        help="RIFT repository root; required for --map_source hd_map or hd_map_fallback.")
    parser.add_argument("--rift_hd_map_dir", type=Path, default=None,
                        help="Directory containing Town*_HD_map.npz; required for HD-map enrichment.")
    parser.add_argument("--max_input_files", type=int, default=-1,
                        help="Process at most this many snapshots; useful for a preflight audit. -1 processes all.")
    return parser.parse_args()


def point_local(ego, point):
    return base.point_local(ego, point)


def point_segment_distance(point, a, b):
    ax, ay = a[:2]
    bx, by = b[:2]
    px, py = point[:2]
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay), math.atan2(dy, dx) if denom else 0.0
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy)), math.atan2(dy, dx)


def local_lanes(scene, fov):
    ego = scene["ego"]
    half = float(fov) / 2.0
    rows = []
    for lane in scene.get("map_lanes", []):
        points = [point_local(ego, point) for point in lane.get("centerline", []) if point is not None]
        if len(points) < 2 or not any(abs(p[0]) <= half and abs(p[1]) <= half for p in points):
            continue
        rows.append({
            "centerline": points,
            "layer": lane.get("layer", "lane"),
            "road_id": lane.get("road_id"),
            "lane_id": lane.get("lane_id"),
            "lane_type": lane.get("lane_type", ""),
            "topology_type": lane.get("topology_type", ""),
            "topology": lane.get("topology", []),
        })
    return rows


def associate(state, lanes, default_half_width):
    best = None
    for lane_idx, lane in enumerate(lanes):
        for point_idx in range(len(lane["centerline"]) - 1):
            distance, tangent = point_segment_distance(state, lane["centerline"][point_idx], lane["centerline"][point_idx + 1])
            if best is None or distance < best[0]:
                best = (distance, lane_idx, tangent)
    if best is None:
        return {"lane_index": None, "distance": None, "tangent_heading": None, "lane_half_width": default_half_width}
    return {"lane_index": int(best[1]), "distance": float(best[0]), "tangent_heading": float(best[2]), "lane_half_width": float(default_half_width)}


def topology_bucket(lanes):
    layers = {str(lane.get("layer", "")) for lane in lanes}
    if any("connector" in layer for layer in layers):
        return "connector_or_junction"
    if len(lanes) >= 4:
        return "multi_lane"
    if len(lanes) >= 2:
        return "opposing_or_parallel"
    return "single_lane_or_sparse"


class HDMapLaneExtractor:
    """Read RIFT's right-hand HD map and return CARLA-world lane polylines."""

    def __init__(self, rift_root: Path, map_dir: Path):
        if not rift_root or not map_dir:
            raise ValueError("--rift_root and --rift_hd_map_dir are required for HD-map enrichment")
        sys.path.insert(0, str(rift_root))
        from shapely.geometry import Point
        from nuplan_plugin.maps.maps_datatypes import SemanticMapLayer
        from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
        self.Point = Point
        self.SemanticMapLayer = SemanticMapLayer
        self.CarlaMap = CarlaMap
        self.map_dir = str(map_dir)
        self.maps = {}

    def _map(self, town):
        town = str(town or "")
        if not town:
            raise ValueError("snapshot has no town")
        if town not in self.maps:
            self.maps[town] = self.CarlaMap(carla_town_name=town, map_data_dir=self.map_dir)
        return self.maps[town]

    @staticmethod
    def _points(obj):
        raw = getattr(obj, "centerline", None)
        if raw is None:
            raw = getattr(obj, "center_coords", None)
        points = []
        for point in raw if raw is not None else []:
            if hasattr(point, "x") and hasattr(point, "y"):
                x, y, heading = float(point.x), float(point.y), float(getattr(point, "heading", 0.0))
            elif len(point) >= 2:
                x, y = float(point[0]), float(point[1])
                heading = float(point[2]) if len(point) >= 3 else 0.0
            else:
                continue
            # HD-map coordinates are right-handed [x, -carla_y]. Convert
            # them back before the existing ego-local CARLA transform.
            points.append([x, -y, -heading])
        return points

    def lanes(self, scene, radius):
        ego = scene["ego"]
        map_api = self._map(scene.get("town"))
        query = self.Point(float(ego["x"]), -float(ego["y"]))
        proximal = map_api.query_proximal_map_data(query, float(radius))
        lanes = []
        for layer, layer_name in (
                (self.SemanticMapLayer.LANE, "lane"),
                (self.SemanticMapLayer.LANE_CONNECTOR, "lane_connector")):
            for obj in proximal.get(layer, []) or []:
                points = self._points(obj)
                if len(points) < 2:
                    continue
                lanes.append({
                    "centerline": points,
                    "layer": layer_name,
                    "road_id": int(getattr(obj, "road_id", -1)),
                    "lane_id": int(getattr(obj, "lane_id", 0)),
                    "lane_type": str(getattr(obj, "lane_type", "")),
                    "topology_type": str(getattr(obj, "topology_type", "")),
                    "topology": [],
                })
        return lanes


def main():
    args = parse_args()
    if args.num_generate_agents != 2:
        raise ValueError("v3 preserves the validated two-target high-value task")
    v2.validate_args(args)
    extractor = None
    if args.map_source != "snapshot":
        extractor = HDMapLaneExtractor(args.rift_root, args.rift_hd_map_dir)
    raw_records, skipped, lane_rejected = [], 0, 0
    town_counts, bucket_counts, map_source_counts, map_errors = Counter(), Counter(), Counter(), Counter()
    processed_files = 0
    for path in sorted(args.input_dir.rglob("*.json")):
        if args.max_input_files > 0 and processed_files >= args.max_input_files:
            break
        processed_files += 1
        scene = base.load_snapshot(path)
        snapshot_lanes = scene.get("map_lanes", []) or []
        use_hd_map = extractor is not None and (args.map_source == "hd_map" or not snapshot_lanes)
        if use_hd_map:
            try:
                scene["map_lanes"] = extractor.lanes(scene, args.fov)
                map_source_counts["hd_map"] += 1
            except Exception as exc:
                map_errors[type(exc).__name__] += 1
                scene["map_lanes"] = []
        else:
            map_source_counts["snapshot"] += 1
        lanes = local_lanes(scene, args.fov)
        records = base.build_records(scene, args)
        if not records:
            skipped += 1
            continue
        for record in records:
            record = json.loads(json.dumps(record))
            record["schema_version"] = "hv_init_inpainting_v3"
            record["condition"]["map"]["lanes_v3"] = lanes
            associations = [associate(state, lanes, args.default_lane_half_width) for state in record["target"]["generated_agent_states"]]
            if any(item["distance"] is None or item["distance"] > args.max_target_lane_distance for item in associations):
                lane_rejected += 1
                continue
            record["target"]["map_associations_v3"] = associations
            record["topology_v3"] = {"bucket": topology_bucket(lanes), "lane_count": len(lanes)}
            raw_records.append(record)
            town_counts[str(record.get("town", "unknown"))] += 1
            bucket_counts[record["topology_v3"]["bucket"]] += 1
    records, reject_counter, rejected, duplicates = v2.filter_and_weight(raw_records, args)
    train, val, test = base.split_records(records, args.val_ratio, args.test_ratio, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(str(args.output_dir / f"{name}.jsonl"), rows)
    summary = {
        "schema_version": "hv_scene_dataset_v3",
        "num_raw_records": len(raw_records), "num_skipped": skipped, "num_lane_rejected": lane_rejected,
        "num_quality_rejected": rejected, "num_duplicates": duplicates,
        "num_train_records": len(train), "num_val_records": len(val), "num_test_records": len(test),
        "town_counts_before_split": dict(town_counts), "topology_bucket_counts_before_split": dict(bucket_counts),
        "quality_reject_reasons": dict(reject_counter), "args": v2.json_safe(vars(args)),
        "processed_files": processed_files, "map_source_counts": dict(map_source_counts),
        "hd_map_errors": dict(map_errors),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
