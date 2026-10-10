#!/usr/bin/env python
"""V2 offline metrics for high-value scene generators.

The original evaluator is kept intact.  This v2 evaluator adds distributional
metrics that are closer to online viability: percentiles, FOV/boundary rates,
pair separation/overlap, physical invalid components, and route distance.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from high_value_init.evaluation_geometry import geometry_metrics, position_l1, seed_everything, sha256_file
from high_value_init.high_value_ldm import match_state_metrics
from high_value_init.high_value_ldm import RouteConditionedHighValueLDM
from high_value_init.rift_dataset import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inpainting", "full_generation"], default="inpainting")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_generated_jsonl", default=None)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--route_num_points", type=int, default=80)
    parser.add_argument("--num_samples_per_condition", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_generated_speed", type=float, default=0.0)
    parser.add_argument("--max_generated_speed", type=float, default=30.0)
    parser.add_argument("--min_generated_length", type=float, default=2.0)
    parser.add_argument("--max_generated_length", type=float, default=8.0)
    parser.add_argument("--min_generated_width", type=float, default=1.4)
    parser.add_argument("--max_generated_width", type=float, default=3.2)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--boundary_margin", type=float, default=2.0)
    parser.add_argument("--min_pair_distance", type=float, default=2.0)
    parser.add_argument("--clamp_generated_states", action="store_true", default=False)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; each (record, sample) receives a stable derived seed.")
    parser.add_argument("--overwrite_output", action="store_true", default=False)
    return parser.parse_args()


def validate_args(args) -> None:
    if args.min_generated_speed > args.max_generated_speed:
        raise ValueError("--min_generated_speed must be <= --max_generated_speed")
    if args.min_generated_length > args.max_generated_length:
        raise ValueError("--min_generated_length must be <= --max_generated_length")
    if args.min_generated_width > args.max_generated_width:
        raise ValueError("--min_generated_width must be <= --max_generated_width")


def target_states(record: Dict) -> List[List[float]]:
    target = record.get("target") or {}
    return list(target.get("generated_agent_states") or target.get("agent_states") or [])


def route_points(record: Dict) -> np.ndarray:
    route = (record.get("condition") or {}).get("ego_route") or {}
    points = route.get("polyline") if isinstance(route, dict) else route
    xy = []
    for point in points or []:
        if isinstance(point, dict):
            xy.append([float(point["x"]), float(point["y"])])
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            xy.append([float(point[0]), float(point[1])])
    return np.asarray(xy, dtype=np.float32)


def point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def distance_to_route_xy(x: float, y: float, route: np.ndarray) -> float:
    if route.shape[0] == 0:
        return float("nan")
    point = np.asarray([x, y], dtype=np.float32)
    if route.shape[0] == 1:
        return float(np.linalg.norm(point - route[0]))
    return min(point_to_segment_distance(point, route[i], route[i + 1]) for i in range(route.shape[0] - 1))


def pair_distances(states: List[List[float]]) -> List[float]:
    distances = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            distances.append(math.hypot(float(states[i][0]) - float(states[j][0]),
                                        float(states[i][1]) - float(states[j][1])))
    return distances


def state_quality_metrics(states: List[List[float]], record: Dict, args) -> Dict[str, float]:
    route = route_points(record)
    half_fov = float(args.fov) / 2.0
    margins = []
    route_distances = []
    invalid_speed = invalid_size = invalid_fov = near_boundary = 0
    for state in states:
        x, y, _heading, speed, length, width = [float(v) for v in state[:6]]
        margin = half_fov - max(abs(x), abs(y))
        margins.append(margin)
        route_distances.append(distance_to_route_xy(x, y, route))
        invalid_speed += int(speed < args.min_generated_speed or speed > args.max_generated_speed)
        invalid_size += int(
            length < args.min_generated_length or length > args.max_generated_length or
            width < args.min_generated_width or width > args.max_generated_width
        )
        invalid_fov += int(margin < 0.0)
        near_boundary += int(margin <= args.boundary_margin)
    pairs = pair_distances(states)
    pair_min = min(pairs) if pairs else float("nan")
    return {
        "generated_min_fov_margin": min(margins) if margins else float("nan"),
        "generated_near_boundary_count": int(near_boundary),
        "generated_off_fov_count": int(invalid_fov),
        "generated_invalid_speed_count": int(invalid_speed),
        "generated_invalid_size_count": int(invalid_size),
        "generated_min_pair_distance": pair_min,
        "generated_overlap_count": int(0 if math.isnan(pair_min) else pair_min < args.min_pair_distance),
        "generated_route_distance_mean": float(np.nanmean(route_distances)) if route_distances else float("nan"),
        "generated_route_distance_max": float(np.nanmax(route_distances)) if route_distances else float("nan"),
    }


def percentile(values: Iterable[float], q: float):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return None
    return float(np.percentile(np.asarray(values), q))


def numeric(values: Iterable[float]) -> Dict:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def summarize(metrics: List[Dict], prefix: str) -> Dict:
    keys = sorted({key for row in metrics for key in row.keys()})
    out = {}
    for key in keys:
        if key.endswith("_count"):
            values = [row.get(key, 0) for row in metrics]
            out[f"{prefix}_{key}_sum"] = int(np.nansum(values))
            out[f"{prefix}_{key}_rate"] = float(np.nanmean([float(v) > 0 for v in values])) if values else None
        else:
            out[f"{prefix}_{key}"] = numeric(row.get(key) for row in metrics)
    return out


def comparable_summary(metrics: List[Dict]) -> Dict:
    """V3-shaped per-metric summary used for frozen V2/V3 comparisons."""
    keys = sorted({key for row in metrics for key in row})
    return {
        "num_samples": len(metrics),
        **{key: numeric(row.get(key) for row in metrics) for key in keys},
    }


def group_rows(rows: List[Dict], field: str) -> Dict:
    groups = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        groups.setdefault(key, []).append(row["metrics"])
    return {key: comparable_summary(metrics) for key, metrics in sorted(groups.items())}


def main():
    args = parse_args()
    validate_args(args)
    records = read_jsonl(args.jsonl)
    if args.max_records and args.max_records > 0:
        records = records[:args.max_records]
    if not records:
        raise ValueError(f"no records in {args.jsonl}")
    outputs = [Path(args.output_json)]
    if args.output_generated_jsonl:
        outputs.append(Path(args.output_generated_jsonl))
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite_output:
        raise FileExistsError(f"refusing to overwrite frozen evaluation outputs: {existing}; pass --overwrite_output explicitly")

    model = RouteConditionedHighValueLDM(
        checkpoint_path=args.checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode=args.mode,
        num_generate_agents=args.num_generate_agents,
        route_num_points=args.route_num_points,
        min_generated_speed=args.min_generated_speed,
        max_generated_speed=args.max_generated_speed,
        min_generated_length=args.min_generated_length,
        max_generated_length=args.max_generated_length,
        min_generated_width=args.min_generated_width,
        max_generated_width=args.max_generated_width,
        clamp_generated_states=args.clamp_generated_states,
    )

    rows = []
    best_metrics = []
    all_sample_metrics = []
    target_quality = []
    for rec_idx, record in enumerate(records):
        target_quality.append(state_quality_metrics(target_states(record), record, args))
        best = None
        for sample_idx in range(args.num_samples_per_condition):
            sample_seed = seed_everything(args.seed, rec_idx, sample_idx)
            out = model.reconstruct_targets(record)
            generated_states = out["generated"]["generated_agent_states"]
            targets = target_states(record)
            metrics = {
                **match_state_metrics(
                    torch.tensor(generated_states, dtype=torch.float32),
                    torch.tensor(targets, dtype=torch.float32),
                    bounds=model.generated_state_bounds,
                ),
                "position_l1": position_l1(generated_states, targets),
                **state_quality_metrics(generated_states, record, args),
                **geometry_metrics(
                    generated_states, record,
                    lane_half_width=args.default_lane_half_width,
                    min_center_clearance=args.min_initial_clearance,
                    fov=args.fov,
                ),
            }
            row = {
                "scene_id": record.get("scene_id"),
                "record_idx": rec_idx,
                "sample_idx": sample_idx,
                "sample_seed": sample_seed,
                "town": record.get("town"),
                "topology_bucket": (record.get("topology_v3") or record.get("topology_v4") or {}).get("bucket"),
                "generated": out["generated"],
                "metrics": metrics,
            }
            rows.append(row)
            all_sample_metrics.append(metrics)
            if best is None or metrics["position_l2"] < best["metrics"]["position_l2"]:
                best = row
        best_metrics.append(best["metrics"])

    summary = {
        "schema_version": "hv_init_offline_eval_v2",
        "mode": args.mode,
        "num_records": len(records),
        "num_samples_per_condition": args.num_samples_per_condition,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "jsonl_sha256": sha256_file(args.jsonl),
        "base_seed": args.seed,
        "sample_seed_scheme": "(base_seed + 1000003*record_index + 9176*sample_index) mod 2^32",
        "clamp_generated_states": args.clamp_generated_states,
        "raw_state_postprocessing": "clamp_ablation" if args.clamp_generated_states else "none",
        "bounds": {
            "speed": [args.min_generated_speed, args.max_generated_speed],
            "length": [args.min_generated_length, args.max_generated_length],
            "width": [args.min_generated_width, args.max_generated_width],
            "fov": args.fov,
            "boundary_margin": args.boundary_margin,
            "min_pair_distance": args.min_pair_distance,
        },
        "sampled": comparable_summary(all_sample_metrics),
        "by_town": group_rows(rows, "town"),
        "by_topology_bucket": group_rows(rows, "topology_bucket"),
        **summarize(all_sample_metrics, "all_samples"),
        **summarize(target_quality, "target"),
        "note": "Raw samples are never selected for online use. RIFT/CARLA online audit is still required for true interaction quality.",
    }
    if args.num_samples_per_condition > 1:
        summary["best_of_n_position_l2_oracle"] = comparable_summary(best_metrics)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.output_generated_jsonl:
        write_jsonl(args.output_generated_jsonl, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
