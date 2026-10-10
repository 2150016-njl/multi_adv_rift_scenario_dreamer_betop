#!/usr/bin/env python
"""Read-only audit of V4C target validity and conditional-mode ambiguity.

Exact repeated model inputs are rare in driving snapshots.  This script keeps
that fact explicit: it reports exact quantized-input duplicates separately
from *coarsened* ego-route/map neighborhoods.  Diversity within a coarsened
neighborhood is a risk signal for an under-conditioned target task, not proof
that two literally identical model inputs require different outputs.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import eval_high_value_scene_generator_v4a as v4a
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--min_initial_clearance", type=float, default=2.5)
    parser.add_argument("--exact_quantization_m", type=float, default=0.10)
    parser.add_argument("--coarse_quantization_m", type=float, default=4.0)
    parser.add_argument("--anchor_quantization_m", type=float, default=4.0)
    parser.add_argument("--distance_quantization_m", type=float, default=5.0)
    parser.add_argument("--max_records", type=int, default=-1)
    return parser.parse_args()


def quantize(value, step):
    if isinstance(value, float):
        return round(value / step) * step if math.isfinite(value) else "nonfinite"
    if isinstance(value, int) or value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, list):
        return [quantize(item, step) for item in value]
    if isinstance(value, dict):
        return {str(key): quantize(value[key], step) for key in sorted(value)}
    return str(value)


def digest(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


def target_mode(record, args):
    supervision = list((record.get("target") or {}).get("route_interaction_supervision") or [])
    mode = []
    for item in supervision:
        item = item or {}
        anchor = item.get("conflict_point_local_xy") or [None, None]
        polyline = item.get("target_route_polyline_local_xy") or []
        direction = None
        if len(polyline) >= 2:
            try:
                dx = float(polyline[1][0]) - float(polyline[0][0])
                dy = float(polyline[1][1]) - float(polyline[0][1])
                direction = round(math.atan2(dy, dx) / (math.pi / 6.0))
            except (IndexError, TypeError, ValueError):
                pass
        try:
            entry = (
                round(float(anchor[0]) / args.anchor_quantization_m),
                round(float(anchor[1]) / args.anchor_quantization_m),
                direction,
                round(float(item.get("target_route_distance_to_conflict_m")) / args.distance_quantization_m),
                round(float(item.get("ego_route_distance_to_conflict_m")) / args.distance_quantization_m),
            )
        except (IndexError, TypeError, ValueError):
            entry = ("missing_route_supervision",)
        mode.append(entry)
    return tuple(sorted(mode))


def target_validity(record, args):
    states = list((record.get("target") or {}).get("generated_agent_states") or [])
    target = __import__("torch").tensor(states, dtype=__import__("torch").float32)
    raw = v4a.raw_geometry(states, record, args.default_lane_half_width, args.min_initial_clearance)
    associated = v4a.associated_lane_geometry(states, target, record, args.default_lane_half_width)
    forward = v4a.forward_geometry(states, args)
    supervision = list((record.get("target") or {}).get("route_interaction_supervision") or [])
    route_valid = 0
    route_score, route_gap, route_threshold = [], [], []
    for item in supervision:
        try:
            gap = float(item["route_distance_difference_m"])
            threshold = float(item["route_distance_difference_threshold_m"])
            score = float(item["interaction_strength"])
            route_valid += int(math.isfinite(gap) and math.isfinite(threshold) and gap <= threshold)
            route_score.append(score)
            route_gap.append(gap)
            route_threshold.append(threshold)
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "target_count": len(states),
        "forward_violation_count": forward["generated_forward_violation_count"],
        "nearest_lane_corridor_failure_count": raw["nearest_lane_corridor_failure_count"],
        "associated_lane_corridor_failure_count": associated["associated_lane_corridor_failure_count"],
        "pair_clearance_failure_count": raw["pair_clearance_failure_count"],
        "route_interaction_supervision_count": len(supervision),
        "route_conflict_label_valid_count": route_valid,
        "mean_interaction_strength": sum(route_score) / len(route_score) if route_score else None,
        "mean_route_gap_m": sum(route_gap) / len(route_gap) if route_gap else None,
        "mean_route_threshold_m": sum(route_threshold) / len(route_threshold) if route_threshold else None,
    }


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def main():
    args = parse_args()
    records = read_jsonl(args.jsonl)
    if args.max_records > 0:
        records = records[:args.max_records]
    if not records:
        raise ValueError("no records")
    exact_groups, coarse_groups = defaultdict(list), defaultdict(list)
    aggregate = Counter()
    for index, record in enumerate(records):
        condition = record.get("condition") or {}
        exact_groups[digest(quantize(condition, args.exact_quantization_m))].append(index)
        coarse_condition = {
            "ego_route": condition.get("ego_route"),
            "map": condition.get("map"),
            "observed_actor_count": len(condition.get("observed_agent_states") or []),
        }
        coarse_groups[digest(quantize(coarse_condition, args.coarse_quantization_m))].append(index)
        validity = target_validity(record, args)
        aggregate["target_actors"] += validity["target_count"]
        aggregate["forward_valid_actors"] += validity["target_count"] - validity["forward_violation_count"]
        aggregate["nearest_lane_valid_actors"] += validity["target_count"] - validity["nearest_lane_corridor_failure_count"]
        aggregate["associated_lane_valid_actors"] += validity["target_count"] - validity["associated_lane_corridor_failure_count"]
        aggregate["route_conflict_labeled_actors"] += validity["route_conflict_label_valid_count"]
        aggregate["pair_clearance_valid_records"] += int(validity["pair_clearance_failure_count"] == 0)
    modes = {index: target_mode(record, args) for index, record in enumerate(records)}
    exact_duplicate = [indices for indices in exact_groups.values() if len(indices) > 1]
    coarse_duplicate = [indices for indices in coarse_groups.values() if len(indices) > 1]
    coarse_multimode = [indices for indices in coarse_duplicate if len({modes[index] for index in indices}) > 1]
    examples = []
    for indices in coarse_multimode[:20]:
        examples.append({
            "coarse_context_size": len(indices),
            "unique_target_modes": len({modes[index] for index in indices}),
            "scene_ids": [records[index].get("scene_id") for index in indices[:8]],
            "modes": [list(modes[index]) for index in indices[:8]],
        })
    report = {
        "schema_version": "v4c_target_identifiability_audit_v1",
        "input_jsonl": args.jsonl,
        "num_records": len(records),
        "target_validity": {
            "target_actor_count": aggregate["target_actors"],
            "forward_valid_actor_rate": fraction(aggregate["forward_valid_actors"], aggregate["target_actors"]),
            "nearest_lane_valid_actor_rate": fraction(aggregate["nearest_lane_valid_actors"], aggregate["target_actors"]),
            "associated_lane_valid_actor_rate": fraction(aggregate["associated_lane_valid_actors"], aggregate["target_actors"]),
            "route_conflict_label_valid_rate": fraction(aggregate["route_conflict_labeled_actors"], aggregate["target_actors"]),
            "pair_clearance_valid_record_rate": fraction(aggregate["pair_clearance_valid_records"], len(records)),
        },
        "conditional_mode_audit": {
            "exact_model_input_quantization_m": args.exact_quantization_m,
            "exact_duplicate_group_count": len(exact_duplicate),
            "exact_duplicate_record_count": sum(len(group) for group in exact_duplicate),
            "coarse_ego_route_map_quantization_m": args.coarse_quantization_m,
            "coarse_duplicate_group_count": len(coarse_duplicate),
            "coarse_duplicate_record_count": sum(len(group) for group in coarse_duplicate),
            "coarse_groups_with_multiple_target_route_modes": len(coarse_multimode),
            "coarse_multimode_group_rate": fraction(len(coarse_multimode), len(coarse_duplicate)),
            "caveat": (
                "Exact groups include the complete model condition; coarse groups deliberately omit observed-state "
                "coordinates and quantize route/map geometry. Coarse multimodality is an under-conditioning risk "
                "signal, not proof that identical model inputs map to different targets."
            ),
            "examples": examples,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
