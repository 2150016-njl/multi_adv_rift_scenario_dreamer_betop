#!/usr/bin/env python
"""Audit whether V4C records select inconsistent target pairs per snapshot.

This is deliberately a data audit, not a target rewrite.  Target actor IDs are
removed from final V4C records to prevent label leakage, so target pairs are
compared through quantized raw target states plus their target-only route
supervision.  A same-scene group can legitimately contain different model
conditions because each chosen target is removed from observed context.  The
report makes that distinction explicit.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", action="append", required=True,
                        help="One or more V4C split JSONLs; may be repeated.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--state_quantization_m", type=float, default=0.25)
    parser.add_argument("--heading_quantization_rad", type=float, default=0.05)
    parser.add_argument("--max_examples", type=int, default=30)
    return parser.parse_args()


def finite_round(value, step):
    value = float(value)
    if not math.isfinite(value):
        return "nonfinite"
    return int(round(value / step))


def digest(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


def state_signature(state, args):
    try:
        return (
            finite_round(state[0], args.state_quantization_m),
            finite_round(state[1], args.state_quantization_m),
            finite_round(state[2], args.heading_quantization_rad),
            finite_round(state[3], 0.25),
            finite_round(state[4], 0.10),
            finite_round(state[5], 0.10),
        )
    except (IndexError, TypeError, ValueError):
        return ("invalid_state",)


def target_pair_signature(record, args):
    target = record.get("target") or {}
    states = [state_signature(row, args) for row in target.get("generated_agent_states") or []]
    routes = []
    for item in target.get("route_interaction_supervision") or []:
        item = item or {}
        anchor = item.get("conflict_point_local_xy") or [None, None]
        try:
            routes.append((
                finite_round(anchor[0], args.state_quantization_m),
                finite_round(anchor[1], args.state_quantization_m),
                finite_round(item.get("target_route_distance_to_conflict_m"), 1.0),
                finite_round(item.get("ego_route_distance_to_conflict_m"), 1.0),
            ))
        except (IndexError, TypeError, ValueError):
            routes.append(("invalid_route_supervision",))
    return digest({"states": sorted(states), "routes": sorted(routes)})


def fleet_signature(record, args):
    condition = record.get("condition") or {}
    observed = [state_signature(row, args) for row in condition.get("observed_agent_states") or []]
    target = [state_signature(row, args) for row in (record.get("target") or {}).get("generated_agent_states") or []]
    # The physical vehicle set before target masking. It intentionally omits
    # route/map, so it tests target-selection variation in the same local fleet.
    return digest(sorted(observed + target))


def condition_signature(record, args):
    condition = record.get("condition") or {}
    return digest({
        "observed": sorted(state_signature(row, args) for row in condition.get("observed_agent_states") or []),
        "ego_route": condition.get("ego_route"),
        "map": condition.get("map"),
        "num_generate_agents": condition.get("num_generate_agents"),
    })


def summarize_groups(groups, key_name, args):
    duplicate_groups = [rows for rows in groups.values() if len(rows) > 1]
    multi_pair = [rows for rows in duplicate_groups if len({row["target_pair"] for row in rows}) > 1]
    same_input_multi_pair = [rows for rows in multi_pair if len({row["condition"] for row in rows}) == 1]
    examples = []
    for rows in multi_pair[:args.max_examples]:
        examples.append({
            key_name: rows[0][key_name],
            "records": len(rows),
            "unique_target_pairs": len({row["target_pair"] for row in rows}),
            "unique_model_conditions": len({row["condition"] for row in rows}),
            "splits": sorted({row["split"] for row in rows}),
            "scene_ids": sorted({row["scene_id"] for row in rows})[:8],
        })
    return {
        "group_count": len(groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_record_count": sum(len(rows) for rows in duplicate_groups),
        "groups_with_multiple_target_pairs": len(multi_pair),
        "records_in_multiple_target_pair_groups": sum(len(rows) for rows in multi_pair),
        "same_model_condition_with_multiple_target_pairs": len(same_input_multi_pair),
        "examples": examples,
    }


def main():
    args = parse_args()
    rows, seen_sources = [], set()
    for path in args.jsonl:
        split = Path(path).stem
        for index, record in enumerate(read_jsonl(path)):
            source = str(record.get("source_file_sha1") or "missing_source_file_sha1")
            scene = str(record.get("scene_id") or "missing_scene_id")
            item = {
                "split": split, "index": index, "source": source, "scene_id": scene,
                "target_pair": target_pair_signature(record, args),
                "condition": condition_signature(record, args),
                "fleet": fleet_signature(record, args),
            }
            rows.append(item)
            seen_sources.add(source)
    if not rows:
        raise ValueError("no records")
    by_source, by_scene, by_fleet, by_condition = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
        by_scene[row["scene_id"]].append(row)
        by_fleet[row["fleet"]].append(row)
        by_condition[row["condition"]].append(row)
    source_cross_split = [key for key, value in by_source.items() if len({row["split"] for row in value}) > 1]
    report = {
        "schema_version": "v4c_snapshot_target_consistency_audit_v1",
        "inputs": args.jsonl,
        "records": len(rows),
        "unique_source_file_sha1": len(seen_sources),
        "policy": {
            "mutates_dataset": False,
            "target_ids_available": False,
            "target_pair_comparison": "quantized raw target states plus target-only route supervision",
            "key_interpretation": (
                "A scene/fleet group with multiple target pairs but different model conditions indicates target masking "
                "changes context. Only same_model_condition_with_multiple_target_pairs is a direct contradictory "
                "supervision condition."
            ),
        },
        "by_source_file": summarize_groups(by_source, "source", args),
        "by_scene_id": summarize_groups(by_scene, "scene_id", args),
        "by_local_fleet": summarize_groups(by_fleet, "fleet", args),
        "by_complete_model_condition": summarize_groups(by_condition, "condition", args),
        "source_file_sha1_cross_split_count": len(source_cross_split),
        "source_file_sha1_cross_split_examples": source_cross_split[:args.max_examples],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
