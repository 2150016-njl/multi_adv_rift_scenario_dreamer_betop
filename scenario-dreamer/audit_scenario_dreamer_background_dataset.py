#!/usr/bin/env python
"""Strict structural audit for the independent directional background task."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from high_value_init.rift_dataset import read_jsonl


# These are *serialized data-field* bans, not textual-substring bans.  The
# task contract deliberately says "not_future_prediction", so scanning the
# whole JSON text for "future" falsely rejects every valid background record.
# Recursing over keys still catches label/identity/outcome leakage while
# allowing descriptive contract strings.
FORBIDDEN_FIELD_NAMES = {
    "rift_label", "rift_labels", "is_interactive",
    "actor_id", "actor_ids", "cbv_actor_id", "cbv_actor_ids",
    "selected_actor_id", "selected_actor_ids", "actor_score", "actor_scores",
    "future", "future_state", "future_states", "future_trajectory", "future_trajectories",
    "trajectory", "trajectories",
}


def forbidden_field_paths(value, path=""):
    """Return paths of forbidden label/identity/outcome fields in a record."""
    hits = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key).lower()
            child_path = f"{path}.{key}" if path else str(key)
            # Future or trajectory fields with an extended spelling are also
            # forbidden; a value such as the declared task string is not.
            forbidden = (
                name in FORBIDDEN_FIELD_NAMES
                or name.startswith("rift_label")
                or name.startswith("future_")
                or name.endswith("_future")
                or "trajectory" in name
                or name.startswith("cbv_")
                or name.endswith("_score")
            )
            if forbidden:
                hits.append(child_path)
            hits.extend(forbidden_field_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(forbidden_field_paths(child, f"{path}[{index}]"))
    return hits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--count_model", required=True)
    parser.add_argument("--summary", required=True,
                        help="Builder summary.json; audit rejects a dataset built with failed diversity gates.")
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--partition_boundary_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_target_agents", type=int, default=8)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def audit_record(record, split, args):
    errors = []
    if record.get("schema_version") != "scenario_dreamer_background_directional_v1":
        errors.append("schema")
    if record.get("split") != split:
        errors.append("split")
    for field_path in forbidden_field_paths(record):
        errors.append(f"forbidden_field:{field_path}")
    condition, target = record.get("condition") or {}, record.get("target") or {}
    contract = record.get("background_contract") or {}
    if contract.get("task") != "current_front_scene_completion_not_future_prediction":
        errors.append("task_contract")
    if contract.get("condition_partition") != "rear_x_lt_boundary" or contract.get("target_partition") != "front_x_gte_boundary":
        errors.append("partition_contract")
    if abs(float(contract.get("partition_boundary_x", float("nan"))) - args.partition_boundary_x) > 1e-6:
        errors.append("partition_boundary")
    observed = condition.get("observed_agent_states") or []
    targets = target.get("generated_agent_states") or []
    if not 1 <= len(targets) <= args.max_target_agents:
        errors.append("variable_target_count")
    if int(condition.get("num_generate_agents", -1)) != len(targets):
        errors.append("target_count_mismatch")
    if 1 + len(observed) + len(targets) > args.max_num_agents:
        errors.append("capacity")
    limit = args.fov / 2.0 - args.target_fov_margin
    for state in observed:
        if not state or float(state[0]) >= args.partition_boundary_x:
            errors.append("condition_not_rear")
            break
    for state in targets:
        if not state or float(state[0]) < args.partition_boundary_x or float(state[0]) > limit or abs(float(state[1])) > limit:
            errors.append("target_not_front")
            break
        if not all(math.isfinite(float(value)) for value in state[:6]):
            errors.append("nonfinite_target")
            break
    return errors


def main():
    args = parse_args()
    if abs(args.fov - 64.0) > 1e-6 or args.max_num_agents != 61:
        raise ValueError("audit requires the frozen 64 m / capacity-61 contract")
    all_records, failures, episodes = [], [], {}
    for split, path in (("train", args.train_jsonl), ("val", args.val_jsonl), ("test", args.test_jsonl)):
        rows = read_jsonl(path)
        if not rows:
            failures.append({"split": split, "error": "empty"})
        for index, record in enumerate(rows):
            errors = audit_record(record, split, args)
            if errors and len(failures) < 100:
                failures.append({"split": split, "index": index, "scene_id": record.get("scene_id"), "errors": errors})
            episode = str(record.get("episode_key"))
            if episode in episodes and episodes[episode] != split:
                failures.append({"episode_key": episode, "error": "episode_leakage", "splits": [episodes[episode], split]})
            episodes[episode] = split
        all_records.extend(rows)
    builder_summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if builder_summary.get("schema_version") != "scenario_dreamer_background_directional_v1":
        failures.append({"summary": "schema"})
    if builder_summary.get("distribution_failures"):
        failures.append({"summary": "distribution_failures", "values": builder_summary.get("distribution_failures")})
    required_records = int(builder_summary.get("min_accepted_records_required", 0))
    if required_records < 30000:
        failures.append({"summary": "accepted_scale_contract_below_30000", "value": required_records})
    if len(all_records) < required_records:
        failures.append({"summary": "accepted_record_count_below_minimum", "records": len(all_records), "required": required_records})
    model = json.loads(Path(args.count_model).read_text(encoding="utf-8"))
    if model.get("estimated_from_split") != "train_only" or model.get("task") != "current_front_scene_completion_not_future_prediction":
        failures.append({"count_model": "not_train_only_current_scene"})
    if int(model.get("max_target_agents", -1)) != args.max_target_agents:
        failures.append({"count_model": "max_target_agents"})
    report = {
        "passed": not failures,
        "records": len(all_records),
        "episode_count": len(episodes),
        "target_count_histogram": dict(sorted(Counter(len((r.get("target") or {}).get("generated_agent_states") or []) for r in all_records).items())),
        "failures": failures[:100],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
