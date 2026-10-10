"""Strict contract audit for V4-A full-FOV, forward-target datasets."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def state_generation_schema_is_clean(record: Dict) -> bool:
    forbidden = ("target_mask", "actor_scores", "cbv_actor_ids", "selected_actor_ids", "state_delta", "state_deltas")
    text = json.dumps(record, ensure_ascii=False)
    return not any(name in text for name in forbidden)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--val_jsonl", type=Path, required=True)
    parser.add_argument("--test_jsonl", type=Path, required=True)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--max_town_fraction", type=float, default=0.35)
    parser.add_argument("--max_topology_fraction", type=float, default=0.75)
    parser.add_argument("--min_records_per_town_per_split", type=int, default=1)
    parser.add_argument("--min_records_per_topology_per_split", type=int, default=1)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def targets(record: Dict) -> List[List[float]]:
    return list((record.get("target") or {}).get("generated_agent_states") or [])


def observed(record: Dict) -> List[List[float]]:
    return list((record.get("condition") or {}).get("observed_agent_states") or [])


def topology(record: Dict) -> str:
    return str((record.get("topology_v4") or {}).get("bucket") or "unknown")


def town(record: Dict) -> str:
    return str(record.get("town") or "unknown")


def out_of_fov(state: List[float], fov: float) -> bool:
    if not isinstance(state, (list, tuple)) or len(state) < 2:
        return True
    try:
        xy = [float(v) for v in state[:2]]
    except (TypeError, ValueError):
        return True
    return not all(math.isfinite(value) for value in xy) or any(abs(value) > fov / 2 for value in xy)


def state_is_finite_vehicle_state(state) -> bool:
    if not isinstance(state, (list, tuple)) or len(state) < 7:
        return False
    try:
        values = [float(value) for value in state[:7]]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values) and values[4] > 0.0 and values[5] > 0.0


def contract_float(contract: Dict, key: str):
    try:
        value = float(contract[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def contract_int(contract: Dict, key: str):
    try:
        return int(contract[key])
    except (KeyError, TypeError, ValueError):
        return None


def audit_record(record: Dict, split: str, args) -> List[str]:
    errors = []
    condition = record.get("condition") or {}
    target_rows = targets(record)
    observed_rows = observed(record)
    contract = record.get("v4_contract") or {}
    map_payload = condition.get("map") or {}
    if record.get("schema_version") != "hv_init_inpainting_v4a_full_fov_forward":
        errors.append("schema_mismatch")
    if not state_generation_schema_is_clean(record):
        errors.append("forbidden_selector_or_delta_field")
    try:
        condition_count = int(condition.get("num_generate_agents", -1))
    except (TypeError, ValueError):
        condition_count = -1
    if len(target_rows) != 2 or condition_count != 2:
        errors.append("fixed_two_target_contract_failed")
    if (record.get("context_sampling") or {}).get("policy") != "full_fov_non_target":
        errors.append("not_full_fov_context_policy")
    if contract.get("condition_region") != "full_fov_non_target":
        errors.append("full_fov_contract_missing")
    if contract.get("target_sector") != "forward":
        errors.append("forward_target_contract_missing")
    contract_forward_min_x = contract_float(contract, "forward_min_x")
    contract_margin = contract_float(contract, "target_fov_margin")
    if contract_forward_min_x is None or abs(contract_forward_min_x - args.forward_min_x) > 1e-6:
        errors.append("forward_min_x_contract_mismatch")
    if contract_margin is None or abs(contract_margin - args.target_fov_margin) > 1e-6:
        errors.append("target_fov_margin_contract_mismatch")
    expected_full_count = len(observed_rows) + len(target_rows)
    if contract_int(contract, "full_fov_actor_count") != expected_full_count:
        errors.append("full_fov_actor_accounting_failed")
    if contract_int(contract, "observed_actor_count") != len(observed_rows):
        errors.append("observed_actor_count_contract_mismatch")
    if contract_int(contract, "target_actor_count") != len(target_rows):
        errors.append("target_actor_count_contract_mismatch")
    if contract.get("agent_capacity_policy") != "reject_record_no_truncation":
        errors.append("capacity_policy_not_explicit_reject")
    if contract_int(contract, "max_num_agents") != args.max_num_agents:
        errors.append("capacity_contract_mismatch")
    try:
        condition_fov = float(condition.get("fov"))
    except (TypeError, ValueError):
        condition_fov = float("nan")
    if not math.isfinite(condition_fov) or abs(condition_fov - args.fov) > 1e-6:
        errors.append("condition_fov_contract_mismatch")
    if 1 + expected_full_count > args.max_num_agents:
        errors.append("scenario_dreamer_capacity_exceeded")
    types = condition.get("observed_agent_types") or []
    if len(types) != len(observed_rows):
        errors.append("observed_type_count_mismatch")
    target_types = (record.get("target") or {}).get("generated_agent_types") or []
    if len(target_types) != len(target_rows):
        errors.append("target_type_count_mismatch")
    limit = args.fov / 2.0 - args.target_fov_margin
    for state in target_rows:
        if not state_is_finite_vehicle_state(state):
            errors.append("target_state_schema_invalid")
            continue
        if out_of_fov(state, args.fov):
            errors.append("target_out_of_fov")
            continue
        if float(state[0]) < args.forward_min_x or float(state[0]) > limit or abs(float(state[1])) > limit:
            errors.append("target_not_in_forward_sector")
    if any(not state_is_finite_vehicle_state(state) for state in observed_rows):
        errors.append("observed_state_schema_invalid")
    if any(out_of_fov(state, args.fov) for state in observed_rows):
        errors.append("observed_out_of_fov")
    if not map_payload.get("lane_polylines") or not map_payload.get("lanes_v3") or not map_payload.get("lanes_v4"):
        errors.append("map_lanes_missing")
    associations = (record.get("target") or {}).get("map_associations_v4") or []
    if len(associations) != len(target_rows):
        errors.append("target_lane_association_count_mismatch")
    for item in associations:
        distance = item.get("distance") if isinstance(item, dict) else None
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            distance = float("nan")
        if not math.isfinite(distance) or distance > args.max_target_lane_distance:
            errors.append("target_lane_association_invalid")
    if record.get("split") != split:
        errors.append("split_field_mismatch")
    return sorted(set(errors))


def count_by(records: Iterable[Dict], key) -> Counter:
    return Counter(key(record) for record in records)


def main():
    args = parse_args()
    if abs(args.fov - 64.0) > 1e-6:
        raise ValueError("V4-A audit requires --fov 64")
    if args.max_num_agents != 61:
        raise ValueError("V4-A audit requires --max_num_agents 61")
    split_rows: List[Tuple[str, List[Dict]]] = [
        ("train", read_jsonl(args.train_jsonl)),
        ("val", read_jsonl(args.val_jsonl)),
        ("test", read_jsonl(args.test_jsonl)),
    ]
    all_rows = [record for _, rows in split_rows for record in rows]
    errors = Counter()
    examples = []
    episodes = defaultdict(set)
    for split, rows in split_rows:
        for index, record in enumerate(rows):
            row_errors = audit_record(record, split, args)
            for error in row_errors:
                errors[error] += 1
            if row_errors and len(examples) < 30:
                examples.append({"split": split, "index": index, "scene_id": record.get("scene_id"), "errors": row_errors})
            episodes[str(record.get("episode_key"))].add(split)
    for episode, memberships in episodes.items():
        if len(memberships) > 1:
            errors["episode_split_leakage"] += 1
            if len(examples) < 30:
                examples.append({"episode_key": episode, "errors": ["episode_split_leakage"], "splits": sorted(memberships)})

    town_counts = count_by(all_rows, town)
    topology_counts = count_by(all_rows, topology)
    total = len(all_rows)
    for label, counts, maximum in (("town", town_counts, args.max_town_fraction), ("topology", topology_counts, args.max_topology_fraction)):
        for key, count in counts.items():
            if total and count / total > maximum + 1e-12:
                errors[f"{label}_fraction_exceeded:{key}"] += 1

    split_summary = {}
    for split, rows in split_rows:
        towns = count_by(rows, town)
        topologies = count_by(rows, topology)
        split_summary[split] = {"records": len(rows), "towns": dict(sorted(towns.items())), "topologies": dict(sorted(topologies.items()))}
        for key in town_counts:
            if towns[key] < args.min_records_per_town_per_split:
                errors[f"town_split_coverage:{split}:{key}"] += 1
        for key in topology_counts:
            if topologies[key] < args.min_records_per_topology_per_split:
                errors[f"topology_split_coverage:{split}:{key}"] += 1

    report = {
        "schema_version": "hv_scene_dataset_v4a_audit_v1",
        "passed": not errors,
        "num_records": total,
        "town_counts": dict(sorted(town_counts.items())),
        "topology_bucket_counts": dict(sorted(topology_counts.items())),
        "split_summary": split_summary,
        "errors": dict(sorted(errors.items())),
        "examples": examples,
        "args": vars(args),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # argparse contains Path values; keep the persisted audit portable while
    # preserving every contract/error field verbatim.
    rendered = json.dumps(report, indent=2, default=str)
    args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
