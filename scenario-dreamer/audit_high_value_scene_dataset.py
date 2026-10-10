#!/usr/bin/env python
"""Audit high-value scene-generation JSONL datasets."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=Path, default=None)
    parser.add_argument("--val_jsonl", type=Path, default=None)
    parser.add_argument("--test_jsonl", type=Path, default=None)
    parser.add_argument("--jsonl", action="append", default=[],
                        help="Extra JSONL to audit. May be repeated.")
    parser.add_argument("--expected_mode", choices=["inpainting", "full_generation"], default=None)
    parser.add_argument("--expected_num_generate_agents", type=int, default=None)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def state_generation_schema_is_clean(record: Dict) -> bool:
    forbidden = ("target_mask", "actor_scores", "cbv_actor_ids", "selected_actor_ids", "state_delta", "state_deltas")
    text = json.dumps(record, ensure_ascii=False)
    return not any(name in text for name in forbidden)


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_inputs(args) -> Iterable[Tuple[str, Path]]:
    for split, path in (("train", args.train_jsonl), ("val", args.val_jsonl), ("test", args.test_jsonl)):
        if path is not None:
            yield split, path
    for idx, raw in enumerate(args.jsonl):
        path = Path(raw)
        yield path.stem or f"jsonl_{idx}", path


def target_states(record: Dict) -> List:
    target = record.get("target") or {}
    return list(target.get("generated_agent_states") or target.get("agent_states") or [])


def observed_states(record: Dict) -> List:
    condition = record.get("condition") or {}
    return list(condition.get("observed_agent_states") or [])


def observed_types(record: Dict) -> List:
    condition = record.get("condition") or {}
    return list(condition.get("observed_agent_types") or [])


def mode_from_schema(record: Dict) -> str:
    schema = str(record.get("schema_version", ""))
    if "full_generation" in schema:
        return "full_generation"
    if "inpainting" in schema:
        return "inpainting"
    return "unknown"


def count_out_of_fov(states: List, fov: float) -> int:
    half = fov / 2.0
    count = 0
    for state in states:
        if len(state) < 2:
            count += 1
            continue
        x, y = float(state[0]), float(state[1])
        if not math.isfinite(x) or not math.isfinite(y) or abs(x) > half or abs(y) > half:
            count += 1
    return count


def audit_record(record: Dict, split: str, args) -> List[str]:
    errors = []
    if not state_generation_schema_is_clean(record):
        errors.append("forbidden_selector_or_delta_field")
    if "condition" not in record:
        errors.append("missing_condition")
    if "target" not in record:
        errors.append("missing_target")
    mode = mode_from_schema(record)
    if args.expected_mode and mode != args.expected_mode:
        errors.append(f"mode_mismatch:{mode}")

    targets = target_states(record)
    observed = observed_states(record)
    types = observed_types(record)
    condition = record.get("condition") or {}
    audit = record.get("rift_audit") or {}
    if args.expected_num_generate_agents is not None and len(targets) != args.expected_num_generate_agents:
        errors.append(f"target_count_mismatch:{len(targets)}")
    if condition.get("num_generate_agents") is not None:
        try:
            num_generate_agents = int(condition["num_generate_agents"])
        except Exception:
            num_generate_agents = None
            errors.append(f"condition_num_generate_agents_invalid:{condition['num_generate_agents']}")
        if num_generate_agents is not None and num_generate_agents != len(targets):
            errors.append(f"condition_num_generate_agents_mismatch:{condition['num_generate_agents']}")
    if types and len(types) != len(observed):
        errors.append(f"observed_type_count_mismatch:{len(types)}")
    if len(targets) == 0:
        errors.append("no_target_states")
    if 1 + len(observed) + len(targets) > args.max_num_agents:
        errors.append(f"scenario_dreamer_agent_capacity_exceeded:{1 + len(observed) + len(targets)}")
    if count_out_of_fov(targets, args.fov) > 0:
        errors.append("target_out_of_fov")
    if count_out_of_fov(observed, args.fov) > 0:
        errors.append("observed_out_of_fov")
    if split in ("train", "val", "test") and record.get("split") and record["split"] != split:
        errors.append(f"split_field_mismatch:{record['split']}")
    policy = context_policy(record)
    if policy.startswith("sparse_"):
        before = audit.get("observed_agent_count_before_context_sampling")
        after = audit.get("observed_agent_count_after_context_sampling")
        if before is None:
            errors.append("sparse_missing_observed_before_sampling")
        if after is None:
            errors.append("sparse_missing_observed_after_sampling")
        if before is not None and after is not None:
            try:
                before_count = int(before)
                after_count = int(after)
            except Exception:
                errors.append("sparse_observed_sampling_count_invalid")
                before_count = None
                after_count = None
            if before_count is None or after_count is None:
                return errors
            if before_count < after_count:
                errors.append("sparse_observed_after_exceeds_before")
            if after_count != len(observed):
                errors.append(f"sparse_observed_after_mismatch:{after_count}")
    return errors


def summarize_numeric(values: List[float]) -> Dict:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def context_policy(record: Dict) -> str:
    context = record.get("context_sampling") or {}
    return str(context.get("policy") or "unspecified")


def observed_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3_4"
    return "5_plus"


def update_split_context_summary(container: Dict, split: str, observed_count: int, policy: str) -> None:
    item = container.setdefault(split, {
        "num_records": 0,
        "observed_count_buckets": Counter(),
        "context_sampling_policy_counts": Counter(),
        "low_observed_count_leq2": 0,
        "zero_observed_count": 0,
    })
    item["num_records"] += 1
    item["observed_count_buckets"][observed_bucket(observed_count)] += 1
    item["context_sampling_policy_counts"][policy] += 1
    item["low_observed_count_leq2"] += int(observed_count <= 2)
    item["zero_observed_count"] += int(observed_count == 0)


def finalize_split_context_summary(container: Dict) -> Dict:
    finalized = {}
    for split, item in sorted(container.items()):
        total = item["num_records"]
        finalized[split] = {
            "num_records": total,
            "observed_count_buckets": dict(sorted(item["observed_count_buckets"].items())),
            "context_sampling_policy_counts": dict(item["context_sampling_policy_counts"].most_common()),
            "low_observed_count_leq2": item["low_observed_count_leq2"],
            "low_observed_count_leq2_rate": item["low_observed_count_leq2"] / total if total else 0.0,
            "zero_observed_count": item["zero_observed_count"],
            "zero_observed_count_rate": item["zero_observed_count"] / total if total else 0.0,
        }
    return finalized


def main():
    args = parse_args()
    datasets = list(iter_inputs(args))
    if not datasets:
        raise ValueError("provide --train_jsonl/--val_jsonl/--test_jsonl or at least one --jsonl")

    all_records = []
    errors_by_type = defaultdict(int)
    examples = []
    episode_splits = defaultdict(set)
    target_counts = []
    observed_counts = []
    positive_counts = []
    context_policies = Counter()
    observed_buckets = Counter()
    sparse_context_records = 0
    observed_before_sampling = []
    observed_after_sampling = []
    split_context_summary = {}

    for split, path in datasets:
        records = read_jsonl(path)
        for idx, record in enumerate(records):
            record_errors = audit_record(record, split, args)
            for error in record_errors:
                errors_by_type[error] += 1
            if record_errors and len(examples) < 20:
                examples.append({
                    "split": split,
                    "index": idx,
                    "scene_id": record.get("scene_id"),
                    "errors": record_errors,
                })
            episode_key = record.get("episode_key")
            if episode_key:
                episode_splits[str(episode_key)].add(split)
            targets = target_states(record)
            observed = observed_states(record)
            target_counts.append(len(targets))
            observed_counts.append(len(observed))
            observed_buckets[observed_bucket(len(observed))] += 1
            policy = context_policy(record)
            context_policies[policy] += 1
            sparse_context_records += int(policy.startswith("sparse_"))
            update_split_context_summary(split_context_summary, split, len(observed), policy)
            audit = record.get("rift_audit") or {}
            positive_counts.append(float(audit.get("num_rift_positive_agents_original", 0)))
            if audit.get("observed_agent_count_before_context_sampling") is not None:
                observed_before_sampling.append(float(audit["observed_agent_count_before_context_sampling"]))
            if audit.get("observed_agent_count_after_context_sampling") is not None:
                observed_after_sampling.append(float(audit["observed_agent_count_after_context_sampling"]))
            all_records.append(record)

    leaked = {
        key: sorted(value)
        for key, value in episode_splits.items()
        if len(value) > 1 and set(value).issubset({"train", "val", "test"})
    }
    for _ in leaked:
        errors_by_type["episode_split_leakage"] += 1

    summary = {
        "schema_version": "hv_init_scene_dataset_audit_v1",
        "passed": not errors_by_type,
        "num_records": len(all_records),
        "num_datasets": len(datasets),
        "fov": args.fov,
        "max_num_agents": args.max_num_agents,
        "target_count": summarize_numeric(target_counts),
        "observed_count": summarize_numeric(observed_counts),
        "observed_count_buckets": dict(sorted(observed_buckets.items())),
        "zero_observed_count": observed_buckets.get("0", 0),
        "zero_observed_count_rate": observed_buckets.get("0", 0) / len(all_records) if all_records else 0.0,
        "low_observed_count_leq2": sum(
            count for bucket, count in observed_buckets.items()
            if bucket in {"0", "1", "2"}
        ),
        "low_observed_count_leq2_rate": (
            sum(count for bucket, count in observed_buckets.items() if bucket in {"0", "1", "2"}) / len(all_records)
            if all_records else 0.0
        ),
        "context_sampling_policy_counts": dict(context_policies.most_common()),
        "split_context_summary": finalize_split_context_summary(split_context_summary),
        "sparse_context_record_count": sparse_context_records,
        "sparse_context_record_rate": sparse_context_records / len(all_records) if all_records else 0.0,
        "observed_count_before_context_sampling": summarize_numeric(observed_before_sampling),
        "observed_count_after_context_sampling": summarize_numeric(observed_after_sampling),
        "rift_positive_count_original": summarize_numeric(positive_counts),
        "errors_by_type": dict(sorted(errors_by_type.items())),
        "examples": examples,
        "episode_split_leakage_examples": dict(list(leaked.items())[:20]),
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
