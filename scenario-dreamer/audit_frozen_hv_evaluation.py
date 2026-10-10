#!/usr/bin/env python
"""Audit that two frozen HV evaluations are comparable raw-sample reports."""

import argparse
import json
from pathlib import Path


REQUIRED_RAW_METRICS = (
    "position_l2", "heading_abs", "speed_l1", "size_l1",
    "physical_invalid_rate", "off_fov_failure_count",
    "obb_axis_clearance_min", "obb_overlap_count",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_json", required=True, type=Path)
    parser.add_argument("--right_json", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--expected_jsonl_sha256", default=None)
    parser.add_argument("--expected_seed", type=int, default=0)
    return parser.parse_args()


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def audit_one(name, report, args):
    errors = []
    if int(report.get("num_samples_per_condition", -1)) != 1:
        errors.append("not_raw_single_sample")
    if report.get("raw_state_postprocessing") != "none":
        errors.append("raw_state_postprocessing_not_none")
    if report.get("clamp_generated_states", False):
        errors.append("clamp_enabled")
    if int(report.get("base_seed", -1)) != int(args.expected_seed):
        errors.append("seed_mismatch")
    if not report.get("checkpoint_sha256"):
        errors.append("missing_checkpoint_sha256")
    if not report.get("jsonl_sha256"):
        errors.append("missing_jsonl_sha256")
    if args.expected_jsonl_sha256 and report.get("jsonl_sha256") != args.expected_jsonl_sha256:
        errors.append("jsonl_sha256_mismatch")
    sampled = report.get("sampled") or {}
    for key in REQUIRED_RAW_METRICS:
        if key not in sampled:
            errors.append(f"missing_sampled_metric:{key}")
    return {"name": name, "errors": errors, "checkpoint_sha256": report.get("checkpoint_sha256"),
            "jsonl_sha256": report.get("jsonl_sha256"), "num_records": report.get("num_records")}


def main():
    args = parse_args()
    left, right = load(args.left_json), load(args.right_json)
    left_audit, right_audit = audit_one("left", left, args), audit_one("right", right, args)
    errors = left_audit["errors"] + right_audit["errors"]
    if left.get("jsonl_sha256") != right.get("jsonl_sha256"):
        errors.append("left_right_jsonl_differ")
    if left.get("num_records") != right.get("num_records"):
        errors.append("left_right_num_records_differ")
    result = {
        "schema_version": "frozen_hv_evaluation_audit_v1",
        "passed": not errors,
        "errors": errors,
        "left": left_audit,
        "right": right_audit,
        "comparison_rule": "raw single samples only; no clamp, oracle, or online selection",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
