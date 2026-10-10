#!/usr/bin/env python
"""Audit immutable A0 -> A1 -> A2 V4-A checkpoint lineage."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED = ("a0_data_only", "a1_forward_support", "a2_obb_clearance")
COMMON = ("train_jsonl", "val_jsonl", "num_generate_agents", "forward_min_x",
          "target_fov_margin", "max_num_agents", "autoencoder_checkpoint", "ldm_checkpoint")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0_checkpoint", type=Path, required=True)
    parser.add_argument("--a1_checkpoint", type=Path, required=True)
    parser.add_argument("--a2_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    config = metadata.get("config") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or metadata.get("trainer") != "v4a" or not isinstance(config, dict):
        raise ValueError(f"{path} is not an audited V4-A trainer checkpoint")
    return metadata, config


def main():
    args = parse_args()
    entries = []
    errors = []
    for expected, path in zip(EXPECTED, (args.a0_checkpoint, args.a1_checkpoint, args.a2_checkpoint)):
        metadata, config = load(path)
        if metadata.get("v4a_stage") != expected or config.get("v4a_stage") != expected:
            errors.append(f"stage_mismatch:{expected}")
        entries.append({"stage": expected, "path": str(path), "sha256": sha256(path),
                        "metadata": metadata, "config": config})
    a0, a1, a2 = entries
    if a1["config"].get("parent_checkpoint_sha256") != a0["sha256"]:
        errors.append("a1_parent_sha256_mismatch")
    if a2["config"].get("parent_checkpoint_sha256") != a1["sha256"]:
        errors.append("a2_parent_sha256_mismatch")
    for key in COMMON:
        values = [entry["config"].get(key) for entry in entries]
        if len({json.dumps(value, sort_keys=True) for value in values}) != 1:
            errors.append(f"common_contract_mismatch:{key}")
    if float(a1["config"].get("forward_support_loss_weight", 0.0)) <= 0.0:
        errors.append("a1_forward_weight_not_positive")
    if float(a1["config"].get("obb_clearance_loss_weight", 0.0)) != 0.0:
        errors.append("a1_obb_weight_not_zero")
    if float(a2["config"].get("forward_support_loss_weight", 0.0)) <= 0.0:
        errors.append("a2_forward_weight_not_positive")
    if float(a2["config"].get("obb_clearance_loss_weight", 0.0)) <= 0.0:
        errors.append("a2_obb_weight_not_positive")
    result = {
        "schema_version": "v4a_training_stage_audit_v1",
        "passed": not errors,
        "errors": errors,
        "stages": [{"stage": item["stage"], "path": item["path"], "sha256": item["sha256"]} for item in entries],
        "note": "Lineage/config audit only; compare raw offline and matched closed-loop results separately.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
