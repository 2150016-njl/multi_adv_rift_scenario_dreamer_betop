#!/usr/bin/env python
"""Merge audited per-route future-outcome sidecars without altering labels."""

import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_jsonl.exists() or args.output_summary.exists():
        raise FileExistsError("refusing to overwrite a merged future-outcome artifact")
    inputs = sorted(path for path in args.input_dir.rglob("future_outcomes.jsonl")
                    if path.resolve() != args.output_jsonl.resolve())
    if not inputs:
        raise ValueError("no future_outcomes.jsonl files under --input_dir")
    seen, rows, duplicate_rows = set(), [], 0
    for path in inputs:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("snapshot_path")), str(row.get("anchor_actor_id")))
            if key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
            row["sidecar_source_path"] = str(path)
            row["sidecar_source_line"] = number
            rows.append(row)
    if not rows:
        raise ValueError("future outcome sidecars contained no records")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    digest = hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest()
    summary = {
        "schema_version": "hv_future_outcome_merge_v1", "input_dir": str(args.input_dir),
        "input_files": [str(path) for path in inputs], "input_file_count": len(inputs),
        "merged_rows": len(rows), "duplicate_rows_skipped": duplicate_rows,
        "output_jsonl": str(args.output_jsonl), "output_sha256": digest,
        "contract": "current-time anchors plus observation-only future labels; no future state is a spawn target",
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
