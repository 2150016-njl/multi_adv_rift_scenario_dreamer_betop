#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha1_path(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def build_source_index(raw_dir):
    index = defaultdict(list)

    for path in sorted(Path(raw_dir).rglob("*.json")):
        key = sha1_path(path)
        index[key].append(path)

    return index


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-records", type=int, default=95)

    args = ap.parse_args()

    dataset = Path(args.dataset)
    raw_dir = Path(args.snapshot_dir)
    output = Path(args.output)

    source_index = build_source_index(raw_dir)

    records = []

    for line_idx, record in read_jsonl(dataset):

        if line_idx >= args.max_records:
            break

        source_key = record.get("source_file_sha1")

        if not source_key:
            raise RuntimeError(
                f"record {line_idx}: missing source_file_sha1; "
                "refuse provenance inference"
            )

        candidates = source_index.get(source_key, [])

        if len(candidates) == 0:
            raise RuntimeError(
                f"record {line_idx}: source_file_sha1={source_key} "
                "does not resolve to any RAW snapshot"
            )

        if len(candidates) != 1:
            raise RuntimeError(
                f"record {line_idx}: source_file_sha1={source_key} "
                f"resolves to {len(candidates)} paths; "
                "refuse collision resolution"
            )

        path = candidates[0]

        payload = json.loads(path.read_text(encoding="utf-8"))
        scene = payload.get("scene", payload)

        if scene.get("town") != record.get("town"):
            raise RuntimeError(
                f"record {line_idx}: town mismatch"
            )

        source_sha256 = sha256_file(path)

        records.append({
            "record_index": line_idx,
            "dataset_idx": record.get("idx"),
            "scene_id": record.get("scene_id"),
            "episode_key": record.get("episode_key"),
            "source_file_sha1": source_key,
            "source_snapshot_path": str(path.resolve()),
            "source_snapshot_sha256": source_sha256,
            "raw_scene_id": scene.get("scene_id"),
            "raw_town": scene.get("town"),
            "raw_frame_id": scene.get("frame_id"),
            "raw_worker": path.parent.name,
        })

    if len(records) != args.max_records:
        raise RuntimeError(
            f"expected {args.max_records} records, got {len(records)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n")

    print(json.dumps({
        "passed": True,
        "records": len(records),
        "output": str(output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()