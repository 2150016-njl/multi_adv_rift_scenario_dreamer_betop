from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

from .common import ContractError, git_commit, read_jsonl, require, sha256_path, write_json


def _required(row: dict[str, Any], keys: list[str], where: str) -> None:
    for k in keys:
        require(k in row and row[k] not in (None, ""), f"{where}: missing {k}")


def _index(rows: list[dict[str, Any]], key: str, name: str) -> dict[Any, dict[str, Any]]:
    out = {}
    for i, r in enumerate(rows):
        _required(r, [key], f"{name}[{i}]")
        value = r[key]
        require(value not in out, f"{name}: duplicate {key}={value}")
        out[value] = r
    return out


def freeze(args: argparse.Namespace) -> int:
    pairs_path = Path(args.pairs).resolve()
    test_path = Path(args.test_index).resolve()
    pairs = read_jsonl(pairs_path)
    test = read_jsonl(test_path)

    require(len(pairs) == 95, f"P0 requires exactly 95 paired rows, got {len(pairs)}")
    require(len(test) == 3727, f"P0 requires exactly 3727 test rows, got {len(test)}")

    pi = _index(pairs, "record_index", "pairs")
    ti = _index(test, "record_index", "test_index")
    require(set(pi).issubset(set(ti)), "Every paired record_index must exist in the 3727-line test index")

    files: dict[str, str] = {}

    def bind(label: str, path: str | Path) -> None:
        p = Path(path).resolve()
        require(p.exists(), f"P0 missing {label}: {p}")
        files[label] = sha256_path(p)

    bind("pairs_jsonl", pairs_path)
    bind("test_index_jsonl", test_path)
    bind("v2_checkpoint", args.v2_checkpoint)
    bind("v3_checkpoint", args.v3_checkpoint)
    bind("rift_recognition_config", args.rift_recognition_config)
    for i, src in enumerate(args.rift_source_file or []):
        bind(f"rift_source_file_{i:03d}", src)

    # The manifest is intentionally explicit: no filename discovery and no
    # "closest checkpoint" logic is allowed.
    for i, row in enumerate(pairs):
        _required(row, ["scene_id", "snapshot_path", "v2_output_path", "v3_output_path"],
                  f"pairs[{i}]")
        require(Path(row["snapshot_path"]).exists(), f"pairs[{i}]: snapshot missing")
        require(Path(row["v2_output_path"]).exists(), f"pairs[{i}]: V2 output missing")
        require(Path(row["v3_output_path"]).exists(), f"pairs[{i}]: V3 output missing")
        require(row["scene_id"] == ti[row["record_index"]]["scene_id"],
                f"record {row['record_index']}: pair/test scene_id mismatch")

    for i, row in enumerate(test):
        _required(row, ["scene_id", "v2_output_path", "v3_output_path"], f"test_index[{i}]")
        require(Path(row["v2_output_path"]).exists(), f"test_index[{i}]: V2 output missing")
        require(Path(row["v3_output_path"]).exists(), f"test_index[{i}]: V3 output missing")

    out = Path(args.output).resolve()
    require(not out.exists(), f"Refuse overwrite: {out}")

    manifest = {
        "schema_version": "hv_v8_p0_manifest_v1",
        "status": "PASS",
        "protocol": {
            "source_of_truth": "1015技术路线.md",
            "main_experiment": "one-shot",
            "best_of_n": "ablation_only",
            "generator_target": "A_initial_native_positive",
            "application_value": "B_closed_loop_ego_effect",
            "vehicle_blueprint": "vehicle.tesla.model3",
            "generated_variables": ["approach_id", "s", "d", "delta_yaw", "speed_mps"],
            "generated_dimensions": ["x", "y", "yaw", "speed_mps"],
            "forbidden_generation_fallbacks": [
                "snap", "clamp", "retry", "rerank", "oracle_replacement",
                "posthoc_pose_correction", "force_k0", "force_k1"
            ],
        },
        "cohort": {
            "paired_records": 95,
            "test_records": 3727,
            "paired_record_indices": sorted(pi),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "carla_version_declared": args.carla_version,
        },
        "git_commit": git_commit(Path(args.repo_root).resolve()),
        "files_sha256": files,
        "rows": {
            "pairs": pairs,
            "test_index": test,
        },
    }
    write_json(out, manifest)
    print(f"P0_PASS {out}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--test-index", required=True)
    p.add_argument("--v2-checkpoint", required=True)
    p.add_argument("--v3-checkpoint", required=True)
    p.add_argument("--rift-recognition-config", required=True)
    p.add_argument("--rift-root", required=True)
    p.add_argument("--rift-source-file", action="append",
                   help="Explicit RIFT source/config files to hash; repeat. Do not hash the whole tree.")
    p.add_argument("--carla-version", default="0.9.15")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        raise SystemExit(freeze(args))
    except ContractError as e:
        print(f"P0_FAIL {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
