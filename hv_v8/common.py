from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


class ContractError(RuntimeError):
    pass


def require(cond: bool, message: str) -> None:
    if not cond:
        raise ContractError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        rows = []
        for p in sorted(x for x in path.rglob("*") if x.is_file()):
            rows.append((str(p.relative_to(path)), sha256_path(p)))
        return sha256_bytes(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode())
    raise FileNotFoundError(path)


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as e:
        return f"unavailable:{type(e).__name__}:{e}"


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"Expected JSON object: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ContractError(f"{path}:{lineno}: invalid JSON: {e}") from e
            require(isinstance(obj, dict), f"{path}:{lineno}: JSONL row must be object")
            rows.append(obj)
    return rows


def write_json(path: Path, obj: Any, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def new_run_dir(base: Path, name: str) -> Path:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    for i in range(10000):
        p = base / f"{name}_{i:04d}"
        try:
            p.mkdir()
            return p
        except FileExistsError:
            continue
    raise RuntimeError(f"Cannot allocate unique run directory under {base}")


def yaw_wrap(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def lane_uid(wp: Any) -> tuple[int, int, int]:
    return (int(wp.road_id), int(wp.section_id), int(wp.lane_id))


def lane_uid_str(uid: tuple[int, int, int]) -> str:
    return f"{uid[0]}:{uid[1]}:{uid[2]}"


def finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))
