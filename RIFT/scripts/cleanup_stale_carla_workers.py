#!/usr/bin/env python
"""Safely report or terminate CARLA servers from explicitly named old runs.

The script never kills a generic/unclassified CarlaUE4 process.  A process is
eligible only when an ancestor command line contains one of --stale_root and
does not contain an output root from an active launch plan.  Default mode is
report-only; --terminate sends SIGTERM only (never SIGKILL).
"""
import argparse
import json
import os
import re
import signal
import time
from pathlib import Path


def command_line(pid):
    try:
        return Path("/proc").joinpath(str(pid), "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def parent_pid(pid):
    try:
        for line in Path("/proc").joinpath(str(pid), "status").read_text(errors="replace").splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return 0


def ancestry(pid, limit=20):
    rows = []
    seen = set()
    while pid > 1 and pid not in seen and len(rows) < limit:
        seen.add(pid)
        rows.append((pid, command_line(pid)))
        pid = parent_pid(pid)
    return rows


def active_roots(plan_paths):
    roots = set()
    for path in plan_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        root = data.get("output_root")
        if root:
            roots.add(str(root))
    return sorted(roots)


def is_carla(cmd):
    return "CarlaUE4" in cmd and "-carla-rpc-port=" in cmd


def rpc_port(cmd):
    match = re.search(r"-carla-rpc-port=(\d+)", cmd)
    return int(match.group(1)) if match else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--active_launch_plan", type=Path, action="append", required=True,
                   help="Repeat for every launch plan that is currently allowed to run.")
    p.add_argument("--stale_root", action="append", required=True,
                   help="Repeat only for old experiment output roots you explicitly authorize for cleanup.")
    p.add_argument("--stale_rpc_port", type=int, action="append", default=[],
                   help="Explicit old CARLA RPC port allowed for cleanup when the old process has already lost its launcher ancestry.")
    p.add_argument("--terminate", action="store_true", default=False,
                   help="Send SIGTERM to eligible stale CARLA servers. Default is report-only.")
    p.add_argument("--output_json", type=Path, required=True)
    args = p.parse_args()

    if not all(path.is_file() for path in args.active_launch_plan):
        raise ValueError("every --active_launch_plan must be an existing file")
    active = active_roots(args.active_launch_plan)
    stale = sorted(set(args.stale_root))
    overlap = [root for root in stale if any(root.startswith(a) or a.startswith(root) for a in active)]
    if overlap:
        raise ValueError("a stale_root overlaps an active output root: " + repr(overlap))

    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = command_line(pid)
        if not is_carla(cmd):
            continue
        chain = ancestry(pid)
        joined = "\n".join(text for _, text in chain)
        active_match = next((root for root in active if root in joined), None)
        stale_match = next((root for root in stale if root in joined), None)
        port = rpc_port(cmd)
        stale_port_match = port if port in set(args.stale_rpc_port) else None
        action = ("keep_active" if active_match else
                  "terminate_stale" if stale_match else
                  "terminate_stale_port" if stale_port_match is not None else
                  "unclassified_keep")
        row = {"pid": pid, "carla_command": cmd, "action": action,
               "rpc_port": port, "active_root_match": active_match, "stale_root_match": stale_match,
               "stale_rpc_port_match": stale_port_match,
               "ancestry_pids": [p for p, _ in chain]}
        if args.terminate and action in {"terminate_stale", "terminate_stale_port"}:
            try:
                os.kill(pid, signal.SIGTERM)
                row["signal"] = "SIGTERM"
            except ProcessLookupError:
                row["signal"] = "already_exited"
            except PermissionError as exc:
                row["signal_error"] = str(exc)
        rows.append(row)

    result = {
        "schema_version": "stale_carla_cleanup_v1",
        "mode": "terminate" if args.terminate else "report_only",
        "active_launch_plans": [str(path) for path in args.active_launch_plan],
        "active_output_root_count": len(active),
        "authorized_stale_roots": stale,
        "processes": rows,
        "note": "Unclassified CARLA processes are intentionally not signalled.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
