#!/usr/bin/env python
"""Render a remaining-task-only recovery launcher for background collection.

The original capacity launcher keeps a worker alive until every task assigned
to it has run.  A stalled CARLA task can therefore hold up all later tasks in
that worker.  This renderer reads the immutable source plan and its job logs,
removes only tasks with an explicit ``completed route <id> round <n>`` marker,
and writes a new isolated launcher for the rest.

Recovery output is intentionally separate from the original collection root.
No completed snapshots, logs, plans, or launch scripts are modified.
"""

import argparse
import json
import re
import shlex
from pathlib import Path


COMPLETED = re.compile(r"^completed route (\S+) round (\d+)\s*$", re.MULTILINE)


def quote(value):
    return shlex.quote(str(value))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_plan", type=Path, required=True)
    parser.add_argument("--completed_log_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--rift_root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--rift_hd_map_dir", type=Path, required=True)
    parser.add_argument("--rift_speed_limits_dir", type=Path, required=True)
    parser.add_argument("--shell_output", type=Path, required=True)
    parser.add_argument("--plan_output", type=Path, required=True)
    parser.add_argument("--run_as_user", default="riftuser")
    parser.add_argument("--gpu_count", type=int, default=4)
    parser.add_argument("--processes_per_gpu", type=int, default=2)
    parser.add_argument("--base_port", type=int, default=39000)
    parser.add_argument("--base_tm_port", type=int, default=59000)
    parser.add_argument("--port_stride", type=int, default=20)
    parser.add_argument("--task_timeout_seconds", type=int, default=1200)
    parser.add_argument("--max_frames_per_episode", type=int, default=200)
    parser.add_argument("--sample_interval_s", type=float, default=1.0)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--map_radius", type=float, default=64.0)
    return parser.parse_args()


def completed_tasks(log_root: Path):
    completed = set()
    for path in sorted(log_root.glob("job*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        completed.update((route_id, int(round_index)) for route_id, round_index in COMPLETED.findall(text))
    return completed


def task_key(task):
    return str(task["route_id"]), int(task["round_index"])


def main():
    args = parse_args()
    if args.gpu_count < 1 or args.processes_per_gpu < 1:
        raise ValueError("gpu_count and processes_per_gpu must be positive")
    if args.port_stride < 2 or args.task_timeout_seconds < 1:
        raise ValueError("port_stride must be >= 2 and task_timeout_seconds must be positive")
    if args.output_root.exists() or args.shell_output.exists() or args.plan_output.exists():
        raise FileExistsError("recovery output_root, shell_output, and plan_output must all be new paths")

    source = json.loads(args.source_plan.read_text(encoding="utf-8"))
    if source.get("schema_version") != "scenario_dreamer_background_capacity_launch_v1":
        raise ValueError("source_plan is not a background capacity launch plan")
    source_tasks = list(source.get("tasks") or [])
    if not source_tasks:
        raise ValueError("source_plan contains no tasks")
    keys = [task_key(task) for task in source_tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("source_plan has duplicate route-round tasks")

    completed = completed_tasks(args.completed_log_root)
    expected = set(keys)
    unknown_completed = completed - expected
    missing = [task for task in source_tasks if task_key(task) not in completed]
    if not missing:
        raise ValueError("no incomplete background tasks remain")

    worker_count = min(len(missing), args.gpu_count * args.processes_per_gpu)
    workers = [{"worker_index": index, "gpu": index % args.gpu_count, "tasks": []}
               for index in range(worker_count)]
    for index, source_task in enumerate(missing):
        worker = workers[index % worker_count]
        task = dict(source_task)
        task["recovery_worker_index"] = worker["worker_index"]
        task["recovery_gpu"] = worker["gpu"]
        task["source_output_dir"] = task.get("output_dir")
        task["output_dir"] = str(
            args.output_root / (
                "worker%02d_gpu%d_route%s_round%03d_seed%s" % (
                    worker["worker_index"], worker["gpu"], task["route_id"],
                    int(task["round_index"]), task["seed"],
                )
            )
        )
        worker["tasks"].append(task)

    ports = []
    for worker in workers:
        worker["base_port"] = args.base_port + worker["worker_index"] * args.port_stride
        worker["base_tm_port"] = args.base_tm_port + worker["worker_index"] * args.port_stride
        ports.extend((worker["base_port"], worker["base_tm_port"]))
    if len(ports) != len(set(ports)) or min(ports) < 1024 or max(ports) > 65535:
        raise ValueError("recovery CARLA/TM ports are invalid or collide")

    plan = {
        "schema_version": "scenario_dreamer_background_capacity_recovery_launch_v1",
        "source_plan": str(args.source_plan),
        "completed_log_root": str(args.completed_log_root),
        "output_root": str(args.output_root),
        "source_total_program_count": len(source_tasks),
        "source_completed_task_count": len(expected & completed),
        "source_unknown_completed_markers": sorted("%s:%s" % item for item in unknown_completed),
        "remaining_task_count": len(missing),
        "task_timeout_seconds": args.task_timeout_seconds,
        "invariants": {
            "replays_only_tasks_without_completed_marker": True,
            "separate_recovery_output_root": True,
            "all_raw_traffic": True,
            "raw_state_postprocessing": "none",
            "failed_task_is_not_marked_completed": True,
        },
        "workers": workers,
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    project_root = args.rift_root.parent
    collector = args.rift_root / "scripts" / "collect_high_value_init_scenes.py"
    lines = [
        "#!/usr/bin/env bash",
        "# Generated remaining-task recovery launcher. Run as root; CARLA runs as riftuser.",
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "if [ \"$(id -u)\" -ne 0 ]; then echo 'launch as root for runuser'; exit 2; fi",
        "mkdir -p " + quote(args.output_root),
        "pids=()",
    ]
    for worker in workers:
        log = args.output_root / ("worker%02d.log" % worker["worker_index"])
        lines.extend(["(", "worker_status=0"])
        for task in worker["tasks"]:
            output = quote(task["output_dir"])
            command = [
                quote(args.python), quote(collector),
                "--output_dir", output,
                "--routes", quote(args.routes),
                "--routes_subset", quote(task["route_id"]),
                "--seed", str(task["seed"]),
                "--repetitions", "1", "--no_resume",
                "--base_port", str(worker["base_port"]),
                "--base_tm_port", str(worker["base_tm_port"]),
                "--rift_hd_map_dir", quote(args.rift_hd_map_dir),
                "--rift_speed_limits_dir", quote(args.rift_speed_limits_dir),
                "--sample_interval_s", str(args.sample_interval_s),
                "--actor_radius", str(args.actor_radius),
                "--map_radius", str(args.map_radius),
                "--max_frames_per_episode", str(args.max_frames_per_episode),
                "--min_forward_rift_positive", "0",
                "--forward_min_x", "0.0",
                "--target_fov_margin", "2.0",
                "--hv_sd_fov", "64.0",
            ]
            child = " ; ".join([
                "export ROOT=" + quote(project_root),
                "export CARLA_ROOT=\"$ROOT/carla\"",
                "export CARLA_PYTHONAPI=\"$CARLA_ROOT/PythonAPI/carla\"",
                "export CARLA_EGG=\"$(find \"$CARLA_PYTHONAPI/dist\" -maxdepth 1 -name 'carla-*.egg' | head -n 1)\"",
                "export PYTHONPATH=\"$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}\"",
                "export CUDA_VISIBLE_DEVICES=" + str(worker["gpu"]),
                # ``--routes`` is deliberately relative to RIFT.  The
                # recovery launcher runs from an arbitrary parent shell, so
                # make its working directory explicit before starting run.py.
                "cd \"$ROOT/RIFT\"",
                "exec " + " ".join(command),
            ])
            route_id, round_index = task_key(task)
            lines.extend([
                "mkdir -p " + output,
                "chown " + quote(args.run_as_user + ":" + args.run_as_user) + " " + output,
                "if timeout --signal=TERM --kill-after=60s " + str(args.task_timeout_seconds)
                + " runuser -u " + quote(args.run_as_user) + " -- /bin/bash -lc " + quote(child) + "; then",
                "  echo 'completed route " + route_id + " round " + str(round_index) + "'",
                "else",
                "  task_rc=$?",
                "  echo 'failed route " + route_id + " round " + str(round_index) + " rc='\"$task_rc\" >&2",
                "  worker_status=1",
                "fi",
            ])
        lines.extend(["exit \"$worker_status\"", ") > " + quote(log) + " 2>&1 &", "pids+=(\"$!\")"])
    lines.extend(["status=0", "for pid in \"${pids[@]}\"; do wait \"$pid\" || status=1; done", "exit \"$status\""])
    args.shell_output.parent.mkdir(parents=True, exist_ok=True)
    args.shell_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "shell_output": str(args.shell_output),
        "plan_output": str(args.plan_output),
        "source_total_program_count": len(source_tasks),
        "source_completed_task_count": len(expected & completed),
        "remaining_task_count": len(missing),
        "recovery_worker_count": worker_count,
    }, indent=2))


if __name__ == "__main__":
    main()
