#!/usr/bin/env python
"""Render isolated raw-traffic collection jobs for the background-BV task.

Unlike V4-A, this deliberately uses ``min_forward_rift_positive=0``.  The
background builder ignores RIFT labels and learns rear-condition to current
front scene completion from all ordinary traffic, not from HV-positive scenes.
"""

import argparse
import json
import shlex
from pathlib import Path


def q(value):
    return shlex.quote(str(value))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--routes", required=True)
    p.add_argument("--rift_root", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--run_as_user", default="riftuser")
    p.add_argument("--rift_hd_map_dir", type=Path, required=True)
    p.add_argument("--rift_speed_limits_dir", type=Path, required=True)
    p.add_argument("--shell_output", type=Path, required=True)
    p.add_argument("--plan_output", type=Path, required=True)
    p.add_argument("--gpu_count", type=int, default=4)
    p.add_argument("--processes_per_gpu", type=int, default=3)
    p.add_argument("--seed_rounds", type=int, default=1)
    p.add_argument("--seed_start", type=int, default=20)
    p.add_argument("--base_port", type=int, default=34000)
    p.add_argument("--base_tm_port", type=int, default=54000)
    p.add_argument("--port_stride", type=int, default=20)
    p.add_argument("--max_frames_per_episode", type=int, default=200)
    p.add_argument("--sample_interval_s", type=float, default=1.0)
    p.add_argument("--actor_radius", type=float, default=80.0)
    p.add_argument("--map_radius", type=float, default=64.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu_count < 1 or args.processes_per_gpu < 1 or args.seed_rounds < 1 or args.port_stride < 2:
        raise ValueError("invalid process/port configuration")
    if args.shell_output.exists() or args.plan_output.exists():
        raise FileExistsError("refusing to overwrite launch artifacts")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version")
    if schema not in {"scenario_dreamer_background_collection_manifest_v1",
                      "scenario_dreamer_background_collection_manifest_v2"}:
        raise ValueError("not a background collection manifest")
    if manifest.get("task") != "current_front_scene_completion_not_future_prediction":
        raise ValueError("background manifest has the wrong task")
    if manifest.get("collection_filter") != "none_all_raw_traffic_snapshots":
        raise ValueError("background collection must retain all raw traffic snapshots")
    selected = manifest.get("selected_routes") or {}
    if not selected:
        raise ValueError("manifest has no selected background routes")
    if schema.endswith("_v2"):
        # Planner v2 assigns route diversity and independent seed schedules.
        # Keep each planned worker sequential so one CARLA/TM pair is never
        # reused concurrently.  More requested process slots are harmless;
        # fewer would silently discard a coverage plan, so reject them.
        planned = manifest.get("workers") or []
        if len(planned) > args.gpu_count * args.processes_per_gpu:
            raise ValueError("requested process slots are fewer than V2 planned workers")
        jobs = [{"job_index": int(row["worker_index"]), "gpu": int(row["worker_index"]) % args.gpu_count,
                 "routes": list(row.get("routes") or []), "historical_raw_snapshot_load": 0,
                 "manifest_v2_route_schedule": True} for row in planned if row.get("routes")]
    else:
        jobs = [{"job_index": index, "gpu": index % args.gpu_count, "routes": [], "historical_raw_snapshot_load": 0}
                for index in range(args.gpu_count * args.processes_per_gpu)]
        for route_id, evidence in sorted(selected.items(), key=lambda item: (-int(item[1].get("historical_raw_snapshot_count", 0)), item[0])):
            job = min(jobs, key=lambda row: (row["historical_raw_snapshot_load"], row["job_index"]))
            job["routes"].append(str(route_id))
            job["historical_raw_snapshot_load"] += int(evidence.get("historical_raw_snapshot_count", 0))
        jobs = [job for job in jobs if job["routes"]]
    routes = [route for job in jobs for route in job["routes"]]
    if len(routes) != len(set(routes)) or set(routes) != set(selected):
        raise RuntimeError("route partition is not exact")
    job_count = args.gpu_count * args.processes_per_gpu
    task_index = 0
    for job in jobs:
        index = job["job_index"]
        if job.get("manifest_v2_route_schedule"):
            rounds = []
            for route_id in job["routes"]:
                route_rounds = int(selected[route_id]["recommended_seed_rounds"])
                for route_round in range(route_rounds):
                    rounds.append({
                        "round_index": len(rounds), "route_round_index": route_round,
                        "route_id": route_id, "routes_subset": route_id,
                        "seed": args.seed_start + task_index,
                        "base_port": args.base_port + task_index * args.port_stride,
                        "base_tm_port": args.base_tm_port + task_index * args.port_stride,
                        "output_dir": str(args.output_root / f"job{index:02d}_gpu{job['gpu']}_route{route_id}_round{route_round:02d}_seed{args.seed_start + task_index}"),
                    })
                    task_index += 1
        else:
            rounds = [{
                "round_index": round_index,
                "seed": args.seed_start + round_index * job_count + index,
                "base_port": args.base_port + (round_index * job_count + index) * args.port_stride,
                "base_tm_port": args.base_tm_port + (round_index * job_count + index) * args.port_stride,
                "output_dir": str(args.output_root / f"job{index:02d}_gpu{job['gpu']}_round{round_index:02d}_seed{args.seed_start + round_index * job_count + index}"),
            } for round_index in range(args.seed_rounds)]
        job.update({
            "seed": rounds[0]["seed"], "base_port": rounds[0]["base_port"], "base_tm_port": rounds[0]["base_tm_port"],
            "routes_subset": ",".join(job["routes"]),
            "output_dir": rounds[0]["output_dir"], "rounds": rounds,
        })
    ports = [round_spec["base_port"] for job in jobs for round_spec in job["rounds"]] + [round_spec["base_tm_port"] for job in jobs for round_spec in job["rounds"]]
    if len(ports) != len(set(ports)):
        raise RuntimeError("CARLA/TM port collision")
    if not ports or min(ports) < 1024 or max(ports) > 65535:
        raise ValueError("planned CARLA/TM port is outside the valid user-port range; reduce one wave or use a smaller port stride")
    plan = {
        "schema_version": "scenario_dreamer_background_collection_launch_plan_v1",
        "manifest": str(args.manifest), "output_root": str(args.output_root), "run_as_user": args.run_as_user,
        "job_count": len(jobs), "gpu_count": args.gpu_count, "processes_per_gpu_requested": args.processes_per_gpu,
        "seed_rounds": args.seed_rounds, "total_program_count": sum(len(job["rounds"]) for job in jobs),
        "invariants": {"unique_routes": True, "unique_carla_tm_ports": True, "per_job_output_dir": True,
                       "min_forward_rift_positive": 0, "raw_state_postprocessing": "none",
                       "task": "current_front_scene_completion_not_future_prediction"},
        "jobs": jobs,
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    root, collector = args.rift_root.parent, args.rift_root / "scripts" / "collect_high_value_init_scenes.py"
    lines = ["#!/usr/bin/env bash", "# Run with bash; do not source. CARLA is launched as riftuser, never root.",
             "# Do not inherit a broken interactive-shell PATH into launcher control commands.",
             "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
             "if [ \"$(id -u)\" -ne 0 ]; then echo 'launch this file from root so it can use runuser'; exit 2; fi",
             "mkdir -p " + q(args.output_root), "pids=()"]
    for job in jobs:
        log = q(args.output_root / f"job{job['job_index']:02d}.log")
        lines.append("(")
        for round_spec in job["rounds"]:
            command = [q(args.python), q(collector), "--output_dir", q(round_spec["output_dir"]), "--routes", q(args.routes),
                       "--routes_subset", q(round_spec.get("routes_subset", job["routes_subset"])), "--seed", str(round_spec["seed"]), "--repetitions", "1", "--no_resume",
                       "--base_port", str(round_spec["base_port"]), "--base_tm_port", str(round_spec["base_tm_port"]),
                       "--rift_hd_map_dir", q(args.rift_hd_map_dir), "--rift_speed_limits_dir", q(args.rift_speed_limits_dir),
                       "--sample_interval_s", str(args.sample_interval_s), "--actor_radius", str(args.actor_radius), "--map_radius", str(args.map_radius),
                       "--max_frames_per_episode", str(args.max_frames_per_episode), "--min_forward_rift_positive", "0",
                       "--forward_min_x", "0.0", "--target_fov_margin", "2.0", "--hv_sd_fov", "64.0"]
            child = " ; ".join([f"export ROOT={q(root)}", f"export PY={q(args.python)}", "export CARLA_ROOT=\"$ROOT/carla\"",
                                 "export CARLA_PYTHONAPI=\"$CARLA_ROOT/PythonAPI/carla\"",
                                 'export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name \'carla-*.egg\' | head -n 1)"',
                                 "export PYTHONPATH=\"$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}\"",
                                 f"export CUDA_VISIBLE_DEVICES={job['gpu']}", "exec " + " ".join(command)])
            output = q(Path(round_spec["output_dir"]))
            lines.extend(["mkdir -p " + output, "chown " + q(f"{args.run_as_user}:{args.run_as_user}") + " " + output,
                          "runuser -u " + q(args.run_as_user) + " -- /bin/bash -lc " + q(child),
                          "echo 'completed job " + str(job["job_index"]) + " round " + str(round_spec["round_index"]) + "'"])
        lines.extend([") > " + log + " 2>&1 &", "pids+=(\"$!\")"])
    lines.extend(["status=0", "for pid in \"${pids[@]}\"; do wait \"$pid\" || status=1; done", "exit \"$status\""])
    args.shell_output.parent.mkdir(parents=True, exist_ok=True)
    args.shell_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"shell_output": str(args.shell_output), "plan_output": str(args.plan_output), "job_count": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
