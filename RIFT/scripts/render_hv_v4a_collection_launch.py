#!/usr/bin/env python
"""Render, but do not start, isolated V4-A collection jobs.

The output shell script has one CARLA/TM port pair and one output directory
per job.  It intentionally uses the ordinary labelled-snapshot collector;
the V4-A builder later applies the forward-pair and diversity contracts.
"""

import argparse
import json
import shlex
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--rift_root", type=Path, required=True)
    parser.add_argument("--python", required=True,
                        help="Absolute RIFT Python executable; passed explicitly into the riftuser login shell.")
    parser.add_argument("--run_as_user", default="riftuser",
                        help="CARLA must not be started as root. The generated launcher requires root only to switch to this user.")
    parser.add_argument("--rift_hd_map_dir", type=Path, required=True)
    parser.add_argument("--rift_speed_limits_dir", type=Path, required=True)
    parser.add_argument("--shell_output", type=Path, required=True)
    parser.add_argument("--plan_output", type=Path, required=True)
    parser.add_argument("--gpu_count", type=int, default=4)
    parser.add_argument("--processes_per_gpu", type=int, default=3)
    parser.add_argument("--seed_rounds", type=int, default=1,
                        help="Sequential programs per concurrent job. Each round receives an independent seed, port pair, and output directory.")
    parser.add_argument("--include_exploration", action="store_true", default=False,
                        help="Also collect manifest exploration routes. They improve route/Town coverage but have no historical V4-A yield guarantee.")
    parser.add_argument("--seed_start", type=int, default=8)
    parser.add_argument("--base_port", type=int, default=32000)
    parser.add_argument("--base_tm_port", type=int, default=52000)
    parser.add_argument("--port_stride", type=int, default=20)
    parser.add_argument("--max_frames_per_episode", type=int, default=200)
    parser.add_argument("--sample_interval_s", type=float, default=1.0)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--map_radius", type=float, default=64.0)
    return parser.parse_args()


def q(value):
    return shlex.quote(str(value))


def main():
    args = parse_args()
    if args.gpu_count < 1 or args.processes_per_gpu < 1 or args.seed_rounds < 1 or args.port_stride < 2:
        raise ValueError("gpu_count, processes_per_gpu, and port_stride must be positive; port_stride must be >= 2")
    if args.shell_output.exists() or args.plan_output.exists():
        raise FileExistsError("refusing to overwrite launch artifacts")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = manifest.get("collection_contract") or {}
    if int(contract.get("num_generate_agents", -1)) != 2 or float(contract.get("fov", -1.0)) != 64.0:
        raise ValueError("manifest is not a fixed-two, 64m V4-A plan")
    if float(contract.get("forward_min_x", -1.0)) != 0.0 or int(contract.get("min_forward_rift_positive", -1)) != 2:
        raise ValueError("manifest does not have the x>=0 forward-pair contract")
    selected = manifest.get("selected_routes") or {}
    if not selected:
        raise ValueError("manifest has no selected routes")
    route_evidence = {
        str(route_id): {"candidate_records": int(row.get("candidate_records", 0)), "source": "historical_candidate"}
        for route_id, row in selected.items()
    }
    if args.include_exploration:
        for town, route_ids in (manifest.get("exploration_route_ids_by_town") or {}).items():
            for route_id in route_ids or []:
                route_evidence.setdefault(str(route_id), {"candidate_records": 0, "source": "exploration_no_historical_candidate", "town": str(town)})
    job_count = args.gpu_count * args.processes_per_gpu
    jobs = [{"job_index": index, "gpu": index % args.gpu_count, "routes": [], "historical_candidate_load": 0}
            for index in range(job_count)]
    for route_id, evidence in sorted(route_evidence.items(), key=lambda item: (-int(item[1].get("candidate_records", 0)), item[0])):
        job = min(jobs, key=lambda item: (item["historical_candidate_load"], item["job_index"]))
        job["routes"].append(str(route_id))
        job["historical_candidate_load"] += int(evidence.get("candidate_records", 0))
    jobs = [job for job in jobs if job["routes"]]
    all_routes = [route for job in jobs for route in job["routes"]]
    if len(all_routes) != len(set(all_routes)) or set(all_routes) != set(route_evidence):
        raise RuntimeError("renderer lost or duplicated selected routes")
    used_ports, used_tm_ports = set(), set()
    for job in jobs:
        index = job["job_index"]
        rounds = []
        for round_index in range(args.seed_rounds):
            program_index = round_index * job_count + index
            port = args.base_port + program_index * args.port_stride
            tm_port = args.base_tm_port + program_index * args.port_stride
            if port in used_ports or tm_port in used_tm_ports:
                raise RuntimeError("port collision")
            used_ports.add(port); used_tm_ports.add(tm_port)
            seed = args.seed_start + program_index
            rounds.append({
                "round_index": round_index, "seed": seed, "base_port": port, "base_tm_port": tm_port,
                "output_dir": str(args.output_root / f"job{index:02d}_gpu{job['gpu']}_round{round_index:02d}_seed{seed}"),
            })
        job.update({
            # Kept for compatibility with v1 readers; all execution uses rounds.
            "seed": rounds[0]["seed"], "base_port": rounds[0]["base_port"], "base_tm_port": rounds[0]["base_tm_port"],
            "routes_subset": ",".join(job["routes"]),
            "output_dir": rounds[0]["output_dir"], "rounds": rounds,
        })
        job["route_sources"] = {route_id: route_evidence[route_id]["source"] for route_id in job["routes"]}
    plan = {
        "schema_version": "hv_v4a_collection_launch_plan_v1",
        "manifest": str(args.manifest), "output_root": str(args.output_root),
        "job_count": len(jobs), "gpu_count": args.gpu_count,
        "processes_per_gpu_requested": args.processes_per_gpu,
        "seed_rounds": args.seed_rounds,
        "total_program_count": sum(len(job["rounds"]) for job in jobs),
        "run_as_user": args.run_as_user,
        "include_exploration": bool(args.include_exploration),
        "invariants": {
            "unique_routes": True, "unique_carla_ports": True, "unique_tm_ports": True,
            "per_job_output_dir": True, "forward_min_x": 0.0,
            "min_forward_rift_positive": 2, "raw_state_postprocessing": "none",
        },
        "jobs": jobs,
        "note": "Jobs are historical-evidence-balanced, not accepted-record guarantees. Do not train until the rebuilt V4-A dataset passes strict audit.",
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    collector = args.rift_root / "scripts" / "collect_high_value_init_scenes.py"
    project_root = args.rift_root.parent
    lines = [
        "#!/usr/bin/env bash",
        "# Generated; run with `bash path/to/launch.sh`; do not source this file.",
        "# CARLA refuses root. This launcher is intentionally executed by root only to start each job as riftuser.",
        "# Do not inherit a broken interactive-shell PATH into launcher control commands.",
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "if [ \"$(id -u)\" -ne 0 ]; then echo 'launch this file from root so it can use runuser'; exit 2; fi",
        "mkdir -p " + q(args.output_root),
        "pids=()",
    ]
    for job in jobs:
        log = args.output_root / f"job{job['job_index']:02d}.log"
        lines.append("(")
        for round_spec in job["rounds"]:
            command = [
                q(args.python), q(collector),
                "--output_dir", q(round_spec["output_dir"]), "--routes", q(args.routes),
                "--routes_subset", q(job["routes_subset"]), "--seed", str(round_spec["seed"]),
                "--repetitions", "1", "--no_resume", "--base_port", str(round_spec["base_port"]),
                "--base_tm_port", str(round_spec["base_tm_port"]), "--rift_hd_map_dir", q(args.rift_hd_map_dir),
                "--rift_speed_limits_dir", q(args.rift_speed_limits_dir),
                "--sample_interval_s", str(args.sample_interval_s), "--actor_radius", str(args.actor_radius),
                "--map_radius", str(args.map_radius), "--max_frames_per_episode", str(args.max_frames_per_episode),
                "--min_forward_rift_positive", "2", "--forward_min_x", "0.0", "--target_fov_margin", "2.0", "--hv_sd_fov", "64.0",
            ]
            child = " ; ".join([
                f"export ROOT={q(project_root)}",
                f"export PY={q(args.python)}",
                "export CARLA_ROOT=\"$ROOT/carla\"",
                "export CARLA_PYTHONAPI=\"$CARLA_ROOT/PythonAPI/carla\"",
                'export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name \'carla-*.egg\' | head -n 1)"',
                "export PYTHONPATH=\"$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}\"",
                f"export CUDA_VISIBLE_DEVICES={job['gpu']}",
                "exec " + " ".join(command),
            ])
            lines.append("mkdir -p " + q(Path(round_spec["output_dir"])))
            lines.append("chown " + q(f"{args.run_as_user}:{args.run_as_user}") + " " + q(Path(round_spec["output_dir"])))
            lines.append("runuser -u " + q(args.run_as_user) + " -- /bin/bash -lc " + q(child))
            lines.append("echo 'completed job " + str(job["job_index"]) + " round " + str(round_spec["round_index"]) + "'")
        lines.append(") > " + q(log) + " 2>&1 &")
        lines.append("pids+=(\"$!\")")
    lines.extend(["status=0", "for pid in \"${pids[@]}\"; do wait \"$pid\" || status=1; done", "exit \"$status\""])
    args.shell_output.parent.mkdir(parents=True, exist_ok=True)
    args.shell_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"shell_output": str(args.shell_output), "plan_output": str(args.plan_output), "job_count": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
