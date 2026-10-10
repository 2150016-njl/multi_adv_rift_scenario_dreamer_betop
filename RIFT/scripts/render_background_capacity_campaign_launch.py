#!/usr/bin/env python
"""Render a port-safe launch script for a background capacity campaign."""
import argparse
import json
import shlex
from pathlib import Path


def q(value): return shlex.quote(str(value))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--routes", required=True); p.add_argument("--rift_root", type=Path, required=True); p.add_argument("--python", required=True)
    p.add_argument("--rift_hd_map_dir", type=Path, required=True); p.add_argument("--rift_speed_limits_dir", type=Path, required=True)
    p.add_argument("--shell_output", type=Path, required=True); p.add_argument("--plan_output", type=Path, required=True)
    p.add_argument("--run_as_user", default="riftuser"); p.add_argument("--gpu_count", type=int, default=4); p.add_argument("--processes_per_gpu", type=int, default=4)
    p.add_argument("--seed_start", type=int, default=10000); p.add_argument("--base_port", type=int, default=36000); p.add_argument("--base_tm_port", type=int, default=56000); p.add_argument("--port_stride", type=int, default=20)
    p.add_argument("--max_frames_per_episode", type=int, default=200); p.add_argument("--sample_interval_s", type=float, default=1.0); p.add_argument("--actor_radius", type=float, default=80.0); p.add_argument("--map_radius", type=float, default=64.0)
    a=p.parse_args()
    if a.shell_output.exists() or a.plan_output.exists(): raise FileExistsError("refusing to overwrite launch artifacts")
    m=json.loads(a.manifest.read_text(encoding="utf-8")); selected=m.get("selected_routes") or {}; workers=m.get("workers") or []
    if m.get("schema_version") != "scenario_dreamer_background_capacity_campaign_v1" or not selected or not workers: raise ValueError("not a background capacity manifest")
    if len(workers) > a.gpu_count*a.processes_per_gpu: raise ValueError("fewer process slots than planned workers")
    tasks=[]; index=0
    for worker in workers:
        wid=int(worker["worker_index"]); port=a.base_port+wid*a.port_stride; tm=a.base_tm_port+wid*a.port_stride
        if not (1024 <= port <= 65535 and 1024 <= tm <= 65535): raise ValueError("worker port outside valid range")
        gpu=wid % a.gpu_count
        for rid in worker["routes"]:
            for round_index in range(int(selected[rid]["recommended_seed_rounds"])):
                tasks.append({"worker_index":wid,"gpu":gpu,"route_id":rid,"town":selected[rid]["town"],"round_index":round_index,"seed":a.seed_start+index,"base_port":port,"base_tm_port":tm,"output_dir":str(a.output_root/("job%02d_gpu%d_route%s_round%03d_seed%d"%(wid,gpu,rid,round_index,a.seed_start+index)))})
                index += 1
    plan={"schema_version":"scenario_dreamer_background_capacity_launch_v1","manifest":str(a.manifest),"output_root":str(a.output_root),"job_count":len(workers),"total_program_count":len(tasks),"tasks":tasks,
          "invariants":{"ports_unique_across_concurrent_workers":True,"ports_reused_only_sequentially_within_worker":True,"all_raw_traffic":True,"raw_state_postprocessing":"none"}}
    a.plan_output.parent.mkdir(parents=True,exist_ok=True); a.plan_output.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    root=a.rift_root.parent; collector=a.rift_root/"scripts"/"collect_high_value_init_scenes.py"; grouped={int(w["worker_index"]):[] for w in workers}
    for task in tasks: grouped[task["worker_index"]].append(task)
    lines=["#!/usr/bin/env bash","export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin","if [ \"$(id -u)\" -ne 0 ]; then echo 'launch as root for runuser'; exit 2; fi","mkdir -p "+q(a.output_root),"pids=()"]
    for wid, rows in grouped.items():
        lines.append("(")
        for task in rows:
            out=q(task["output_dir"]); lines.extend(["mkdir -p "+out,"chown "+q(a.run_as_user+":"+a.run_as_user)+" "+out])
            cmd=[q(a.python),q(collector),"--output_dir",out,"--routes",q(a.routes),"--routes_subset",task["route_id"],"--seed",str(task["seed"]),"--repetitions","1","--no_resume","--base_port",str(task["base_port"]),"--base_tm_port",str(task["base_tm_port"]),"--rift_hd_map_dir",q(a.rift_hd_map_dir),"--rift_speed_limits_dir",q(a.rift_speed_limits_dir),"--sample_interval_s",str(a.sample_interval_s),"--actor_radius",str(a.actor_radius),"--map_radius",str(a.map_radius),"--max_frames_per_episode",str(a.max_frames_per_episode),"--min_forward_rift_positive","0","--forward_min_x","0.0","--target_fov_margin","2.0","--hv_sd_fov","64.0"]
            child=" ; ".join(["export ROOT="+q(root),"export CARLA_ROOT=\"$ROOT/carla\"","export CARLA_PYTHONAPI=\"$CARLA_ROOT/PythonAPI/carla\"",'export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name \'carla-*.egg\' | head -n 1)"',"export PYTHONPATH=\"$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}\"","export CUDA_VISIBLE_DEVICES="+str(task["gpu"]),"exec "+" ".join(cmd)])
            lines.extend(["runuser -u "+q(a.run_as_user)+" -- /bin/bash -lc "+q(child),"echo 'completed route "+task["route_id"]+" round "+str(task["round_index"])+"'"])
        lines.extend([") > "+q(a.output_root/("job%02d.log"%wid))+" 2>&1 &","pids+=(\"$!\")"])
    lines.extend(["status=0","for pid in \"${pids[@]}\"; do wait \"$pid\" || status=1; done","exit \"$status\""])
    a.shell_output.parent.mkdir(parents=True,exist_ok=True); a.shell_output.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"shell_output":str(a.shell_output),"plan_output":str(a.plan_output),"job_count":len(workers),"total_program_count":len(tasks)},indent=2))


if __name__ == "__main__": main()
