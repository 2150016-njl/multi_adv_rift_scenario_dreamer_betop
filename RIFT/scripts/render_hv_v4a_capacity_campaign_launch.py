#!/usr/bin/env python
"""Render (but never start) independent route/seed V4-A campaign tasks."""
import argparse
import json
import shlex
from pathlib import Path


def q(x): return shlex.quote(str(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--routes", required=True); p.add_argument("--rift_root", type=Path, required=True); p.add_argument("--python", required=True)
    p.add_argument("--rift_hd_map_dir", type=Path, required=True); p.add_argument("--rift_speed_limits_dir", type=Path, required=True)
    p.add_argument("--shell_output", type=Path, required=True); p.add_argument("--plan_output", type=Path, required=True)
    p.add_argument("--run_as_user", default="riftuser"); p.add_argument("--gpu_count", type=int, default=4)
    p.add_argument("--processes_per_gpu", type=int, default=2); p.add_argument("--seed_start", type=int, default=1000)
    p.add_argument("--base_port", type=int, default=34000); p.add_argument("--base_tm_port", type=int, default=54000)
    p.add_argument("--port_stride", type=int, default=20); p.add_argument("--max_frames_per_episode", type=int, default=200)
    p.add_argument("--sample_interval_s", type=float, default=1.0); p.add_argument("--actor_radius", type=float, default=80.0); p.add_argument("--map_radius", type=float, default=64.0)
    a = p.parse_args()
    if a.shell_output.exists() or a.plan_output.exists(): raise FileExistsError("refusing to overwrite launch artifacts")
    m = json.loads(a.manifest.read_text(encoding="utf-8"))
    c = m.get("collection_contract") or {}
    if m.get("schema_version") != "hv_v4a_capacity_campaign_v1" or c.get("num_generate_agents") != 2 or c.get("forward_min_x") != 0.0:
        raise ValueError("not a fixed-pair forward V4-A capacity manifest")
    selected, workers = m.get("selected_routes") or {}, m.get("workers") or []
    if not selected or len(workers) > a.gpu_count * a.processes_per_gpu: raise ValueError("invalid campaign worker plan")
    tasks=[]; index=0
    for worker in workers:
        gpu=int(worker["worker_index"]) % a.gpu_count
        # A worker runs its route/seed tasks sequentially.  Reusing one
        # worker-local port pair prevents a long campaign from overflowing
        # TCP's 65535 upper bound while remaining collision-free in parallel.
        worker_port=a.base_port + int(worker["worker_index"])*a.port_stride
        worker_tm_port=a.base_tm_port + int(worker["worker_index"])*a.port_stride
        if worker_port > 65535 or worker_tm_port > 65535:
            raise ValueError("worker port exceeds 65535; lower base port or stride")
        for rid in worker.get("routes", []):
            row=selected[rid]
            for round_idx in range(int(row["recommended_seed_rounds"])):
                tasks.append({"worker_index": int(worker["worker_index"]), "gpu":gpu, "route_id":rid, "town":row["town"], "round_index":round_idx,
                              "seed":a.seed_start+index, "base_port":worker_port, "base_tm_port":worker_tm_port,
                              "output_dir":str(a.output_root / ("job%02d_gpu%d_route%s_round%02d_seed%d" % (worker["worker_index"],gpu,rid,round_idx,a.seed_start+index)))})
                index+=1
    grouped={int(w["worker_index"]):[] for w in workers}
    for task in tasks: grouped[task["worker_index"]].append(task)
    plan={"schema_version":"hv_v4a_capacity_campaign_launch_v1", "manifest":str(a.manifest), "output_root":str(a.output_root), "job_count":len(grouped), "total_program_count":len(tasks), "tasks":tasks,
          "invariants":{"ports_unique_across_concurrent_workers":True,"ports_reused_only_sequentially_within_worker":True,"unique_output_dirs":True,"forward_fixed_pair":True,"raw_state_postprocessing":"none"}}
    a.plan_output.parent.mkdir(parents=True, exist_ok=True); a.plan_output.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    root=a.rift_root.parent; collector=a.rift_root/"scripts"/"collect_high_value_init_scenes.py"
    lines=["#!/usr/bin/env bash","export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin","if [ \"$(id -u)\" -ne 0 ]; then echo 'launch as root for runuser'; exit 2; fi","mkdir -p "+q(a.output_root),"pids=()"]
    for wid, rows in grouped.items():
        log=a.output_root/("job%02d.log" % wid); lines.append("(")
        for t in rows:
            out=q(t["output_dir"]); lines.extend(["mkdir -p "+out,"chown "+q(a.run_as_user+":"+a.run_as_user)+" "+out])
            cmd=[q(a.python),q(collector),"--output_dir",out,"--routes",q(a.routes),"--routes_subset",t["route_id"],"--seed",str(t["seed"]),"--repetitions","1","--no_resume","--base_port",str(t["base_port"]),"--base_tm_port",str(t["base_tm_port"]),"--rift_hd_map_dir",q(a.rift_hd_map_dir),"--rift_speed_limits_dir",q(a.rift_speed_limits_dir),"--sample_interval_s",str(a.sample_interval_s),"--actor_radius",str(a.actor_radius),"--map_radius",str(a.map_radius),"--max_frames_per_episode",str(a.max_frames_per_episode),"--min_forward_rift_positive","2","--forward_min_x","0.0","--target_fov_margin","2.0","--hv_sd_fov","64.0"]
            child=" ; ".join(["export ROOT="+q(root),"export CARLA_ROOT=\"$ROOT/carla\"","export CARLA_PYTHONAPI=\"$CARLA_ROOT/PythonAPI/carla\"",'export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name \'carla-*.egg\' | head -n 1)"',"export PYTHONPATH=\"$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}\"","export CUDA_VISIBLE_DEVICES="+str(t["gpu"]),"exec "+" ".join(cmd)])
            lines.extend(["runuser -u "+q(a.run_as_user)+" -- /bin/bash -lc "+q(child),"echo 'completed route "+t["route_id"]+" round "+str(t["round_index"])+"'"])
        lines.extend([") > "+q(log)+" 2>&1 &","pids+=(\"$!\")"])
    lines.extend(["status=0","for pid in \"${pids[@]}\"; do wait \"$pid\" || status=1; done","exit \"$status\""])
    a.shell_output.parent.mkdir(parents=True, exist_ok=True); a.shell_output.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"shell_output":str(a.shell_output),"plan_output":str(a.plan_output),"jobs":len(grouped),"programs":len(tasks)},indent=2))


if __name__ == "__main__": main()
