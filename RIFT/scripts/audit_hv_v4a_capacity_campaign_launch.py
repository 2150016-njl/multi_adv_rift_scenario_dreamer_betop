#!/usr/bin/env python
"""Structural audit for a rendered quota-driven V4-A capacity campaign."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--plan",type=Path,required=True); p.add_argument("--output_json",type=Path,required=True); a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding="utf-8")); tasks=plan.get("tasks") or []; errors=[]
    if plan.get("schema_version")!="hv_v4a_capacity_campaign_launch_v1": errors.append("schema")
    if not tasks: errors.append("no_tasks")
    # Output directories are never shared.  Ports intentionally may repeat
    # only within one sequential worker; sharing a port between workers would
    # make concurrent CARLA processes collide.
    values=[t.get("output_dir") for t in tasks]
    if len(values)!=len(set(values)): errors.append("duplicate_output_dir")
    for field in ("base_port", "base_tm_port"):
        owners={}
        for task in tasks:
            value=task.get(field); worker=task.get("worker_index")
            if value in owners and owners[value] != worker:
                errors.append("port_shared_across_workers:"+field); break
            owners[value]=worker
    keys=[(t.get("route_id"),t.get("round_index")) for t in tasks]
    if len(keys)!=len(set(keys)): errors.append("duplicate_route_round")
    for t in tasks:
        if not (1024 <= int(t.get("base_port",0)) <= 65535 and 1024 <= int(t.get("base_tm_port",0)) <= 65535): errors.append("invalid_port"); break
        if not t.get("route_id") or not t.get("town") or int(t.get("seed",-1))<0: errors.append("missing_task_provenance"); break
    # Refuse to label a rendered launch as passed when the upstream capacity
    # plan already says its seed cap cannot meet the requested wave volume.
    manifest_path=Path(plan.get("manifest", ""))
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        goal=manifest.get("collection_goal") or {}
        if float(goal.get("estimated_wave_shortfall", 0.0)) > 1e-9:
            errors.append("estimated_wave_shortfall")
    result={"schema_version":"hv_v4a_capacity_campaign_launch_audit_v1","passed":not errors,"errors":errors,"job_count":plan.get("job_count"),"total_program_count":len(tasks),"town_task_counts":{}}
    for t in tasks: result["town_task_counts"][str(t["town"])]=result["town_task_counts"].get(str(t["town"]),0)+1
    a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
    if errors: raise SystemExit(2)

if __name__=="__main__": main()
