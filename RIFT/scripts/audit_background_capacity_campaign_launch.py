#!/usr/bin/env python
"""Audit concurrency, route, and raw-state invariants of a background campaign."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--plan",type=Path,required=True); p.add_argument("--output_json",type=Path,required=True); a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding="utf-8")); tasks=plan.get("tasks") or []; errors=[]
    if plan.get("schema_version")!="scenario_dreamer_background_capacity_launch_v1": errors.append("schema")
    if not tasks: errors.append("no_tasks")
    if len({x.get("output_dir") for x in tasks})!=len(tasks): errors.append("duplicate_output_dir")
    if len({(x.get("route_id"),x.get("seed")) for x in tasks})!=len(tasks): errors.append("route_seed_collision")
    for field in ("base_port","base_tm_port"):
        owners={}
        for task in tasks:
            value=task.get(field); worker=task.get("worker_index")
            if not isinstance(value,int) or not 1024 <= value <= 65535: errors.append("invalid_"+field); break
            if value in owners and owners[value]!=worker: errors.append("port_shared_across_workers:"+field); break
            owners[value]=worker
    inv=plan.get("invariants") or {}
    if not inv.get("all_raw_traffic") or inv.get("raw_state_postprocessing")!="none": errors.append("task_contract")
    manifest=Path(plan.get("manifest", ""))
    if manifest.is_file():
        goal=(json.loads(manifest.read_text(encoding="utf-8")).get("collection_goal") or {})
        if float(goal.get("estimated_shortfall",0.0))>1e-9: errors.append("estimated_collection_shortfall")
    report={"schema_version":"scenario_dreamer_background_capacity_launch_audit_v1","passed":not errors,"errors":errors,"job_count":plan.get("job_count"),"total_program_count":len(tasks)}
    a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
    if errors: raise SystemExit(2)


if __name__=="__main__": main()
