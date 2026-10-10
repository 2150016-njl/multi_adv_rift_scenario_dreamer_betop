#!/usr/bin/env python
"""Audit launch-plan isolation before expensive multi-CARLA collection."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    errors, jobs = [], plan.get("jobs") or []
    if plan.get("schema_version") != "hv_v4a_collection_launch_plan_v1":
        errors.append("schema")
    routes = [route for job in jobs for route in (job.get("routes") or [])]
    rounds = [round_spec for job in jobs for round_spec in (job.get("rounds") or [{
        "base_port": job.get("base_port"), "base_tm_port": job.get("base_tm_port"), "output_dir": job.get("output_dir"), "seed": job.get("seed"),
    }])]
    ports = [round_spec.get("base_port") for round_spec in rounds]
    tm_ports = [round_spec.get("base_tm_port") for round_spec in rounds]
    dirs = [round_spec.get("output_dir") for round_spec in rounds]
    if not jobs:
        errors.append("no_jobs")
    if len(routes) != len(set(routes)):
        errors.append("duplicate_route")
    if len(ports) != len(set(ports)) or len(tm_ports) != len(set(tm_ports)):
        errors.append("port_collision")
    if len(dirs) != len(set(dirs)):
        errors.append("output_dir_collision")
    for job in jobs:
        if not job.get("routes_subset") or job.get("routes_subset") != ",".join(job.get("routes") or []):
            errors.append(f"routes_subset:{job.get('job_index')}")
        if int(job.get("seed", -1)) < 0 or int(job.get("gpu", -1)) < 0:
            errors.append(f"seed_or_gpu:{job.get('job_index')}")
        if set(job.get("route_sources") or {}) != set(job.get("routes") or []):
            errors.append(f"route_source_provenance:{job.get('job_index')}")
        if not job.get("rounds"):
            errors.append(f"missing_rounds:{job.get('job_index')}")
    if int(plan.get("total_program_count", -1)) != len(rounds):
        errors.append("total_program_count")
    invariants = plan.get("invariants") or {}
    if float(invariants.get("forward_min_x", -1.0)) != 0.0 or int(invariants.get("min_forward_rift_positive", -1)) != 2:
        errors.append("forward_pair_contract")
    if invariants.get("raw_state_postprocessing") != "none":
        errors.append("raw_state_postprocessing")
    if not str(plan.get("run_as_user") or ""):
        errors.append("missing_run_as_user")
    report = {"schema_version": "hv_v4a_collection_launch_audit_v1", "passed": not errors, "errors": errors,
              "job_count": len(jobs), "total_program_count": len(rounds),
              "gpu_counts": {str(gpu): sum(int(job.get("gpu", -1)) == gpu for job in jobs) for gpu in sorted({job.get("gpu") for job in jobs})}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
