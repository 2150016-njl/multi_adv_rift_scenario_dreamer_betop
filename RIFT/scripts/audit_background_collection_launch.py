#!/usr/bin/env python
"""Audit isolated background collection launch plans."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--plan", type=Path, required=True); p.add_argument("--output_json", type=Path, required=True); args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8")); jobs, errors = plan.get("jobs") or [], []
    if plan.get("schema_version") != "scenario_dreamer_background_collection_launch_plan_v1": errors.append("schema")
    routes = [route for job in jobs for route in (job.get("routes") or [])]
    rounds = [round_spec for job in jobs for round_spec in (job.get("rounds") or [])]
    ports = [round_spec.get("base_port") for round_spec in rounds] + [round_spec.get("base_tm_port") for round_spec in rounds]
    if not jobs: errors.append("no_jobs")
    if len(routes) != len(set(routes)): errors.append("duplicate_routes")
    if len(ports) != len(set(ports)): errors.append("port_collision")
    if any(not isinstance(port, int) or port < 1024 or port > 65535 for port in ports): errors.append("port_out_of_range")
    if not rounds or int(plan.get("total_program_count", -1)) != len(rounds): errors.append("round_contract")
    if len({round_spec.get("output_dir") for round_spec in rounds}) != len(rounds): errors.append("output_collision")
    route_seed_pairs = [(round_spec.get("route_id"), round_spec.get("seed")) for round_spec in rounds if round_spec.get("route_id")]
    if len(route_seed_pairs) != len(set(route_seed_pairs)): errors.append("route_seed_collision")
    if any(round_spec.get("route_id") and round_spec.get("routes_subset") != round_spec.get("route_id") for round_spec in rounds): errors.append("v2_route_task_contract")
    if any(job.get("routes_subset") != ",".join(job.get("routes") or []) for job in jobs): errors.append("routes_subset")
    invariants = plan.get("invariants") or {}
    if invariants.get("task") != "current_front_scene_completion_not_future_prediction" or int(invariants.get("min_forward_rift_positive", -1)) != 0: errors.append("background_task_or_filter")
    if invariants.get("raw_state_postprocessing") != "none" or not plan.get("run_as_user"): errors.append("raw_or_user_contract")
    report = {"schema_version": "scenario_dreamer_background_collection_launch_audit_v1", "passed": not errors, "errors": errors, "job_count": len(jobs), "total_program_count": len(rounds)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))
    if errors: raise SystemExit(2)


if __name__ == "__main__": main()
