#!/usr/bin/env python
"""Audit background collection manifest quotas and worker route isolation."""
import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output_json", type=Path, required=True); args = p.parse_args()
    if not args.manifest.is_file():
        raise ValueError("manifest must be an existing regular file: %s" % args.manifest)
    data = json.loads(args.manifest.read_text(encoding="utf-8")); errors = []
    schema = data.get("schema_version")
    if schema not in {"scenario_dreamer_background_collection_manifest_v1", "scenario_dreamer_background_collection_manifest_v2"}: errors.append("schema")
    if data.get("task") != "current_front_scene_completion_not_future_prediction" or data.get("collection_filter") != "none_all_raw_traffic_snapshots": errors.append("task_or_filter")
    workers = data.get("workers") or []; seen = []; worker_quota = Counter()
    for worker in workers:
        if worker.get("routes_subset") != ",".join(worker.get("routes") or []): errors.append("routes_subset")
        seen.extend(worker.get("routes") or []); worker_quota.update(worker.get("accepted_record_quota_by_town") or {})
    if len(seen) != len(set(seen)): errors.append("duplicate_route_across_workers")
    if schema == "scenario_dreamer_background_collection_manifest_v1":
        quotas = data.get("accepted_record_quota_by_town") or {}
        if data.get("dominant_town") in quotas: errors.append("dominant_quota")
        if {str(k): int(v) for k,v in worker_quota.items()} != {str(k): int(v) for k,v in quotas.items()}: errors.append("worker_quota")
    else:
        goal, coverage, selected = data.get("collection_goal") or {}, data.get("coverage") or {}, data.get("selected_routes") or {}
        if int(goal.get("target_total_accepted_records", 0)) < 30000: errors.append("accepted_scale_below_tens_of_thousands")
        if int(goal.get("requested_wave_new_accepted_records", 0)) <= 0: errors.append("missing_wave_goal")
        if int(coverage.get("min_routes_per_town", 0)) < 2: errors.append("route_diversity_contract")
        if data.get("unresolved_planning_reasons"): errors.append("unresolved_route_coverage")
        capacity = coverage.get("route_capacity_by_town") or {}
        selected_by_town = Counter(str(row.get("town")) for row in selected.values())
        for town, row in capacity.items():
            required = int((row or {}).get("required_routes_this_wave", 0))
            if selected_by_town[str(town)] < required: errors.append("route_coverage:%s" % town)
        if not capacity: errors.append("missing_capacity_aware_route_contract")
        if data.get("dominant_town") in {row.get("town") for row in selected.values()}: errors.append("dominant_route_selected")
        if not selected or any(int(row.get("recommended_seed_rounds", 0)) < 1 for row in selected.values()): errors.append("missing_route_seed_schedule")
    report = {"passed": not errors, "errors": errors, "schema_version": schema,
              "note": "Passed manifest is a collection plan, not dataset acceptance evidence."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))
    if errors: raise SystemExit(2)


if __name__ == "__main__":
    from collections import Counter
    main()
