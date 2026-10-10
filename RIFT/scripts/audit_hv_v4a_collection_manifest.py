#!/usr/bin/env python
"""Strict structural audit for a V4-A targeted collection manifest."""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes_xml", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def catalog(path):
    root = ET.parse(path).getroot()
    return {str(node.get("id")): str(node.get("town")) for node in root.findall(".//route")
            if node.get("id") and node.get("town")}


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    routes = catalog(args.routes_xml)
    errors = []
    if manifest.get("schema_version") != "hv_v4a_targeted_collection_manifest_v1":
        errors.append("schema_version")
    contract = manifest.get("collection_contract") or {}
    if float(contract.get("fov", -1.0)) != 64.0:
        errors.append("fov_not_64")
    if int(contract.get("num_generate_agents", -1)) != 2:
        errors.append("not_fixed_two_targets")
    if float(contract.get("forward_min_x", -1.0)) != 0.0:
        errors.append("forward_min_x_not_zero")
    if int(contract.get("min_forward_rift_positive", -1)) != 2:
        errors.append("forward_positive_threshold")
    topology_limit = float(contract.get("max_topology_fraction_final_dataset_gate", 0.75))
    if not 0.0 < topology_limit <= 1.0:
        errors.append("max_topology_fraction_final_dataset_gate")
    if contract.get("raw_state_postprocessing") != "none":
        errors.append("raw_state_postprocessing")

    balance = manifest.get("required_balance") or {}
    existing = manifest.get("existing_distribution") or {}
    dominant = existing.get("dominant_town")
    if balance.get("dominant_town_policy") != "no_new_generic_collection":
        errors.append("dominant_town_policy")
    quotas = balance.get("accepted_record_quota_by_town") or {}
    required = int(balance.get("required_new_non_dominant_accepted_records", -1))
    quota_total = sum(int(value) for value in quotas.values())
    # A scale campaign may deliberately collect more than the minimum needed
    # merely to reduce the current dominant-Town fraction.  Equality was only
    # valid for the old balance-only planner and made every >=30k campaign
    # fail its audit despite a sound quota plan.
    if quota_total < required:
        errors.append("town_quota_below_balance_minimum")
    if dominant in quotas:
        errors.append("dominant_town_has_quota")

    feasibility = manifest.get("historical_one_pass_feasibility")
    # Older manifests predate the capacity forecast.  They remain structurally
    # auditable but must be regenerated before using the new forecast to size
    # a collection campaign.
    forecast_warning = None
    if feasibility is None:
        feasibility, shortfall = {}, {}
        forecast_warning = "manifest_predates_one_pass_feasibility; regenerate to obtain collection-capacity diagnostics"
    else:
        shortfall = feasibility.get("quota_shortfall_against_one_historical_pass_by_town") or {}
        if set(shortfall) != set(quotas):
            errors.append("one_pass_shortfall_town_keys")
        for town, value in shortfall.items():
            if int(value) < 0:
                errors.append(f"negative_one_pass_shortfall:{town}")

    selected = manifest.get("selected_routes") or {}
    workers = manifest.get("workers") or []
    worker_route_ids = []
    worker_quotas = {}
    for worker in workers:
        route_ids = list(worker.get("routes") or [])
        if worker.get("routes_subset") != ",".join(route_ids):
            errors.append(f"worker_routes_subset:{worker.get('worker_index')}")
        worker_route_ids.extend(route_ids)
        for town, value in (worker.get("accepted_record_quota_by_town") or {}).items():
            worker_quotas[town] = int(worker_quotas.get(town, 0)) + int(value)
    if {str(key): int(value) for key, value in worker_quotas.items()} != {
            str(key): int(value) for key, value in quotas.items()}:
        errors.append("worker_town_quota_sum")
    if len(worker_route_ids) != len(set(worker_route_ids)):
        errors.append("duplicate_route_across_workers")
    if set(worker_route_ids) != set(selected):
        errors.append("worker_selected_route_mismatch")
    for route_id, evidence in selected.items():
        if route_id not in routes:
            errors.append(f"unknown_route:{route_id}")
            continue
        if routes[route_id] != evidence.get("town"):
            errors.append(f"route_town_mismatch:{route_id}")
        if routes[route_id] == dominant:
            errors.append(f"dominant_town_route_selected:{route_id}")
        if int(evidence.get("candidate_records", 0)) <= 0:
            errors.append(f"route_without_candidate_evidence:{route_id}")
    target_total = int(balance.get("minimum_total_records_if_dominant_unchanged", 0))
    dominant_count = int(existing.get("dominant_town_records", 0))
    maximum = float(existing.get("max_town_fraction", 0.0))
    if target_total <= 0 or maximum <= 0 or dominant_count / target_total > maximum + 1e-12:
        errors.append("town_fraction_math")
    forecast = balance.get("forecast_town_counts_if_quotas_met") or {}
    forecast_total = int(sum(int(value) for value in forecast.values()))
    expected_forecast_total = int(sum(int(value) for value in (existing.get("town_counts") or {}).values())) + quota_total
    if forecast_total != expected_forecast_total:
        errors.append("forecast_total")
    if forecast_total < target_total:
        errors.append("forecast_below_balance_minimum")
    for town, count in forecast.items():
        if forecast_total <= 0 or int(count) / forecast_total > maximum + 1e-12:
            errors.append(f"forecast_town_fraction:{town}")

    scale = manifest.get("scale_collection_goal") or {}
    requested_total = int(scale.get("target_total_accepted_records", 0))
    if requested_total > 0:
        if forecast_total < requested_total:
            errors.append("forecast_below_scale_target")
        floor = int(scale.get("minimum_accepted_records_per_town", 0))
        if floor > 0:
            for town, count in forecast.items():
                if town != dominant and int(count) < floor:
                    errors.append(f"forecast_town_below_floor:{town}")

    result = {
        "schema_version": "hv_v4a_targeted_collection_manifest_audit_v1",
        "passed": not errors,
        "errors": errors,
        "selected_route_count": len(selected),
        "worker_count": len(workers),
        "unresolved_planning_reasons": manifest.get("unresolved_planning_reasons") or [],
        "one_historical_pass_meets_all_town_quotas": feasibility.get("one_historical_pass_meets_all_town_quotas"),
        "one_pass_quota_shortfall_by_town": shortfall,
        "quota_total": quota_total,
        "forecast_total": forecast_total,
        "forecast_warning": forecast_warning,
        "note": "A passed manifest is safe to collect from; it is not proof that the future dataset will pass V4-A diversity gates.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
