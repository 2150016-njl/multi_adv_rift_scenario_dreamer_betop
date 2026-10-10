#!/usr/bin/env python
"""One-shot, route-covering capacity plan for the directional background task.

This planner deliberately does not rescan raw snapshots or HD maps.  It uses
the builder's accepted-record distribution, covers every non-dominant XML
route at least once, then assigns additional independent seeds subject to a
per-Town fraction ceiling.  The builder remains the only acceptance authority.
"""
import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--existing_diagnostics", type=Path, required=True)
    p.add_argument("--routes_xml", type=Path, required=True)
    p.add_argument("--output_manifest", type=Path, required=True)
    p.add_argument("--target_total_accepted_records", type=int, default=50000)
    p.add_argument("--max_town_fraction", type=float, default=.35)
    p.add_argument("--min_accepted_records_per_town", type=int, default=2000)
    p.add_argument("--min_seed_rounds_per_route", type=int, default=8)
    p.add_argument("--max_seed_rounds_per_route", type=int, default=100)
    p.add_argument("--estimated_accepted_per_route_seed", type=float, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed_start", type=int, default=10000)
    return p.parse_args()


def catalog(path):
    root = ET.parse(path).getroot()
    out = {str(x.get("id")): str(x.get("town")) for x in root.findall(".//route") if x.get("id") and x.get("town")}
    if not out:
        raise ValueError("routes_xml contains no route/town catalog")
    return out


def main():
    a = args_parser()
    if a.output_manifest.exists():
        raise FileExistsError("refusing to overwrite manifest: %s" % a.output_manifest)
    if not a.existing_diagnostics.is_file() or not a.routes_xml.is_file():
        raise ValueError("diagnostics and routes_xml must be files")
    if not (0 < a.max_town_fraction < 1 and 0 < a.estimated_accepted_per_route_seed and a.workers > 0 and
            0 < a.min_seed_rounds_per_route <= a.max_seed_rounds_per_route):
        raise ValueError("invalid capacity/seed arguments")
    diag = json.loads(a.existing_diagnostics.read_text(encoding="utf-8"))
    if diag.get("schema_version") != "scenario_dreamer_background_directional_v1":
        raise ValueError("existing_diagnostics is not a background builder report")
    existing = Counter({str(k): int(v) for k, v in (diag.get("town_counts") or {}).items()})
    accepted = int(diag.get("records_before_split", 0))
    if not existing or accepted <= 0:
        raise ValueError("diagnostics lack accepted Town counts")
    if a.target_total_accepted_records <= accepted:
        raise ValueError("target_total_accepted_records must exceed existing accepted records")
    routes = catalog(a.routes_xml)
    dominant, dominant_count = max(existing.items(), key=lambda x: (x[1], x[0]))
    by_town = defaultdict(list)
    for rid, town in routes.items():
        if town != dominant:
            by_town[town].append(rid)
    if not by_town:
        raise ValueError("no non-dominant XML routes available")
    rounds = {rid: a.min_seed_rounds_per_route for town in by_town for rid in by_town[town]}
    target_new = a.target_total_accepted_records - accepted
    required_tasks = math.ceil(target_new / a.estimated_accepted_per_route_seed)
    if len(rounds) * a.max_seed_rounds_per_route < required_tasks:
        raise ValueError("route/seed capacity below target; raise max_seed_rounds_per_route or lower target")

    def town_tasks(town):
        return sum(rounds[r] for r in by_town[town])

    # Every loop adds one route-seed task.  First satisfy modest Town floors,
    # then waterfill projected accepted counts, while retaining round-robin
    # route selection inside a Town so one route cannot dominate.
    while sum(rounds.values()) < required_tasks:
        total_after = accepted + (sum(rounds.values()) + 1) * a.estimated_accepted_per_route_seed
        candidates = []
        for town, ids in by_town.items():
            projected = existing[town] + (town_tasks(town) + 1) * a.estimated_accepted_per_route_seed
            if projected <= a.max_town_fraction * total_after + 1e-9 and any(rounds[r] < a.max_seed_rounds_per_route for r in ids):
                below_floor = projected < a.min_accepted_records_per_town
                candidates.append((0 if below_floor else 1, projected, town))
        if not candidates:
            raise RuntimeError("no Town can receive another seed without violating max_town_fraction or route cap")
        town = min(candidates)[2]
        ids = [r for r in by_town[town] if rounds[r] < a.max_seed_rounds_per_route]
        rid = min(ids, key=lambda r: (rounds[r], hashlib.sha256((str(a.seed_start) + ":" + r).encode()).hexdigest()))
        rounds[rid] += 1

    projected = {town: existing[town] + town_tasks(town) * a.estimated_accepted_per_route_seed for town in set(routes.values())}
    estimated_new = sum(rounds.values()) * a.estimated_accepted_per_route_seed
    selected = {rid: {"town": town, "route_selection_source": "all_non_dominant_xml_routes_balanced_capacity",
                      "recommended_seed_rounds": rounds[rid], "estimated_accepted_per_seed": a.estimated_accepted_per_route_seed,
                      "estimated_accepted_records": rounds[rid] * a.estimated_accepted_per_route_seed}
                for town, ids in by_town.items() for rid in sorted(ids)}
    workers = [{"worker_index": i, "seed": a.seed_start + i, "routes": [], "estimated_accepted_load": 0.0} for i in range(a.workers)]
    for rid, row in sorted(selected.items(), key=lambda kv: (-kv[1]["estimated_accepted_records"], kv[0])):
        worker = min(workers, key=lambda x: (x["estimated_accepted_load"], x["worker_index"]))
        worker["routes"].append(rid); worker["estimated_accepted_load"] += row["estimated_accepted_records"]
    for worker in workers:
        worker["routes_subset"] = ",".join(worker["routes"])
        worker["towns"] = sorted({selected[r]["town"] for r in worker["routes"]})
    manifest = {
        "schema_version": "scenario_dreamer_background_capacity_campaign_v1",
        "task": "current_front_scene_completion_not_future_prediction",
        "collection_filter": "none_all_raw_traffic_snapshots", "raw_state_postprocessing": "none",
        "planning_mode": "all_non_dominant_xml_routes_balanced_capacity_no_rescan",
        "dominant_town": dominant, "dominant_town_policy": "reuse_existing_only_no_new_collection",
        "existing_accepted_records": accepted, "existing_town_counts": dict(sorted(existing.items())),
        "collection_goal": {"target_total_accepted_records": a.target_total_accepted_records,
            "required_new_accepted_records": target_new, "estimated_new_accepted_records": estimated_new,
            "estimated_shortfall": max(0.0, target_new-estimated_new), "estimated_accepted_per_route_seed": a.estimated_accepted_per_route_seed},
        "coverage": {"route_count": len(selected), "all_non_dominant_xml_routes": True,
            "min_seed_rounds_per_route": a.min_seed_rounds_per_route, "max_seed_rounds_per_route": a.max_seed_rounds_per_route,
            "projected_accepted_town_counts": dict(sorted(projected.items()))},
        "selected_routes": selected, "workers": workers,
        "training_gate": "Merge collection and require the strict background builder to pass scale, Town/topology, split and raw-state audits before deployment.",
    }
    a.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    a.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
