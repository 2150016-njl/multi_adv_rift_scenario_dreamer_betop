#!/usr/bin/env python
"""Plan coverage-first raw CARLA background-traffic collection.

Town-fraction balance is only a safety gate.  The target is a requested final
number of *accepted* directional background records, with route and Town
coverage; raw snapshots remain an uncertain upstream quantity.
"""
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "scenario_dreamer_background_collection_manifest_v2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--existing_diagnostics", type=Path, required=True)
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--routes_xml", type=Path, required=True)
    p.add_argument("--output_manifest", type=Path, required=True)
    p.add_argument("--max_town_fraction", type=float, default=.35)
    p.add_argument("--target_total_accepted_records", type=int, default=30000)
    p.add_argument("--collection_wave_accepted_records", type=int, default=4000)
    p.add_argument("--min_accepted_records_per_town", type=int, default=2000)
    p.add_argument("--min_routes_per_town", type=int, default=8)
    p.add_argument("--max_routes_per_town", type=int, default=16)
    p.add_argument("--min_seed_rounds_per_route", type=int, default=3)
    p.add_argument("--max_seed_rounds_per_route", type=int, default=8,
                   help="Per-wave cap. Re-plan from measured builder diagnostics instead of repeating one route indefinitely.")
    p.add_argument("--historical_seed_count", type=int, default=8)
    p.add_argument("--estimated_acceptance_rate", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed_start", type=int, default=8)
    return p.parse_args()


def route_catalog(path):
    root = ET.parse(path).getroot()
    result = {str(n.get("id")): str(n.get("town")) for n in root.findall(".//route")
              if n.get("id") and n.get("town")}
    if not result:
        raise ValueError("route XML has no id/town catalog")
    return result


def route_id(scene):
    nested = scene.get("scenario") if isinstance(scene.get("scenario"), dict) else {}
    for value in (scene.get("route_id"), nested.get("route_id"), nested.get("name"),
                  scene.get("scene_id"), scene.get("episode_key")):
        text = str(value or "")
        found = re.search(r"RouteScenario_(\d+)", text)
        if found:
            return found.group(1)
        if text.isdigit():
            return text
    return None


def waterfill(existing, towns, amount):
    allocation = Counter()
    for _ in range(max(0, int(amount))):
        town = min(towns, key=lambda key: (int(existing.get(key, 0)) + allocation[key], key))
        allocation[town] += 1
    return allocation


def choose_routes(evidence, catalog, towns_to_collect, args):
    by_town = defaultdict(list)
    for rid, count in evidence.items():
        by_town[catalog[rid]].append((rid, count))
    available = defaultdict(list)
    for rid, town in catalog.items():
        if town in towns_to_collect:
            available[town].append(rid)
    selected, unresolved, capacity = {}, [], {}
    for town in towns_to_collect:
        # Historical evidence ranks routes, but routes absent from the old
        # snapshot archive are retained as explicit exploration routes.  A
        # benchmark Town with only four XML routes cannot satisfy an absolute
        # eight-route floor; covering all four is the strongest honest claim.
        all_routes = sorted(available[town])
        required = min(args.min_routes_per_town, len(all_routes))
        capacity[town] = {"available_benchmark_routes": len(all_routes), "required_routes_this_wave": required}
        ranked = [rid for rid, _ in sorted(by_town[town], key=lambda item: (-item[1], item[0]))]
        ranked.extend(rid for rid in all_routes if rid not in set(ranked))
        for rid in ranked[:min(args.max_routes_per_town, len(ranked))]:
            selected[rid] = {
                "town": town,
                "historical_raw_snapshot_count": int(evidence.get(rid, 0)),
                "route_selection_source": "historical_evidence" if rid in evidence else "exploration_no_historical_snapshot",
            }
        selected_count = sum(1 for row in selected.values() if row["town"] == town)
        if selected_count < required:
            unresolved.append("route_coverage_below_capacity_aware_requirement:%s:%d<%d" % (town, selected_count, required))
    return selected, unresolved, capacity


def attach_seed_plan(selected, quota, rate, args):
    grouped = defaultdict(list)
    for rid, item in selected.items():
        grouped[item["town"]].append((rid, item))
    output = {}
    for town, rows in grouped.items():
        requested = max(int(quota.get(town, 0)), len(rows))
        base, remainder = divmod(requested, len(rows))
        for index, (rid, item) in enumerate(sorted(rows)):
            route_quota = base + int(index < remainder)
            raw_per_seed = item["historical_raw_snapshot_count"] / float(args.historical_seed_count)
            accepted_per_seed = raw_per_seed * rate
            uncapped = (math.ceil(route_quota / accepted_per_seed)
                        if accepted_per_seed > 0 else args.max_seed_rounds_per_route)
            rounds = max(args.min_seed_rounds_per_route, min(args.max_seed_rounds_per_route, int(uncapped)))
            output[rid] = {
                **item,
                "estimated_accepted_record_quota": route_quota,
                "historical_raw_per_seed_estimate": raw_per_seed,
                "estimated_accepted_per_seed": accepted_per_seed,
                "recommended_seed_rounds": rounds,
                "estimated_accepted_records_this_wave": rounds * accepted_per_seed,
                "seed_rounds_capped": uncapped > args.max_seed_rounds_per_route,
            }
    return output


def plan_workers(selected, args):
    workers = [{"worker_index": index, "seed": args.seed_start + index, "routes": [],
                "estimated_accepted_record_load": 0.0} for index in range(args.workers)]
    for rid, item in sorted(selected.items(), key=lambda pair: (-pair[1]["estimated_accepted_records_this_wave"], pair[0])):
        worker = min(workers, key=lambda row: (row["estimated_accepted_record_load"], row["worker_index"]))
        worker["routes"].append(rid)
        worker["estimated_accepted_record_load"] += item["estimated_accepted_records_this_wave"]
    for worker in workers:
        worker["routes_subset"] = ",".join(worker["routes"])
        worker["towns"] = sorted({selected[rid]["town"] for rid in worker["routes"]})
    return [row for row in workers if row["routes"]]


def main():
    args = parse_args()
    if args.output_manifest.exists():
        raise FileExistsError("refusing to overwrite manifest: %s" % args.output_manifest)
    if (not args.existing_diagnostics.is_file() or not args.input_dir.is_dir() or not args.routes_xml.is_file() or
            not 0 < args.max_town_fraction < 1 or args.workers < 1 or
            args.min_routes_per_town < 1 or args.max_routes_per_town < args.min_routes_per_town or
            args.min_seed_rounds_per_route < 1 or args.max_seed_rounds_per_route < args.min_seed_rounds_per_route):
        raise ValueError("invalid planner inputs: existing_diagnostics/routes_xml must be files and input_dir must be a directory")
    diag = json.loads(args.existing_diagnostics.read_text(encoding="utf-8"))
    if diag.get("schema_version") != "scenario_dreamer_background_directional_v1":
        raise ValueError("diagnostics are not from background builder")
    existing = Counter({str(k): int(v) for k, v in (diag.get("town_counts") or {}).items()})
    accepted, processed = int(diag.get("records_before_split", 0)), int(diag.get("processed_files", 0))
    if not existing or accepted <= 0 or processed <= 0:
        raise ValueError("diagnostics lack accepted-record and processed-file counts")
    rate = float(args.estimated_acceptance_rate) if args.estimated_acceptance_rate is not None else accepted / processed
    if not 0 < rate <= 1:
        raise ValueError("estimated_acceptance_rate must be in (0, 1]")
    catalog = route_catalog(args.routes_xml)
    evidence, input_rejections = Counter(), Counter()
    for path in args.input_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scene = payload.get("scene", payload) if isinstance(payload, dict) else {}
            rid = route_id(scene)
            town = str(scene.get("town") or (scene.get("scenario") or {}).get("town") or "")
            if rid not in catalog:
                input_rejections["unmapped_route"] += 1
            elif catalog[rid] != town:
                input_rejections["town_mismatch"] += 1
            else:
                evidence[rid] += 1
        except Exception:
            input_rejections["unreadable"] += 1
    dominant, dominant_count = max(existing.items(), key=lambda item: (item[1], item[0]))
    candidate_towns = sorted({catalog[rid] for rid in evidence if catalog[rid] != dominant})
    towns_to_collect = [town for town in candidate_towns if existing.get(town, 0) < args.min_accepted_records_per_town]
    if not towns_to_collect:
        raise ValueError("no non-dominant Town is below min_accepted_records_per_town")
    floor_quota = Counter({town: max(0, args.min_accepted_records_per_town - existing.get(town, 0))
                           for town in towns_to_collect})
    balance_delta = max(0, math.ceil(dominant_count / args.max_town_fraction) - sum(existing.values()))
    desired_delta = max(args.target_total_accepted_records - sum(existing.values()), balance_delta, sum(floor_quota.values()))
    desired = Counter(floor_quota)
    desired.update(waterfill(existing + floor_quota, towns_to_collect, desired_delta - sum(floor_quota.values())))
    wave_size = min(args.collection_wave_accepted_records, desired_delta)
    wave = Counter()
    for _ in range(wave_size):
        town = min(towns_to_collect, key=lambda key: (-(desired[key] - wave[key]), existing.get(key, 0) + wave[key], key))
        wave[town] += 1
    selected, unresolved, route_capacity = choose_routes(evidence, catalog, towns_to_collect, args)
    selected = attach_seed_plan(selected, wave, rate, args)
    workers = plan_workers(selected, args)
    estimate = sum(item["estimated_accepted_records_this_wave"] for item in selected.values())
    manifest = {
        "schema_version": SCHEMA,
        "task": "current_front_scene_completion_not_future_prediction",
        "collection_filter": "none_all_raw_traffic_snapshots",
        "raw_state_postprocessing": "none",
        "dominant_town": dominant,
        "dominant_town_policy": "no_new_generic_collection",
        "existing_accepted_records": accepted,
        "existing_town_counts": dict(sorted(existing.items())),
        "acceptance_rate_estimate": {"value": rate, "source": "builder_diagnostics_global", "warning": "A rate estimate is not an acceptance guarantee; rebuild after every wave."},
        "collection_goal": {"target_total_accepted_records": args.target_total_accepted_records,
                            "minimum_accepted_records_per_town": args.min_accepted_records_per_town,
                            "max_town_fraction": args.max_town_fraction,
                            "required_total_new_accepted_records": desired_delta,
                            "requested_wave_new_accepted_records": wave_size,
                            "estimated_wave_accepted_records_from_route_seed_plan": estimate},
        "coverage": {"towns_to_collect": towns_to_collect,
                     "desired_accepted_quota_by_town": dict(sorted(desired.items())),
                     "wave_accepted_quota_by_town": dict(sorted(wave.items())),
                     "min_routes_per_town": args.min_routes_per_town,
                     "route_capacity_by_town": dict(sorted(route_capacity.items())),
                     "historical_seed_count": args.historical_seed_count},
        "selected_routes": dict(sorted(selected.items())),
        "workers": workers,
        "input_rejections": dict(sorted(input_rejections.items())),
        "unresolved_planning_reasons": unresolved,
        "training_gate": "Do not train from planned yield. Merge completed waves, rebuild the directional dataset, and require Town, topology, split and raw-state audits to pass.",
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
