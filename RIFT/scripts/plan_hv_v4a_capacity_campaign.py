#!/usr/bin/env python
"""Capacity-plan a large, quota-driven V4-A forward-HV collection wave.

Unlike the legacy V4-A planner, this planner turns an *accepted-record*
shortfall into independent route/seed tasks.  It never treats predicted yield
as collected data: the V4 builder remains the final quantity/quality/diversity
gate after every wave.
"""
import argparse
import json
import math
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--existing_diagnostics", type=Path, default=None,
                   help="Strict/probe V4-A diagnostics; may be absent only with --summary_only_recovery.")
    p.add_argument("--summary_only_recovery", action="store_true",
                   help="Allow a missing diagnostics file and size the campaign from additional_collection_summary only. Historical data receives zero quantity credit.")
    p.add_argument("--additional_collection_summary", type=Path, default=None,
                   help="Completed V4-A collection summary. Its valid forward-pair Town counts are used as a balance upper bound.")
    p.add_argument("--additional_retention_fraction", type=float, default=1.0,
                   help="Conservative fraction of additional valid snapshots assumed to survive dedup/quality for quantity planning.")
    p.add_argument("--prior_manifest", type=Path, default=None,
                   help="Optional earlier V4-A manifest used only if a legacy diagnostics file lacks accepted Town counts.")
    p.add_argument("--routes_xml", type=Path, required=True)
    p.add_argument("--output_manifest", type=Path, required=True)
    p.add_argument("--rift_root", type=Path, required=True)
    p.add_argument("--rift_hd_map_dir", type=Path, required=True)
    p.add_argument("--target_total_accepted_records", type=int, default=30000)
    p.add_argument("--wave_accepted_records", type=int, default=8000,
                   help="Accepted-record target for this launch.  Set 0 for one full campaign that covers the entire remaining deficit.")
    p.add_argument("--min_accepted_records_per_town", type=int, default=2000)
    p.add_argument("--max_town_fraction", type=float, default=.35)
    p.add_argument("--min_routes_per_town", type=int, default=8)
    p.add_argument("--max_routes_per_town", type=int, default=12)
    p.add_argument("--min_seed_rounds_per_route", type=int, default=5)
    p.add_argument("--max_seed_rounds_per_route", type=int, default=30)
    p.add_argument("--uniform_all_route_seed_rounds", type=int, default=0,
                   help="When positive, collect every XML route.  Each non-dominant Town route gets this many seeds instead of quota-driven allocation.")
    p.add_argument("--dominant_town_seed_rounds", type=int, default=0,
                   help="Controlled seeds per route for the historical dominant Town when it is explicitly included.")
    p.add_argument("--include_dominant_town_in_uniform", action="store_true",
                   help="Opt in to collecting the historical dominant Town in uniform all-route mode.  Off by default.")
    p.add_argument("--exclude_towns", default="",
                   help="Comma-separated Town names that must receive no new generic collection (for example Town12,Town13).")
    p.add_argument("--historical_seed_count", type=int, default=8)
    p.add_argument("--estimated_acceptance_rate", type=float, default=None)
    p.add_argument("--estimated_accepted_per_route_seed", type=float, default=5.0,
                   help="Conservative targeted-pair yield used for task sizing; it is not accepted-data evidence.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed_start", type=int, default=1000)
    p.add_argument("--fov", type=float, default=64.0)
    p.add_argument("--actor_radius", type=float, default=80.0)
    p.add_argument("--forward_min_x", type=float, default=0.0)
    p.add_argument("--target_fov_margin", type=float, default=2.0)
    p.add_argument("--max_target_lane_distance", type=float, default=5.0)
    return p.parse_args()


def catalog(path):
    root = ET.parse(path).getroot()
    result = {str(n.get("id")): str(n.get("town")) for n in root.findall(".//route") if n.get("id") and n.get("town")}
    if not result:
        raise ValueError("routes XML contains no route/town catalog")
    return result


def waterfill(existing, towns, amount):
    result = Counter()
    for _ in range(max(0, int(amount))):
        town = min(towns, key=lambda x: (existing.get(x, 0) + result[x], x))
        result[town] += 1
    return result


def main():
    a = args_parser()
    if a.output_manifest.exists():
        raise FileExistsError("refusing to overwrite manifest: %s" % a.output_manifest)
    diagnostics_is_file = a.existing_diagnostics is not None and a.existing_diagnostics.is_file()
    if not (a.input_dir.is_dir() and a.routes_xml.is_file()):
        raise ValueError("input_dir must be a directory and routes_xml must be a file")
    if not diagnostics_is_file and not a.summary_only_recovery:
        raise ValueError("existing_diagnostics must be a file, or use --summary_only_recovery")
    required_paths = {
        "input_dir": (a.input_dir, a.input_dir.is_dir()),
        "existing_diagnostics": (a.existing_diagnostics,
                                 diagnostics_is_file or a.summary_only_recovery),
        "routes_xml": (a.routes_xml, a.routes_xml.is_file()),
    }
    invalid_paths = {key: str(path) for key, (path, valid) in required_paths.items() if not valid}
    if invalid_paths:
        details = {key: {"path": str(path), "exists": path is not None and path.exists(),
                         "is_file": path is not None and path.is_file(), "is_dir": path is not None and path.is_dir()}
                   for key, (path, _) in required_paths.items()}
        raise ValueError("invalid required paths: " + json.dumps(invalid_paths, sort_keys=True) +
                         "; details=" + json.dumps(details, sort_keys=True))
    if a.additional_collection_summary is not None and not a.additional_collection_summary.is_file():
        raise ValueError("additional_collection_summary must be an existing file when supplied")
    if a.prior_manifest is not None and not a.prior_manifest.is_file():
        raise ValueError("prior_manifest must be an existing file when supplied")
    if a.workers < 1 or a.min_routes_per_town < 1 or a.max_routes_per_town < a.min_routes_per_town:
        raise ValueError("invalid worker/route limits")
    if a.uniform_all_route_seed_rounds < 0 or a.dominant_town_seed_rounds < 0:
        raise ValueError("uniform seed-round counts must be non-negative")
    if a.include_dominant_town_in_uniform and not a.uniform_all_route_seed_rounds:
        raise ValueError("include_dominant_town_in_uniform requires uniform_all_route_seed_rounds")
    if a.include_dominant_town_in_uniform and not a.dominant_town_seed_rounds:
        raise ValueError("dominant_town_seed_rounds must be positive in uniform all-route mode")
    if diagnostics_is_file:
        diag = json.loads(a.existing_diagnostics.read_text(encoding="utf-8"))
        if diag.get("schema_version") != "hv_scene_dataset_v4a_full_fov_forward":
            raise ValueError("existing diagnostics are not a V4-A builder report")
    else:
        diag = {}
    # A strict completed dataset has split-level ``distribution``.  A large
    # but still below-scale probe has only pre-split accepted counts; those
    # are the correct evidence for planning the next collection wave.
    distribution = diag.get("distribution") or diag.get("accepted_record_distribution_pre_split") or {}
    existing = Counter({str(k): int(v) for k, v in (distribution.get("town_counts") or {}).items()})
    accepted = int(diag.get("num_records_after_v2_quality_and_dedup", sum(existing.values())))
    processed = int(diag.get("processed_files", 0))
    count_provenance = "diagnostics" if diagnostics_is_file else "summary_only_recovery_zero_historical_credit"
    if not existing and a.prior_manifest is not None:
        prior = json.loads(a.prior_manifest.read_text(encoding="utf-8"))
        prior_counts = ((prior.get("existing_distribution") or {}).get("town_counts") or
                        prior.get("existing_town_counts") or {})
        existing = Counter({str(k): int(v) for k, v in prior_counts.items()})
        count_provenance = "prior_manifest_partial"
    if not existing and a.summary_only_recovery:
        # additional_collection_summary below provides the only quantitative
        # evidence.  Do not invent historical accepted counts.
        pass
    elif not existing or accepted <= 0:
        raise ValueError("diagnostics must contain accepted Town counts, or supply --prior_manifest with historical Town counts")
    # Diagnostics-only builders prior to this patch have no Town distribution.
    # Their total is still valid, while the older manifest has a conservative
    # partial Town attribution.  We retain the difference explicitly instead
    # of pretending it has a Town label.
    unattributed = max(0, accepted - sum(existing.values()))
    rate = float(a.estimated_acceptance_rate) if a.estimated_acceptance_rate is not None else (accepted / processed if processed else 1.0 if a.summary_only_recovery else 0.0)
    if not 0 < rate <= 1:
        raise ValueError("diagnostics must contain processed_files, or pass --estimated_acceptance_rate")
    if a.estimated_accepted_per_route_seed <= 0:
        raise ValueError("estimated_accepted_per_route_seed must be positive")
    if not 0 < a.additional_retention_fraction <= 1:
        raise ValueError("additional_retention_fraction must be in (0, 1]")
    routes = catalog(a.routes_xml)

    # A completed collection has passed the target selector but not the final
    # builder's dedup/quality gates.  For balance we count every such record
    # (worst case: all survive); for quantity we use a caller-specified lower
    # retention fraction.  This avoids both Town over-collection and the old
    # planner's inflated accepted-yield claim.
    additional_raw_by_town = Counter()
    additional_raw_total = 0
    if a.additional_collection_summary is not None:
        summary = json.loads(a.additional_collection_summary.read_text(encoding="utf-8"))
        if summary.get("schema_version") != "hv_init_rift_labeled_snapshot_summary_v2_forward_coverage":
            raise ValueError("additional_collection_summary is not a forward-coverage snapshot summary")
        if int(summary.get("num_forward_positive_scenes", 0)) != int(summary.get("valid_rift_labeled_snapshot_count", -1)):
            raise ValueError("additional collection includes scenes outside the forward fixed-pair contract")
        additional_raw_by_town = Counter({str(k): int(v) for k, v in (summary.get("town_counts") or {}).items()})
        additional_raw_total = sum(additional_raw_by_town.values())
        if additional_raw_total <= 0:
            raise ValueError("additional collection summary has no valid snapshots")

    balance_existing = existing + additional_raw_by_town
    conservative_accepted = accepted + math.floor(additional_raw_total * a.additional_retention_fraction)
    dominant, dominant_count = max(balance_existing.items(), key=lambda item: (item[1], item[0]))
    excluded = {x.strip() for x in a.exclude_towns.split(",") if x.strip()}
    excluded.add(dominant)
    towns = sorted(set(routes.values()) - excluded)
    if not towns:
        raise ValueError("no collectable Town remains after dominant/exclude_towns policy")
    floor = Counter({t: max(0, a.min_accepted_records_per_town - balance_existing.get(t, 0)) for t in towns})
    balance_need = max(0, math.ceil(dominant_count / a.max_town_fraction) - conservative_accepted)
    required = max(a.target_total_accepted_records - conservative_accepted, balance_need, sum(floor.values()))
    desired = Counter(floor)
    desired.update(waterfill(balance_existing + desired, towns, required - sum(desired.values())))
    if a.wave_accepted_records < 0:
        raise ValueError("wave_accepted_records must be non-negative")
    wave_size = required if a.wave_accepted_records == 0 else min(required, a.wave_accepted_records)
    wave = Counter()
    for _ in range(wave_size):
        # Counter.__getitem__ supplies zero for an unseen Town; dict.get
        # would return None and break the integer quota calculation.
        town = min(towns, key=lambda x: (-(desired[x] - wave[x]), balance_existing[x] + wave[x], x))
        wave[town] += 1

    by_town = defaultdict(list)
    for rid, town in routes.items():
        by_town[town].append(rid)
    selected, exploration = {}, []
    if a.uniform_all_route_seed_rounds:
        # Full-campaign mode: route diversity comes first.  The historical
        # dominant Town is excluded unless explicitly requested; it already
        # has substantial evidence and must not be re-collected by default.
        for town in sorted(by_town):
            if town in excluded and not (town == dominant and a.include_dominant_town_in_uniform):
                continue
            rounds = a.dominant_town_seed_rounds if town == dominant else a.uniform_all_route_seed_rounds
            if rounds > a.max_seed_rounds_per_route:
                raise ValueError("uniform seed rounds exceed max_seed_rounds_per_route")
            for rid in sorted(by_town[town]):
                selected[rid] = {"town": town, "historical_forward_pair_candidates": None,
                                 "route_selection_source": "uniform_all_xml_routes_no_hdmap_rescan",
                                 "accepted_quota_this_wave": rounds * float(a.estimated_accepted_per_route_seed),
                                 "estimated_accepted_per_seed": float(a.estimated_accepted_per_route_seed),
                                 "recommended_seed_rounds": rounds, "seed_rounds_capped": False}
                exploration.append(rid)
    else:
        for town in towns:
            # Metadata-only planning intentionally avoids a second full HD-map
            # extraction over tens of thousands of snapshots.  It creates broad
            # route exploration; the strict builder remains the sole acceptance
            # authority once collection finishes.
            ranked = sorted(by_town[town], key=lambda rid: hashlib.sha256((str(a.seed_start) + ":" + rid).encode()).hexdigest())[:a.max_routes_per_town]
            if len(ranked) < min(a.min_routes_per_town, len(by_town[town])):
                raise RuntimeError("route coverage logic failed for " + town)
            quota_base, rem = divmod(wave[town], len(ranked))
            for idx, rid in enumerate(ranked):
                quota = quota_base + int(idx < rem)
                per_seed_accepted = float(a.estimated_accepted_per_route_seed)
                uncapped = math.ceil(max(1, quota) / per_seed_accepted)
                rounds = max(a.min_seed_rounds_per_route, min(a.max_seed_rounds_per_route, uncapped))
                selected[rid] = {"town": town, "historical_forward_pair_candidates": None,
                                 "route_selection_source": "metadata_coverage_exploration_no_hdmap_rescan",
                                 "accepted_quota_this_wave": quota, "estimated_accepted_per_seed": per_seed_accepted,
                                 "recommended_seed_rounds": rounds, "seed_rounds_capped": uncapped > a.max_seed_rounds_per_route}
                exploration.append(rid)

    jobs = [{"worker_index": i, "seed": a.seed_start + i, "routes": [], "estimated_accepted_load": 0.0} for i in range(a.workers)]
    for rid, row in sorted(selected.items(), key=lambda kv: (-(kv[1]["estimated_accepted_per_seed"] * kv[1]["recommended_seed_rounds"]), kv[0])):
        job = min(jobs, key=lambda x: (x["estimated_accepted_load"], x["worker_index"]))
        job["routes"].append(rid)
        job["estimated_accepted_load"] += row["estimated_accepted_per_seed"] * row["recommended_seed_rounds"]
    for job in jobs:
        job["routes_subset"] = ",".join(job["routes"])
        job["towns"] = sorted({selected[r]["town"] for r in job["routes"]})

    estimated_wave = sum(r["estimated_accepted_per_seed"] * r["recommended_seed_rounds"] for r in selected.values())
    # A legacy diagnostics file may give only a partial Town attribution.
    # Assign every unattributed accepted record to the dominant Town for a
    # conservative fraction check rather than understating the imbalance.
    estimated_dominant = balance_existing[dominant] + unattributed + sum(r["estimated_accepted_per_seed"] * r["recommended_seed_rounds"] for r in selected.values() if r["town"] == dominant)
    if estimated_dominant / (conservative_accepted + estimated_wave) > a.max_town_fraction:
        raise ValueError("planned dominant Town share exceeds max_town_fraction")
    manifest = {"schema_version": "hv_v4a_capacity_campaign_v1", "task": "current_full_fov_condition_forward_fixed_pair",
                "collection_contract": {"fov": 64.0, "num_generate_agents": 2, "forward_min_x": 0.0,
                                        "min_forward_rift_positive": 2, "raw_state_postprocessing": "none"},
                "planning_mode": "uniform_all_xml_routes_no_hdmap_rescan" if a.uniform_all_route_seed_rounds else "metadata_only_no_hdmap_rescan",
                "existing_accepted_records": accepted, "existing_town_counts": dict(sorted(existing.items())),
                "additional_valid_forward_snapshots": additional_raw_total,
                "additional_valid_forward_snapshots_by_town": dict(sorted(additional_raw_by_town.items())),
                "additional_retention_fraction_for_quantity": a.additional_retention_fraction,
                "conservative_accepted_before_new_campaign": conservative_accepted,
                "balance_upper_bound_town_counts_before_new_campaign": dict(sorted(balance_existing.items())),
                "unattributed_existing_accepted_records": unattributed,
                "existing_count_provenance": count_provenance,
                "dominant_town": dominant,
                "excluded_from_new_collection": sorted(excluded),
                "dominant_town_policy": ("controlled_all_route_coverage" if a.include_dominant_town_in_uniform else "no_new_generic_collection"),
                "acceptance_rate_estimate": rate,
                "collection_goal": {"target_total_accepted_records": a.target_total_accepted_records,
                                    "required_new_accepted_records": required, "requested_wave_accepted_records": wave_size,
                                    "estimated_wave_accepted_records": estimated_wave,
                                    "estimated_wave_shortfall": max(0.0, wave_size - estimated_wave)},
                "coverage": {"desired_accepted_quota_by_town": dict(sorted(desired.items())),
                             "wave_accepted_quota_by_town": dict(sorted(wave.items())),
                             "min_routes_per_town": a.min_routes_per_town, "max_routes_per_town": a.max_routes_per_town},
                "selected_routes": dict(sorted(selected.items())), "exploration_route_ids": sorted(exploration),
                "workers": jobs, "planning_rejections": {"not_scanned": "intentional: no second HD-map extraction before collection"},
                "training_gate": "Merge this wave and require the strict V4-A builder to pass >=30k, Town/topology and split gates before training. Estimates are intentionally conservative; only the strict builder is acceptance evidence."}
    a.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    a.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
