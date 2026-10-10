#!/usr/bin/env python
"""Create a source-aware, paired report for frozen RIFT online runs.

This is an analysis-only companion to ``scripts/run.py``.  It never launches
CARLA, changes actors, applies a selection policy, or edits an existing dump.
It combines the observation-only HV dump/lifecycle evidence with RIFT's route
statistics so no-HV, V2, V3, and V4-A runs can be compared reproducibly.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from summarize_high_value_generation_online import (
    load_lifecycle_events,
    load_lifecycle_metadata,
    load_payloads,
    numeric,
    summarize_lifecycle,
    summarize_payloads,
    summarize_statistics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs=3,
        metavar=("NAME", "DUMP_DIR_OR_DASH", "STATISTICS_DIR_OR_JSON"),
        action="append",
        required=True,
        help="Repeat once per method. Use '-' for a no-HV run without a dump directory.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Name of one --run entry used for matched per-route deltas, normally no_hv.",
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, default=None)
    return parser.parse_args()


def find_statistics_path(value):
    path = Path(value)
    if path.is_file():
        return path
    candidates = sorted(
        path.rglob("simulation_results.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    ) if path.exists() else []
    return candidates[0] if candidates else None


def collision_count(record):
    infractions = record.get("infractions") or {}
    return sum(len(infractions.get(key) or []) for key in (
        "collisions_pedestrian", "collisions_vehicle", "collisions_layout",
    ))


def route_key(record, index):
    """Use a stable route identifier when the RIFT statistics schema exposes it."""
    candidates = (
        record.get("route_id"),
        record.get("scenario_id"),
        record.get("route"),
        record.get("name"),
        (record.get("meta") or {}).get("route_id"),
        (record.get("meta") or {}).get("scenario_id"),
    )
    for value in candidates:
        if value is not None and str(value):
            return str(value)
    return "record_index:%d" % int(index)


def load_route_rows(statistics_path):
    if statistics_path is None or not statistics_path.exists():
        return {"route_key_source": "unavailable", "rows": []}
    payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    records = ((payload.get("_checkpoint") or {}).get("records") or [])
    rows = []
    for index, record in enumerate(records):
        scores = record.get("scores") or {}
        rows.append({
            "route_key": route_key(record, index),
            "route_index": index,
            "route_completion": float(scores.get("score_route", 0.0)),
            "driving_score": float(scores.get("score_composed", 0.0)),
            "penalty_score": float(scores.get("score_penalty", 0.0)),
            "collision_count": collision_count(record),
        })
    return {"route_key_source": "schema_field_or_record_index_fallback", "rows": rows}


def compact_online_summary(dump_dir):
    if dump_dir is None:
        return {
            "dump_available": False,
            "reason": "no_dump_directory",
        }
    if not dump_dir.exists():
        return {
            "dump_available": False,
            "reason": "missing_dump_directory",
            "dump_dir": str(dump_dir),
        }
    payloads = load_payloads(dump_dir)
    events = load_lifecycle_events(dump_dir)
    metadata = load_lifecycle_metadata(dump_dir)
    return {
        "dump_available": True,
        "dump_dir": str(dump_dir),
        "metadata": metadata,
        "summary": summarize_payloads(payloads, events, metadata),
        "lifecycle_detail": summarize_lifecycle(events, metadata),
    }


def summarize_paired_delta(baseline_rows, candidate_rows):
    baseline_by_key = {row["route_key"]: row for row in baseline_rows}
    candidate_by_key = {row["route_key"]: row for row in candidate_rows}
    matched_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    metrics = ("route_completion", "driving_score", "penalty_score", "collision_count")
    deltas = {
        metric: numeric(
            candidate_by_key[key][metric] - baseline_by_key[key][metric]
            for key in matched_keys
        )
        for metric in metrics
    }
    return {
        "matched_route_count": len(matched_keys),
        "baseline_only_route_count": len(set(baseline_by_key) - set(candidate_by_key)),
        "candidate_only_route_count": len(set(candidate_by_key) - set(baseline_by_key)),
        "candidate_minus_baseline": deltas,
    }


def markdown(report):
    lines = [
        "# Matched HV Online Evaluation",
        "",
        "This report is observational. RIFT audit labels are not a sample selector or actor-state editor.",
        "",
        "| method | routes | completion mean | driving score mean | collision-route ratio | model calls | spawn success | HV orange-to-green |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["runs"]:
        statistics = item.get("rift_statistics") or {}
        online = (item.get("online") or {}).get("summary") or {}
        lifecycle = online.get("lifecycle") or {}
        route = (statistics.get("route_completion") or {}).get("mean")
        driving = (statistics.get("driving_score") or {}).get("mean")
        collision = statistics.get("collision_routes_ratio")
        spawn = online.get("carla_spawn_success_rate")
        conversion = lifecycle.get("orange_to_green_conversion_rate")
        values = (
            item["name"],
            statistics.get("num_routes", 0),
            "" if route is None else "%.4f" % route,
            "" if driving is None else "%.4f" % driving,
            "" if collision is None else "%.4f" % collision,
            online.get("generator_called_count", 0),
            "" if spawn is None else "%.4f" % spawn,
            "" if conversion is None else "%.4f" % conversion,
        )
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % values)
    if report.get("paired_against_baseline"):
        lines.extend(["", "## Paired deltas", ""])
        for name, value in sorted(report["paired_against_baseline"].items()):
            score = ((value.get("candidate_minus_baseline") or {}).get("driving_score") or {}).get("mean")
            completion = ((value.get("candidate_minus_baseline") or {}).get("route_completion") or {}).get("mean")
            lines.append(
                "- %s: matched routes=%s, delta driving score=%s, delta completion=%s" % (
                    name,
                    value.get("matched_route_count", 0),
                    "n/a" if score is None else "%.4f" % score,
                    "n/a" if completion is None else "%.4f" % completion,
                )
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    names = [item[0] for item in args.run]
    if len(set(names)) != len(names):
        raise ValueError("--run names must be unique")
    if args.baseline is not None and args.baseline not in names:
        raise ValueError("--baseline must name one --run entry")

    runs = []
    route_rows_by_name = {}
    for name, dump_value, statistics_value in args.run:
        dump_dir = None if dump_value == "-" else Path(dump_value)
        statistics_path = find_statistics_path(statistics_value)
        route_rows = load_route_rows(statistics_path)
        route_rows_by_name[name] = route_rows["rows"]
        runs.append({
            "name": name,
            "statistics_path": None if statistics_path is None else str(statistics_path),
            "rift_statistics": summarize_statistics(statistics_path),
            "route_rows": route_rows,
            "online": compact_online_summary(dump_dir),
        })

    paired = {}
    if args.baseline is not None:
        baseline_rows = route_rows_by_name[args.baseline]
        for name, candidate_rows in route_rows_by_name.items():
            if name != args.baseline:
                paired[name] = summarize_paired_delta(baseline_rows, candidate_rows)

    report = {
        "schema_version": "hv_online_matched_comparison_v1",
        "baseline": args.baseline,
        "runs": runs,
        "paired_against_baseline": paired,
        "interpretation": {
            "raw_state_policy": "No lane snapping, heading replacement, clamp, audit reranking, or audit-driven registration is performed by this report.",
            "driving_score": "Ego driving score is a route outcome, not a generator-only metric. Interpret paired deltas only when routes, seeds, policies, and simulator configuration match.",
            "oracle": "This report contains online one-draw behavior only; offline best-of-N oracle results must not be mixed into it.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
