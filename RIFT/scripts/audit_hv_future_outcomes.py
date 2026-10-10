#!/usr/bin/env python
"""Audit that future labels are sidecars, current-anchor based, and non-causal."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--outcomes_jsonl", type=Path, required=True); parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args(); failures, rows = [], []
    for index, line in enumerate(args.outcomes_jsonl.read_text(encoding="utf-8").splitlines()):
        if not line.strip(): continue
        row = json.loads(line); rows.append(row)
        for key, value in (("schema_version", "hv_future_outcome_v1"), ("task", "future_outcome_supervision_for_current_time_anchor_only"),
                           ("future_positions_used_as_spawn_state", False), ("counterfactual_ego_completion_impact_available", False)):
            if row.get(key) != value: failures.append({"index": index, "field": key, "value": row.get(key)})
        if not row.get("snapshot_path") or row.get("anchor_actor_id") is None or not isinstance(row.get("anchor_raw_actor_state"), dict):
            failures.append({"index": index, "field": "missing_current_anchor"})
        if not isinstance(row.get("route_corridor_overlap_any"), bool) or "proxy only" not in str(row.get("route_corridor_overlap_definition", "")):
            failures.append({"index": index, "field": "route_corridor_proxy_contract"})
        if row.get("activation_delay_s") is not None and float(row["activation_delay_s"]) < 0:
            failures.append({"index": index, "field": "negative_activation_delay"})
    report = {"passed": not failures, "rows": len(rows), "activation_rate": (sum(bool(row.get("normal_cbv_activated")) for row in rows) / len(rows)) if rows else None,
              "horizon_completed_rate": (sum(bool(row.get("horizon_completed")) for row in rows) / len(rows)) if rows else None,
              "route_corridor_overlap_rate": (sum(bool(row.get("route_corridor_overlap_any")) for row in rows) / len(rows)) if rows else None,
              "town_counts": dict(Counter(str(row.get("town")) for row in rows)), "failures": failures[:100]}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))
    if failures: raise SystemExit(2)


if __name__ == "__main__": main()
