#!/usr/bin/env python
"""Static audit for count-only background replenishment policy."""

import argparse
import json
from pathlib import Path

from high_value_init.background_controller import BackgroundReplenishmentController


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--count_model", required=True); parser.add_argument("--output_json", required=True)
    args = parser.parse_args(); controller = BackgroundReplenishmentController(args.count_model)
    first = controller.plan(map_bucket="multi_lane", route_bucket="route_straight", num_before=2,
                            live_model_background=0, non_target_context_count=3, seed=7)
    repeat = controller.plan(map_bucket="multi_lane", route_bucket="route_straight", num_before=2,
                             live_model_background=first["desired_model_background"], non_target_context_count=3, seed=7)
    overflow = controller.plan(map_bucket="multi_lane", route_bucket="route_straight", num_before=2,
                               live_model_background=0, non_target_context_count=61, seed=7)
    errors = []
    if first["k_missing"] != first["desired_model_background"]: errors.append("initial_missing_not_desired")
    if repeat["k_missing"] != 0: errors.append("controller_readds_complete_batch")
    if overflow["accepted"] or overflow["rejection_reason"] != "graph_capacity_rejected_no_context_truncation": errors.append("capacity_not_rejected")
    report = {"passed": not errors, "errors": errors, "first_plan": first, "repeat_plan": repeat, "overflow_plan": overflow,
              "note": "Controller is count-only; CARLA spawn/delete/Traffic Manager integration is intentionally gated on raw model evaluation."}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True); Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(2)


if __name__ == "__main__": main()
