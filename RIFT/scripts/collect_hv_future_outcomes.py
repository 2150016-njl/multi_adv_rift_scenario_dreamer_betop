#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Forwarding entry for current-anchor HV future-outcome sidecars.

It deliberately starts an ordinary RIFT rollout: no HV generator, no state
post-processing, and no audit-driven registration is enabled here.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True,
                        help="New sidecar directory; never point this at a training dataset.")
    parser.add_argument("--routes", required=True)
    parser.add_argument("--routes_subset", default="")
    parser.add_argument("--ego_cfg", default="pdm_lite.yaml")
    parser.add_argument("--cbv_cfg", default="rift_pluto.yaml")
    parser.add_argument("--cbv_recog", default="rule")
    parser.add_argument("--repetitions", "-rep", type=int, default=1)
    parser.add_argument("--num_scenario", "-ns", type=int, default=1)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--base_port", type=int, default=None)
    parser.add_argument("--base_tm_port", type=int, default=None)
    parser.add_argument("--rift_hd_map_dir", required=True)
    parser.add_argument("--rift_speed_limits_dir", required=True)
    parser.add_argument("--horizon_s", type=float, default=5.0)
    parser.add_argument("--anchor_interval_s", type=float, default=1.0)
    parser.add_argument("--observe_interval_steps", type=int, default=1)
    parser.add_argument("--max_frames_per_episode", type=int, default=200)
    parser.add_argument("--max_anchors_per_episode", type=int, default=1000000)
    parser.add_argument("--min_forward_rift_positive", type=int, default=2)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--fov", type=float, default=64.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_forward_rift_positive < 2:
        raise ValueError("future-outcome anchors require at least two forward RIFT-positive actors")
    command = [
        sys.executable, str(Path(__file__).with_name("run.py")),
        "--mode", "collect_hv_future_outcome",
        "--output_dir", str(Path(args.output_dir) / "rift_statistics"),
        "--hv_future_outcome_dir", args.output_dir,
        "--routes", args.routes, "--ego_cfg", args.ego_cfg, "--cbv_cfg", args.cbv_cfg,
        "--cbv_recog", args.cbv_recog, "-rep", str(args.repetitions), "-ns", str(args.num_scenario),
        "--seed", str(args.seed), "--rift_hd_map_dir", args.rift_hd_map_dir,
        "--rift_speed_limits_dir", args.rift_speed_limits_dir,
        "--hv_future_outcome_horizon_s", str(args.horizon_s),
        "--hv_future_outcome_anchor_interval_s", str(args.anchor_interval_s),
        "--hv_future_outcome_observe_interval_steps", str(args.observe_interval_steps),
        "--hv_future_outcome_min_forward_rift_positive", str(args.min_forward_rift_positive),
        "--hv_generation_max_frames_per_episode", str(args.max_frames_per_episode),
        "--hv_generation_max_snapshots_per_episode", str(args.max_anchors_per_episode),
        "--hv_generation_forward_min_x", str(args.forward_min_x),
        "--hv_generation_target_fov_margin", str(args.target_fov_margin),
        "--hv_sd_fov", str(args.fov),
    ]
    if args.routes_subset:
        command.extend(["--routes_subset", args.routes_subset])
    if args.no_resume:
        command.append("--no_resume")
    if args.base_port is not None:
        command.extend(["--base_port", str(args.base_port)])
    if args.base_tm_port is not None:
        command.extend(["--base_tm_port", str(args.base_tm_port)])
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
