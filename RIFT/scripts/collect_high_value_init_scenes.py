#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Forwarding entry for RIFT high-value scene snapshot collection."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--hv_generation_dump_dir", dest="output_dir", required=True)
    parser.add_argument("--routes", default=None)
    parser.add_argument("--routes_subset", default="")
    parser.add_argument("--ego_cfg", default="pdm_lite.yaml")
    parser.add_argument("--cbv_cfg", default="rift_pluto.yaml")
    parser.add_argument("--cbv_recog", default="rule")
    parser.add_argument("--repetitions", "-rep", type=int, default=1)
    parser.add_argument("--num_scenario", "-ns", type=int, default=1)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--no_resume", action="store_true", default=False)
    parser.add_argument("--base_port", type=int, default=None)
    parser.add_argument("--base_tm_port", type=int, default=None)
    parser.add_argument("--rift_hd_map_dir", default=None)
    parser.add_argument("--rift_speed_limits_dir", default=None)
    parser.add_argument("--sample_interval_s", type=float, default=1.0)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--map_radius", type=float, default=64.0)
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Convenience alias for --routes_subset 0-(max_episodes-1) when route ids are contiguous.")
    parser.add_argument("--max_frames_per_episode", type=int, default=200)
    parser.add_argument("--max_snapshots_per_episode", type=int, default=1000000)
    parser.add_argument("--min_forward_rift_positive", type=int, default=0,
                        help="Only persist snapshots with at least this many forward-sector RIFT-positive actors. 0 keeps all snapshots.")
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--hv_sd_fov", type=float, default=64.0)
    parser.add_argument("--extra_run_arg", action="append", default=[],
                        help="Extra raw argument forwarded to scripts/run.py; repeat as needed.")
    return parser.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).with_name("run.py")
    cmd = [
        sys.executable,
        str(script),
        "--mode", "collect_hv_generation",
        "--ego_cfg", args.ego_cfg,
        "--cbv_cfg", args.cbv_cfg,
        "--cbv_recog", args.cbv_recog,
        "-rep", str(args.repetitions),
        "-ns", str(args.num_scenario),
        "--seed", str(args.seed),
        "--hv_generation_dump_dir", args.output_dir,
        "--hv_generation_sample_interval_s", str(args.sample_interval_s),
        "--hv_actor_radius", str(args.actor_radius),
        "--hv_map_radius", str(args.map_radius),
        "--hv_generation_max_frames_per_episode", str(args.max_frames_per_episode),
        "--hv_generation_max_snapshots_per_episode", str(args.max_snapshots_per_episode),
        "--hv_generation_collect_min_forward_rift_positive", str(args.min_forward_rift_positive),
        "--hv_generation_forward_min_x", str(args.forward_min_x),
        "--hv_generation_target_fov_margin", str(args.target_fov_margin),
        "--hv_sd_fov", str(args.hv_sd_fov),
    ]
    if args.routes:
        cmd.extend(["--routes", args.routes])
    routes_subset = args.routes_subset
    if not routes_subset and args.max_episodes is not None:
        routes_subset = f"0-{int(args.max_episodes) - 1}"
    if routes_subset:
        cmd.extend(["--routes_subset", routes_subset])
    if args.no_resume:
        cmd.append("--no_resume")
    if args.base_port is not None:
        cmd.extend(["--base_port", str(args.base_port)])
    if args.base_tm_port is not None:
        cmd.extend(["--base_tm_port", str(args.base_tm_port)])
    if args.rift_hd_map_dir:
        cmd.extend(["--rift_hd_map_dir", args.rift_hd_map_dir])
    if args.rift_speed_limits_dir:
        cmd.extend(["--rift_speed_limits_dir", args.rift_speed_limits_dir])
    cmd.extend(args.extra_run_arg)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
