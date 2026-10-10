#!/usr/bin/env python
"""Visualize high-value scene-generation JSONL records."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_plots", type=int, default=100)
    parser.add_argument("--radius", type=float, default=80.0)
    return parser.parse_args()


def draw_box(ax, state, color, label=None, linewidth=1.5, alpha=1.0):
    x, y, heading, _speed, length, width = [float(v) for v in state[:6]]
    rect = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    rect.set_transform(Affine2D().rotate(heading).translate(x, y) + ax.transData)
    ax.add_patch(rect)


def target_states(record):
    target = record.get("target", {})
    return target.get("generated_agent_states") or target.get("agent_states") or []


def plot_record(record, path, radius):
    condition = record["condition"]
    fig, ax = plt.subplots(figsize=(9, 9))

    for lane in condition.get("map", {}).get("lane_polylines", []):
        if len(lane) >= 2:
            xs = [float(p[0]) for p in lane]
            ys = [float(p[1]) for p in lane]
            ax.plot(xs, ys, color="0.85", linewidth=0.8)

    route = condition.get("ego_route", {}).get("polyline", [])
    if len(route) >= 2:
        ax.plot([float(p[0]) for p in route], [float(p[1]) for p in route], color="black", linewidth=2.0, label="ego route")

    draw_box(ax, condition.get("ego_state", [0, 0, 0, 0, 5.176, 2.297, 1]), "tab:blue", label="ego", linewidth=2.0)

    first_observed = True
    for state in condition.get("observed_agent_states", []):
        draw_box(ax, state, "0.55", label="observed" if first_observed else None, linewidth=1.0, alpha=0.8)
        first_observed = False

    first_target = True
    for state in target_states(record):
        draw_box(ax, state, "tab:red", label="target" if first_target else None, linewidth=2.0)
        first_target = False

    ax.set_title(
        f"{record.get('scene_id', 'scene')} | observed={len(condition.get('observed_agent_states', []))} "
        f"target={len(target_states(record))}"
    )
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    records = read_jsonl(args.jsonl)[:args.max_plots]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(records):
        scene_id = str(record.get("scene_id", idx)).replace("/", "_").replace("\\", "_").replace(":", "_")
        plot_record(record, out_dir / f"{idx:04d}_{scene_id}.png", args.radius)
    print(json.dumps({"output_dir": str(out_dir), "num_plots": len(records)}, indent=2))


if __name__ == "__main__":
    main()

