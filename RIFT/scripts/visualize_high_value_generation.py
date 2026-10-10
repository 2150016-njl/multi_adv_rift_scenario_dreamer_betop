#!/usr/bin/env python
"""Save BEV PNGs for high-value generated-scene dumps."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_plots", type=int, default=100)
    parser.add_argument("--radius", type=float, default=90.0)
    return parser.parse_args()


def local_xy(ego, x, y):
    dx = float(x) - float(ego["x"])
    dy = float(y) - float(ego["y"])
    yaw = math.radians(float(ego.get("yaw_deg", 0.0)))
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


def local_yaw(ego, yaw_deg):
    return math.radians(float(yaw_deg) - float(ego.get("yaw_deg", 0.0)))


def draw_box(ax, x, y, heading, length, width, color, linewidth=1.5, alpha=1.0, label=None):
    rect = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        fill=False,
        linewidth=linewidth,
        edgecolor=color,
        alpha=alpha,
        label=label,
    )
    rect.set_transform(Affine2D().rotate(heading).translate(x, y) + ax.transData)
    ax.add_patch(rect)


def plot_payload(payload, path, radius):
    scene = payload.get("scene") or payload
    proposal = payload.get("proposal") or {}
    audit = payload.get("audit_after_generation") or {}
    ego = scene["ego"]
    fig, ax = plt.subplots(figsize=(9, 9))

    for lane in scene.get("map_lanes", []):
        pts = [local_xy(ego, p[0], p[1]) for p in lane.get("centerline", []) if len(p) >= 2]
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="0.85", linewidth=0.8)

    route_pts = []
    for point in scene.get("ego_route", []):
        if isinstance(point, dict):
            route_pts.append(local_xy(ego, point["x"], point["y"]))
        elif len(point) >= 2:
            route_pts.append(local_xy(ego, point[0], point[1]))
    if len(route_pts) >= 2:
        xs, ys = zip(*route_pts)
        ax.plot(xs, ys, color="black", linewidth=2.0, label="ego route")

    draw_box(ax, 0.0, 0.0, 0.0, float(ego.get("length", 5.176)), float(ego.get("width", 2.297)), "tab:blue", 2.0, label="ego")

    for actor in scene.get("agents", []):
        x, y = local_xy(ego, actor["x"], actor["y"])
        draw_box(
            ax,
            x,
            y,
            local_yaw(ego, actor.get("yaw_deg", 0.0)),
            float(actor.get("length", 4.5)),
            float(actor.get("width", 2.0)),
            "0.65",
            linewidth=1.0,
            alpha=0.8,
        )

    generated_states = proposal.get("generated_agent_states") or []
    for state in generated_states:
        draw_box(
            ax,
            float(state[0]),
            float(state[1]),
            float(state[2]),
            float(state[4]),
            float(state[5]),
            "tab:green",
            linewidth=2.0,
            label="generated",
        )

    audit_labels = audit.get("rift_labels", [])
    actor_by_id = {int(a["actor_id"]): a for a in audit.get("agents", []) if "actor_id" in a}
    first_positive = True
    for label in audit_labels:
        if not label.get("is_interactive", False):
            continue
        actor = actor_by_id.get(int(label.get("actor_id", -1)))
        if actor:
            x, y = local_xy(ego, actor["x"], actor["y"])
            draw_box(
                ax,
                x,
                y,
                local_yaw(ego, actor.get("yaw_deg", 0.0)),
                float(actor.get("length", 4.5)),
                float(actor.get("width", 2.0)),
                "tab:red",
                linewidth=2.0,
                label="RIFT-positive" if first_positive else None,
            )
            first_positive = False
        point = label.get("interaction_point")
        if point:
            if isinstance(point, dict):
                px, py = local_xy(ego, point["x"], point["y"])
            else:
                px, py = local_xy(ego, point[0], point[1])
            ax.scatter([px], [py], color="tab:red", marker="x", s=35)

    result = payload.get("result") or {}
    ax.set_title(
        f"{scene.get('scene_id', 'scene')} | gen={len(generated_states)} "
        f"spawn={len(result.get('spawned_actor_ids') or [])} "
        f"rift_pos={result.get('num_rift_positive_agents_after_generation')}"
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.dump_dir.glob("*.json"))[:args.max_plots]
    for idx, path in enumerate(paths):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        out = args.output_dir / f"{idx:04d}_{path.stem}.png"
        plot_payload(payload, out, args.radius)
    print(json.dumps({"output_dir": str(args.output_dir), "num_plots": len(paths)}, indent=2))


if __name__ == "__main__":
    main()

