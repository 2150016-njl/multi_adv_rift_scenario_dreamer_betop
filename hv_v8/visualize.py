from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .common import read_json


def plot_record(report_path: Path, scene_path: Path, out_png: Path) -> None:
    report = read_json(report_path)
    scene = read_json(scene_path)
    ego = scene["ego"]
    yaw = __import__("math").radians(float(ego["yaw_deg"]))
    c, s = __import__("math").cos(yaw), __import__("math").sin(yaw)

    def local(x, y):
        dx, dy = x - float(ego["x"]), y - float(ego["y"])
        return c * dx + s * dy, -s * dx + c * dy

    fig, ax = plt.subplots(figsize=(11, 9))
    route_xy = [local(float(p["x"]), float(p["y"])) for p in scene["ego_route"]]
    if route_xy:
        ax.plot([p[0] for p in route_xy], [p[1] for p in route_xy], linewidth=2, label="ego future route")

    for a in report["approaches"]:
        pts = a.get("centerline_samples_local", [])
        if pts:
            xs, ys = zip(*[(p[1], p[2]) for p in pts])
            ax.plot(xs, ys, linewidth=1, alpha=0.35)

    gt = report.get("gt_audit", {})
    for row in gt.get("rows", []):
        actor = next((a for a in scene.get("actors", []) if a.get("actor_id") == row.get("actor_id")), None)
        if actor is None:
            continue
        x, y = local(float(actor["x"]), float(actor["y"]))
        if row.get("mapped_to_approach"):
            ax.scatter([x], [y], marker="o", s=55, label="GT mapped")
        else:
            ax.scatter([x], [y], marker="x", s=70, label="GT UNMAPPED")

    ax.scatter([0], [0], marker="*", s=130, label="ego")
    ax.axvspan(0, 32, alpha=0.04, label="slot0 x>0")
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ego-local x [m], forward")
    ax.set_ylabel("ego-local y [m], right")
    ax.set_title(
        f"P2 support audit | record={report['record_index']} | "
        f"approaches={report['approach_count']} | "
        f"recall={gt.get('support_recall_over_native_rows')}"
    )
    # De-duplicate legend labels.
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    plot_record(Path(args.report), Path(args.scene), Path(args.output))


if __name__ == "__main__":
    main()
