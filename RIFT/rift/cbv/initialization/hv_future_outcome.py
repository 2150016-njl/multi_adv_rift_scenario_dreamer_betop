"""Observation-only future-outcome sidecars for current-time HV anchors.

An anchor retains its *current* raw CARLA state in the normal snapshot.  This
collector writes a separate future label keyed by snapshot path and actor ID.
Future positions are never emitted as generated/spawn states.  In particular,
the record contains no online score, selector, state edit, or CBV action.
"""

import json
import math
from pathlib import Path


def _forward_interactive_actor_ids(snapshot, fov, lower_x, margin):
    ego = snapshot.get("ego") or {}
    actors = {int(row["actor_id"]): row for row in (snapshot.get("agents") or []) if row.get("actor_id") is not None}
    half, limit = float(fov) / 2.0, float(fov) / 2.0 - float(margin)
    selected = []
    for label in snapshot.get("rift_labels") or []:
        if not label.get("is_interactive", False):
            continue
        try:
            actor = actors[int(label["actor_id"])]
            dx, dy = float(actor["x"]) - float(ego["x"]), float(actor["y"]) - float(ego["y"])
            yaw = math.radians(float(ego.get("yaw_deg", 0.0)))
            x, y = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        except (KeyError, TypeError, ValueError):
            continue
        if float(lower_x) <= x <= limit and abs(y) <= limit and abs(x) <= half:
            selected.append(int(actor["actor_id"]))
    return sorted(set(selected))


def _closing_ttc(ego, actor):
    """Euclidean relative-closing TTC proxy, not a route-conflict oracle."""
    try:
        rx, ry = float(actor["x"]) - float(ego["x"]), float(actor["y"]) - float(ego["y"])
        vx, vy = float(actor.get("vx", 0.0)) - float(ego.get("vx", 0.0)), float(actor.get("vy", 0.0)) - float(ego.get("vy", 0.0))
        distance = math.hypot(rx, ry)
        closing = -(rx * vx + ry * vy) / max(distance, 1e-6)
        return distance / closing if closing > 1e-4 else None
    except (KeyError, TypeError, ValueError):
        return None


def _route_corridor_distance(actor, route):
    """Distance to serialized ego-route samples; a transparent proxy only."""
    try:
        points = [(float(point["x"]), float(point["y"])) for point in (route or [])
                  if point.get("x") is not None and point.get("y") is not None]
        if not points:
            return None
        ax, ay = float(actor["x"]), float(actor["y"])
        return min(math.hypot(ax - x, ay - y) for x, y in points)
    except (KeyError, TypeError, ValueError):
        return None


class FutureOutcomeCollector:
    """Accumulate future observations without changing a live rollout."""

    def __init__(self, output_dir, horizon_s, frame_rate, fov=64.0, forward_min_x=0.0,
                 target_fov_margin=2.0, route_conflict_distance_m=3.5):
        self.output_dir = Path(output_dir)
        self.horizon_s = float(horizon_s)
        self.frame_rate = float(frame_rate)
        self.fov, self.forward_min_x, self.target_fov_margin = float(fov), float(forward_min_x), float(target_fov_margin)
        self.route_conflict_distance_m = float(route_conflict_distance_m)
        if self.horizon_s <= 0.0 or self.frame_rate <= 0.0 or self.route_conflict_distance_m <= 0.0:
            raise ValueError("future outcome horizon and frame rate must be positive")
        self.pending = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "future_outcomes.jsonl"

    def anchor(self, snapshot, snapshot_path):
        actor_by_id = {int(row["actor_id"]): row for row in (snapshot.get("agents") or []) if row.get("actor_id") is not None}
        ids = _forward_interactive_actor_ids(snapshot, self.fov, self.forward_min_x, self.target_fov_margin)
        frame = int(snapshot.get("frame_id") or 0)
        for actor_id in ids:
            actor = actor_by_id.get(actor_id)
            if actor is None:
                continue
            self.pending.append({
                "snapshot_path": str(snapshot_path), "scene_id": snapshot.get("scene_id"), "town": snapshot.get("town"),
                "anchor_frame_id": frame, "anchor_actor_id": actor_id,
                # Current state is trace metadata, never a future target.
                "anchor_raw_actor_state": {key: actor.get(key) for key in ("x", "y", "z", "yaw_deg", "vx", "vy", "speed_mps", "length", "width")},
                "normal_cbv_activated": False, "activation_frame_id": None,
                "future_rift_interactive_any": False, "min_relative_closing_ttc_s": None,
                # This is deliberately a route-corridor proximity proxy, not
                # a claim of a counterfactual route conflict or collision.
                "route_corridor_overlap_any": False, "min_distance_to_ego_route_m": None,
                "collision_after_activation": False, "actor_unavailable": False,
            })
        return len(ids)

    def has_pending(self):
        return any(not row.get("written") for row in self.pending)

    def observe(self, snapshot, active_cbv_ids, cbv_collisions):
        frame = int(snapshot.get("frame_id") or 0)
        actors = {int(row["actor_id"]): row for row in (snapshot.get("agents") or []) if row.get("actor_id") is not None}
        future_interactive = {int(row["actor_id"]) for row in (snapshot.get("rift_labels") or [])
                              if row.get("is_interactive", False) and row.get("actor_id") is not None}
        active_cbv_ids = {int(value) for value in (active_cbv_ids or [])}
        ego = snapshot.get("ego") or {}
        for row in self.pending:
            if row.get("written"):
                continue
            actor_id = int(row["anchor_actor_id"])
            actor = actors.get(actor_id)
            if actor is None:
                if actor_id not in active_cbv_ids:
                    row["actor_unavailable"] = True
            else:
                row["future_rift_interactive_any"] |= actor_id in future_interactive
                ttc = _closing_ttc(ego, actor)
                if ttc is not None:
                    before = row.get("min_relative_closing_ttc_s")
                    row["min_relative_closing_ttc_s"] = ttc if before is None else min(before, ttc)
                route_distance = _route_corridor_distance(actor, snapshot.get("ego_route"))
                if route_distance is not None:
                    before = row.get("min_distance_to_ego_route_m")
                    row["min_distance_to_ego_route_m"] = route_distance if before is None else min(before, route_distance)
                    row["route_corridor_overlap_any"] |= route_distance <= self.route_conflict_distance_m
            if actor_id in active_cbv_ids and not row["normal_cbv_activated"]:
                row["normal_cbv_activated"], row["activation_frame_id"] = True, frame
            if row["normal_cbv_activated"] and cbv_collisions and cbv_collisions.get(actor_id) is not None:
                row["collision_after_activation"] = True
            if (frame - int(row["anchor_frame_id"])) / self.frame_rate >= self.horizon_s:
                self._write(row, terminal_reason="horizon_complete")

    def finalize_episode(self, terminal_reason="episode_finished"):
        for row in self.pending:
            if not row.get("written"):
                self._write(row, terminal_reason=terminal_reason)

    def _write(self, row, terminal_reason):
        row["written"] = True
        row["schema_version"] = "hv_future_outcome_v1"
        row["task"] = "future_outcome_supervision_for_current_time_anchor_only"
        row["terminal_reason"] = str(terminal_reason)
        row["horizon_completed"] = terminal_reason == "horizon_complete"
        row["activation_delay_s"] = (
            None if row["activation_frame_id"] is None else
            (int(row["activation_frame_id"]) - int(row["anchor_frame_id"])) / self.frame_rate
        )
        # Causality needs matched interventions; do not mislabel one rollout's
        # route completion as an actor-specific completion impact.
        row["counterfactual_ego_completion_impact_available"] = False
        row["future_positions_used_as_spawn_state"] = False
        row["route_corridor_overlap_definition"] = (
            f"observed actor distance to serialized ego-route samples <= {self.route_conflict_distance_m:g}m; "
            "proxy only, not a counterfactual route-conflict label"
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
