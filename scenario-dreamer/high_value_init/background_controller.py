"""Count-only ordinary-BV replenishment policy.

This module deliberately has no CARLA import and cannot spawn, destroy,
snap, clamp, rank, or register actors.  It decides only how many model
background slots are missing.  The caller must supply the count of *live
model-originated* background actors; CARLA traffic and HV actors never count
toward that inventory.
"""

import json
import math
import random
from pathlib import Path


class BackgroundCountModel:
    """Read train-only empirical ``P(N_after | map, route, N_before)``."""

    _SCHEMA = "scenario_dreamer_background_count_model_v1"

    def __init__(self, path):
        self.path = str(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != self._SCHEMA:
            raise ValueError("not a background count model v1")
        if payload.get("estimated_from_split") != "train_only":
            raise ValueError("background count model must be estimated from train split only")
        if payload.get("task") != "current_front_scene_completion_not_future_prediction":
            raise ValueError("background count model task is not current-front scene completion")
        self.max_target_agents = int(payload.get("max_target_agents", 0))
        if self.max_target_agents < 1:
            raise ValueError("background count model has invalid max_target_agents")
        self.global_pmf = self._validate_pmf(payload.get("global_pmf"), "global_pmf")
        self.conditional_pmf = {
            str(key): self._validate_pmf(value, f"conditional_pmf[{key}]")
            for key, value in (payload.get("conditional_pmf") or {}).items()
        }

    def _validate_pmf(self, pmf, name):
        if not isinstance(pmf, dict):
            raise ValueError(f"{name} must be a dictionary")
        values = {}
        for count in range(1, self.max_target_agents + 1):
            value = float(pmf.get(str(count), 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} has invalid probability")
            values[count] = value
        total = sum(values.values())
        if total <= 0.0:
            raise ValueError(f"{name} has zero total probability")
        return {count: value / total for count, value in values.items()}

    def _key(self, map_bucket, route_bucket, num_before):
        return f"{str(map_bucket)}|{str(route_bucket)}|{min(max(int(num_before), 0), 20)}"

    def pmf(self, map_bucket, route_bucket, num_before):
        key = self._key(map_bucket, route_bucket, num_before)
        return self.conditional_pmf.get(key, self.global_pmf), key in self.conditional_pmf

    def target_count(self, map_bucket, route_bucket, num_before, seed=None):
        pmf, conditional = self.pmf(map_bucket, route_bucket, num_before)
        rng = random.Random(seed) if seed is not None else random
        draw, cumulative = rng.random(), 0.0
        for count in sorted(pmf):
            cumulative += pmf[count]
            if draw <= cumulative:
                return count, conditional
        return max(pmf), conditional


class BackgroundReplenishmentController:
    """Compute a bounded ``K_missing`` without touching simulation state."""

    def __init__(self, count_model_path, graph_capacity=61):
        self.count_model = BackgroundCountModel(count_model_path)
        self.graph_capacity = int(graph_capacity)
        if self.graph_capacity != 61:
            raise ValueError("background controller freezes Scenario Dreamer graph capacity at 61")

    def plan(self, *, map_bucket, route_bucket, num_before, live_model_background, non_target_context_count, seed=None):
        """Return an observationally auditable replenishment decision.

        ``non_target_context_count`` includes ego plus all model/CARLA context
        that the actual generator will condition on.  If the requested count
        cannot fit the frozen graph, the plan rejects the event rather than
        truncating context or silently reducing the generated count.
        """
        live = max(0, int(live_model_background))
        context = max(1, int(non_target_context_count))
        desired, conditional = self.count_model.target_count(map_bucket, route_bucket, num_before, seed=seed)
        missing = max(0, int(desired) - live)
        capacity_remaining = self.graph_capacity - context
        rejection = None
        if desired > self.count_model.max_target_agents:
            rejection = "desired_count_exceeds_model_training_max"
        elif missing > capacity_remaining:
            rejection = "graph_capacity_rejected_no_context_truncation"
        return {
            "schema_version": "scenario_dreamer_background_replenishment_plan_v1",
            "task": "current_front_scene_completion_not_future_prediction",
            "map_bucket": str(map_bucket),
            "route_bucket": str(route_bucket),
            "num_before": int(num_before),
            "live_model_background": live,
            "desired_model_background": int(desired),
            "k_missing": int(missing),
            "count_pmf_was_conditional": bool(conditional),
            "non_target_context_count": context,
            "graph_capacity": self.graph_capacity,
            "capacity_remaining": capacity_remaining,
            "accepted": rejection is None,
            "rejection_reason": rejection,
            "side_effects": "none_count_only",
        }

