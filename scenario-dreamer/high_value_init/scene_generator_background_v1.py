"""Online ordinary-BV generator for the independent directional task.

This entry is intentionally not wired into the legacy HV initializer yet.
That initializer's audit/gating semantics are for fixed-two HV candidates.
Use this class only through the future dedicated background lifecycle hook,
after the raw feasibility/spawn/TM gate has passed.
"""

import json
import math

import build_high_value_scene_dataset_v3 as v3
from build_scenario_dreamer_background_dataset import normalize_raw_snapshot
import torch

from high_value_init.background_controller import BackgroundReplenishmentController
from high_value_init.background_ldm import RouteConditionedBackgroundLDM


class ScenarioDreamerBackgroundSceneGeneratorV1:
    """Raw rear-condition -> current-front-BV sampler with ``K_missing`` only."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", config=None):
        config = config or {}
        self.fov = float(config.get("background_fov", 64.0))
        self.boundary = float(config.get("background_partition_boundary_x", 0.0))
        self.margin = float(config.get("background_target_fov_margin", 2.0))
        self.actor_radius = float(config.get("background_actor_radius", 80.0))
        self.max_num_agents = int(config.get("background_max_num_agents", 61))
        if abs(self.fov - 64.0) > 1e-6 or self.max_num_agents != 61 or abs(self.boundary) > 1e-6:
            raise ValueError("background v1 requires frozen fov=64, capacity=61, and x=0 directional partition")
        if bool(config.get("hv_generation_clamp_generated_states", False)):
            raise ValueError("background v1 forbids online clamping")
        if not checkpoint_path or not config.get("background_count_model"):
            raise ValueError("background v1 requires a trained checkpoint and --background_count_model")
        self._require_background_checkpoint(checkpoint_path)
        self.controller = BackgroundReplenishmentController(config["background_count_model"], graph_capacity=self.max_num_agents)
        self.model = RouteConditionedBackgroundLDM(
            checkpoint_path=checkpoint_path, autoencoder_checkpoint=config.get("hv_autoencoder_checkpoint"),
            ldm_checkpoint=config.get("hv_ldm_checkpoint"), device=device, mode="inpainting", num_generate_agents=1,
            route_num_points=int(config.get("hv_route_num_points", config.get("hv_max_route_points", 80))),
            builder_cfg={"reject_observed_over_capacity": True}, clamp_generated_states=False,
        )

    @staticmethod
    def _require_background_checkpoint(path):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        config = metadata.get("config") if isinstance(metadata, dict) else None
        if not isinstance(metadata, dict) or metadata.get("trainer") != "scenario_dreamer_background" or not isinstance(config, dict):
            raise ValueError("background v1 requires a checkpoint written by train_scenario_dreamer_background.py")
        if config.get("task") != "current_front_scene_completion_not_future_prediction":
            raise ValueError("background checkpoint has the wrong task contract")

    def _record(self, snapshot):
        # The live snapshot's optional RIFT labels are intentionally ignored.
        norm = normalize_raw_snapshot(snapshot, actor_radius=self.actor_radius, fov=self.fov)
        rear = [row for row in norm["agents"] if float(row["state"][0]) < self.boundary]
        lanes = v3.local_lanes(snapshot, self.fov)
        if not lanes:
            raise ValueError("background v1 refuses a snapshot without local map lanes")
        route_bucket = "route_unknown"
        points = (norm.get("ego_route") or {}).get("polyline") or []
        if len(points) >= 2:
            route_bucket = "route_straight"  # Same compact metadata bucket as the dataset; it is not a new model input.
        return {
            "schema_version": "scenario_dreamer_background_directional_online_v1",
            "scene_id": norm["scene_id"],
            "condition": {"ego_state": norm["ego_state"], "fov": self.fov, "ego_route": norm["ego_route"],
                          "map": {"lane_polylines": norm["map"]["lane_polylines"], "lanes_v3": lanes},
                          "observed_agent_states": [row["state"] for row in rear], "observed_agent_types": [1 for _ in rear],
                          "num_generate_agents": 0},
            "count_features": {"map_bucket": v3.topology_bucket(lanes), "route_bucket": route_bucket, "num_before": len(rear)},
        }

    def generate(self, snapshot):
        record = self._record(snapshot)
        live = int((snapshot.get("extra") or {}).get("live_model_background_count", 0))
        seed = (snapshot.get("extra") or {}).get("background_count_seed")
        plan = self.controller.plan(map_bucket=record["count_features"]["map_bucket"], route_bucket=record["count_features"]["route_bucket"],
                                    num_before=record["count_features"]["num_before"], live_model_background=live,
                                    non_target_context_count=1 + len(record["condition"]["observed_agent_states"]), seed=seed)
        if not plan["accepted"] or plan["k_missing"] == 0:
            return {"schema_version": "scenario_dreamer_background_online_v1", "generated_agent_states": [], "generated_agent_types": [],
                    "generated_agent_sources": [], "background_replenishment_plan": plan, "raw_state_postprocessing": "none"}
        record["condition"]["num_generate_agents"] = int(plan["k_missing"])
        generated = self.model.generate(record, int(plan["k_missing"]))
        states = generated.get("generated_agent_states") or []
        limit = self.fov / 2.0 - self.margin
        violations = sum(int(not (math.isfinite(float(s[0])) and math.isfinite(float(s[1])) and self.boundary <= float(s[0]) <= limit and abs(float(s[1])) <= limit)) for s in states)
        generated.update({"schema_version": "scenario_dreamer_background_online_v1", "source": "scenario_dreamer_background_directional_v1",
                          "generated_agent_sources": ["scenario_dreamer_background"] * len(states), "background_replenishment_plan": plan,
                          "raw_state_postprocessing": "none", "raw_generated_forward_violation_count": violations,
                          "rift_labels_used_as_input": False})
        return generated

