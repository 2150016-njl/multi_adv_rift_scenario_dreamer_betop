"""Online V4-A entry: full-FOV condition with two forward HV targets.

V4-A reuses the V3 network exactly.  It exposes a distinct online contract so
the rollout cannot accidentally hide context actors, change the requested
target count, silently truncate a full-FOV condition, or apply output-state
postprocessing.
"""

import math

import torch

from high_value_init.high_value_ldm_v3 import RouteConditionedHighValueLDMV3
from high_value_init.scene_generator import _snapshot_to_record


class ScenarioDreamerHighValueSceneGeneratorV4A:
    """RIFT-facing raw V4-A sampler; RIFT remains the only normal CBV route."""

    _MODE = "full_fov_inpainting"

    def __init__(self, checkpoint_path: str, device: str = "cuda", config=None):
        config = config or {}
        self.mode = str(config.get("hv_generation_mode", self._MODE))
        if self.mode != self._MODE:
            raise ValueError(f"V4-A requires --hv_generation_mode {self._MODE}")
        self.num_generate_agents = int(config.get("hv_num_generate_agents", 2))
        if self.num_generate_agents != 2:
            raise ValueError("V4-A is trained and validated for exactly two high-value targets")
        self.fov = float(config.get("hv_sd_fov", 64.0))
        if abs(self.fov - 64.0) > 1e-6:
            raise ValueError("V4-A requires --hv_sd_fov 64 to match training normalization")
        self.snapshot_map_coordinate_mode = str(
            config.get("hv_snapshot_map_coordinate_mode", "legacy")
        )
        if self.snapshot_map_coordinate_mode != "right_handed":
            raise ValueError(
                "V4-A HD-map training requires --hv_snapshot_map_coordinate_mode right_handed"
            )
        self.actor_radius = float(config.get("hv_actor_radius", 80.0))
        self.forward_min_x = float(config.get("hv_generation_forward_min_x", 0.0))
        self.target_fov_margin = float(config.get("hv_generation_target_fov_margin", 2.0))
        self.forward_limit = self.fov / 2.0 - self.target_fov_margin
        if not 0.0 <= self.forward_min_x <= self.forward_limit:
            raise ValueError("V4-A forward target interval is empty or invalid")
        if bool(config.get("hv_generation_clamp_generated_states", False)):
            raise ValueError("V4-A forbids --hv_generation_clamp_generated_states; evaluate raw output")
        if bool(config.get("hv_generation_register_cbv", False)) or bool(
                config.get("hv_generation_register_cbv_from_audit", False)):
            raise ValueError("V4-A forbids generation-time CBV registration; use normal CBVs_recog() only")
        if not checkpoint_path:
            raise ValueError("--hv_generation_checkpoint is required for V4-A online generation")
        self._require_v4a_checkpoint(checkpoint_path)
        self.model = RouteConditionedHighValueLDMV3(
            checkpoint_path=checkpoint_path,
            autoencoder_checkpoint=config.get("hv_autoencoder_checkpoint"),
            ldm_checkpoint=config.get("hv_ldm_checkpoint"),
            device=device,
            mode="inpainting",
            num_generate_agents=2,
            route_num_points=int(config.get("hv_route_num_points", config.get("hv_max_route_points", 80))),
            use_route_condition=True,
            clamp_generated_states=False,
            builder_cfg={"reject_observed_over_capacity": True},
        )
        self.max_num_agents = int(self.model.graph_builder.max_num_agents)
        if self.max_num_agents != 61:
            raise ValueError(f"V4-A expects frozen Scenario Dreamer capacity 61, got {self.max_num_agents}")

    def _require_v4a_checkpoint(self, checkpoint_path: str) -> None:
        """Prevent an architecture-compatible V2/V3 checkpoint being mislabeled V4-A."""
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        config = metadata.get("config") if isinstance(metadata, dict) else None
        if not isinstance(metadata, dict) or metadata.get("trainer") != "v4a" or not isinstance(config, dict):
            raise ValueError("V4-A online generation requires a checkpoint written by train_high_value_scene_generator_v4a.py")
        if int(config.get("num_generate_agents", -1)) != 2:
            raise ValueError("V4-A checkpoint does not declare the fixed-two-target contract")
        try:
            trained_forward_min_x = float(config["forward_min_x"])
            trained_fov_margin = float(config["target_fov_margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("V4-A checkpoint is missing its forward-sector training contract") from exc
        if not math.isfinite(trained_forward_min_x) or abs(trained_forward_min_x - self.forward_min_x) > 1e-6:
            raise ValueError("V4-A online --hv_generation_forward_min_x differs from checkpoint training contract")
        if not math.isfinite(trained_fov_margin) or abs(trained_fov_margin - self.target_fov_margin) > 1e-6:
            raise ValueError("V4-A online --hv_generation_target_fov_margin differs from checkpoint training contract")
        self.v4a_stage = str(config.get("v4a_stage") or "")
        if self.v4a_stage not in {"a0_data_only", "a1_forward_support", "a2_obb_clearance"}:
            raise ValueError("V4-A checkpoint is missing a recognized isolated stage contract")

    def _check_condition_contract(self, record) -> int:
        condition = record.get("condition") or {}
        observed = condition.get("observed_agent_states") or []
        lanes = ((condition.get("map") or {}).get("lane_polylines") or [])
        if not lanes:
            raise ValueError("V4-A refuses an online condition without local map lanes")
        total_agents = 1 + len(observed) + self.num_generate_agents
        if total_agents > self.max_num_agents:
            raise ValueError(
                f"V4-A full-FOV condition has ego + {len(observed)} context + 2 targets = {total_agents}; "
                f"Scenario Dreamer capacity is {self.max_num_agents}. No context truncation is permitted."
            )
        return len(observed)

    def _forward_diagnostics(self, states):
        violations = 0
        for state in states:
            try:
                x, y = float(state[0]), float(state[1])
            except (IndexError, TypeError, ValueError):
                violations += 1
                continue
            valid = (
                math.isfinite(x)
                and math.isfinite(y)
                and self.forward_min_x <= x <= self.forward_limit
                and abs(y) <= self.forward_limit
            )
            violations += int(not valid)
        return {
            "target_sector": "forward",
            "forward_min_x": self.forward_min_x,
            "target_fov_margin": self.target_fov_margin,
            "forward_limit": self.forward_limit,
            "raw_generated_forward_count": len(states) - violations,
            "raw_generated_forward_violation_count": violations,
            "raw_state_postprocessing": "none",
        }

    def generate(self, snapshot):
        record = _snapshot_to_record(
            snapshot,
            mode=self.mode,
            num_generate_agents=2,
            actor_radius=self.actor_radius,
            fov=self.fov,
        )
        observed_count = self._check_condition_contract(record)
        generated = self.model.generate(record, num_generate_agents=2)
        states = generated.get("generated_agent_states") or []
        generated.update({
            "scene_id": snapshot.get("scene_id"),
            "coordinate_frame": "ego_local_rh_x_forward",
            "requested_generation_mode": self.mode,
            "model_generation_mode": "inpainting",
            "rift_labels_used_as_input": False,
            "source": "scenario_dreamer_route_conditioned_ldm_v4a_full_fov_forward",
            "v4a_checkpoint_stage": self.v4a_stage,
            "generated_agent_sources": ["high_value"] * len(states),
            "v4a_condition_contract": {
                "condition_region": "full_fov_non_target",
                "observed_actor_count": observed_count,
                "max_num_agents": self.max_num_agents,
                "agent_capacity_policy": "reject_no_truncation",
            },
            "v4a_raw_output_diagnostics": self._forward_diagnostics(states),
        })
        return generated
