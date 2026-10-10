"""Online V2 HV generator with selectable CARLA or Scenario Dreamer BV context."""

from high_value_init.high_value_ldm import RouteConditionedHighValueLDM
from high_value_init.scene_generator import _snapshot_to_record


class ScenarioDreamerComposedHighValueSceneGeneratorV2:
    """Condition two V2 HV candidates on CARLA BVs or model-generated BVs."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", config=None):
        config = config or {}
        self.mode = config.get("hv_generation_mode", "sparse_inpainting")
        self.model_mode = "inpainting" if self.mode == "sparse_inpainting" else self.mode
        self.background_source = str(config.get("sd_background_source", "carla"))
        self.background_count = int(config.get("sd_background_num_agents", 6))
        self.hv_count = int(config.get("hv_num_generate_agents", 2))
        if self.background_source not in ("carla", "scenario_dreamer"):
            raise ValueError("sd_background_source must be 'carla' or 'scenario_dreamer'")
        if self.background_count < 0 or self.hv_count != 2:
            raise ValueError("background count must be non-negative and V2 HV generation requires exactly two actors")
        if self.background_source == "scenario_dreamer" and self.background_count <= 0:
            raise ValueError("Scenario Dreamer background mode requires sd_background_num_agents > 0")
        self.actor_radius = float(config.get("hv_actor_radius", 80.0))
        self.fov = float(config.get("hv_sd_fov", 64.0))
        self.autoencoder_checkpoint = config.get("hv_autoencoder_checkpoint")
        self.ldm_checkpoint = config.get("hv_ldm_checkpoint")
        self.route_num_points = int(config.get("hv_route_num_points", config.get("hv_max_route_points", 80)))
        self.generation_iteration = 0
        self.background = None
        if self.background_source == "scenario_dreamer":
            background_checkpoint = config.get("sd_background_checkpoint") or None
            self.background = RouteConditionedHighValueLDM(
                checkpoint_path=background_checkpoint,
                autoencoder_checkpoint=self.autoencoder_checkpoint,
                ldm_checkpoint=self.ldm_checkpoint,
                device=device,
                mode="inpainting",
                num_generate_agents=self.background_count,
                route_num_points=self.route_num_points,
                use_route_condition=bool(background_checkpoint),
                clamp_generated_states=False,
            )
        self.high_value = RouteConditionedHighValueLDM(
            checkpoint_path=checkpoint_path,
            autoencoder_checkpoint=self.autoencoder_checkpoint,
            ldm_checkpoint=self.ldm_checkpoint,
            device=device,
            mode=self.model_mode,
            num_generate_agents=2,
            route_num_points=self.route_num_points,
            use_route_condition=True,
            clamp_generated_states=False,
        )

    @staticmethod
    def _condition(record, appended_states, count):
        record = dict(record)
        condition = dict(record["condition"])
        existing_states = list(condition.get("observed_agent_states") or [])
        existing_types = list(condition.get("observed_agent_types") or [])
        condition["observed_agent_states"] = existing_states + list(appended_states)
        condition["observed_agent_types"] = existing_types + [1 for _ in appended_states]
        condition["num_generate_agents"] = int(count)
        record["condition"] = condition
        return record

    def generate(self, snapshot):
        base = _snapshot_to_record(
            snapshot,
            mode=self.mode,
            num_generate_agents=self.background_count,
            actor_radius=self.actor_radius,
            fov=self.fov,
        )
        # In CARLA mode, snapshot BVs are already the ordinary-BV context. In
        # Scenario Dreamer mode, retain snapshot BVs and append this iteration's
        # generated BVs before producing exactly two HV candidates.
        background_context_count = len(base["condition"].get("observed_agent_states") or [])
        background_states = []
        if self.background is not None:
            background_record = self._condition(base, [], self.background_count)
            background = self.background.generate(background_record, self.background_count)
            background_states = background["generated_agent_states"]
        hv_record = self._condition(base, background_states, self.hv_count)
        high_value = self.high_value.generate(hv_record, self.hv_count)
        high_value_states = high_value["generated_agent_states"]
        self.generation_iteration += 1
        return {
            "schema_version": "scenario_dreamer_composed_hv_v2_online_v1",
            "generation_mode": self.mode,
            "scene_id": snapshot.get("scene_id"),
            "coordinate_frame": "ego_local_rh_x_forward",
            "generated_agent_states": background_states + high_value_states,
            "generated_agent_types": [1 for _ in range(len(background_states) + len(high_value_states))],
            "generated_agent_sources": (
                ["scenario_dreamer_background"] * len(background_states) +
                ["high_value"] * len(high_value_states)
            ),
            "background_generated_count": len(background_states),
            "high_value_generated_count": len(high_value_states),
            "background_source": self.background_source,
            "generation_iteration": int(self.generation_iteration),
            "background_context_count": int(background_context_count),
            "high_value_context_count": int(background_context_count + len(background_states)),
            "rift_labels_used_as_input": False,
            "source": (
                "scenario_dreamer_background_then_v2_hv"
                if self.background_source == "scenario_dreamer"
                else "carla_background_then_v2_hv"
            ),
        }
