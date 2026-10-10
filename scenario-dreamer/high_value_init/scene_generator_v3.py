"""V3 RIFT-facing generator entry; v2 entry remains unchanged."""

import torch

from high_value_init.high_value_ldm_v3 import RouteConditionedHighValueLDMV3
from high_value_init.scene_generator import _snapshot_to_record


class ScenarioDreamerHighValueSceneGeneratorV3:
    """Online wrapper for v3 checkpoints with the same raw-state contract."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", config=None):
        config = config or {}
        self.mode = config.get("hv_generation_mode", "sparse_inpainting")
        self.model_mode = "inpainting" if self.mode == "sparse_inpainting" else self.mode
        self.num_generate_agents = int(config.get("hv_num_generate_agents", 2))
        if self.num_generate_agents != 2:
            raise ValueError("v3 online generator is trained and validated for exactly two high-value actors")
        self.actor_radius = float(config.get("hv_actor_radius", 80.0))
        self.fov = float(config.get("hv_sd_fov", 64.0))
        self.snapshot_map_coordinate_mode = str(
            config.get("hv_snapshot_map_coordinate_mode", "legacy")
        )
        if self.snapshot_map_coordinate_mode != "right_handed":
            raise ValueError(
                "V3 HD-map training requires --hv_snapshot_map_coordinate_mode right_handed"
            )
        if not checkpoint_path:
            raise ValueError("--hv_generation_checkpoint is required for v3 online generation")
        self._require_v3_checkpoint(checkpoint_path)
        self.model = RouteConditionedHighValueLDMV3(
            checkpoint_path=checkpoint_path,
            autoencoder_checkpoint=config.get("hv_autoencoder_checkpoint"),
            ldm_checkpoint=config.get("hv_ldm_checkpoint"),
            device=device,
            mode=self.model_mode,
            num_generate_agents=2,
            route_num_points=int(config.get("hv_route_num_points", config.get("hv_max_route_points", 80))),
            use_route_condition=True,
            clamp_generated_states=False,
        )

    @staticmethod
    def _require_v3_checkpoint(checkpoint_path: str) -> None:
        """Reject architecture-compatible checkpoints from a different experiment contract."""
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        train_config = metadata.get("config") if isinstance(metadata, dict) else None
        if not isinstance(metadata, dict) or metadata.get("trainer") != "v3" or not isinstance(train_config, dict):
            raise ValueError(
                "V3 online generation requires a checkpoint written by "
                "train_high_value_scene_generator_v3.py"
            )
        try:
            target_count = int(train_config.get("num_generate_agents", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("V3 checkpoint has an invalid target-count contract") from exc
        if target_count != 2:
            raise ValueError("V3 checkpoint does not declare the fixed-two-target contract")

    def generate(self, snapshot):
        record = _snapshot_to_record(
            snapshot,
            mode=self.mode,
            num_generate_agents=2,
            actor_radius=self.actor_radius,
            fov=self.fov,
        )
        generated = self.model.generate(record, num_generate_agents=2)
        generated.update({
            "scene_id": snapshot.get("scene_id"),
            "coordinate_frame": "ego_local_rh_x_forward",
            "requested_generation_mode": self.mode,
            "model_generation_mode": self.model_mode,
            "rift_labels_used_as_input": False,
            "source": "scenario_dreamer_route_conditioned_ldm_v3",
        })
        return generated
