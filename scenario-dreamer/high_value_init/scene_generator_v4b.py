"""Online entry for V4B checkpoints; V4A remains immutable."""

import math

import torch

from high_value_init.scene_generator_v4a import ScenarioDreamerHighValueSceneGeneratorV4A


class ScenarioDreamerHighValueSceneGeneratorV4B(ScenarioDreamerHighValueSceneGeneratorV4A):
    """Apply the unchanged raw-state V4 contract to an explicitly V4B checkpoint.

    V4B geometry losses add no inference parameters, so generation correctly
    uses the frozen V3-compatible Scenario Dreamer backbone inherited from
    V4A. This class only protects checkpoint provenance and annotates dumps;
    it does not select, reject, edit, or register online samples.
    """

    _VALID_STAGES = {
        "b0_data_only", "b1_forward_support", "b2_obb_clearance", "b3_joint_geometry",
    }

    def _require_v4a_checkpoint(self, checkpoint_path: str) -> None:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        config = metadata.get("config") if isinstance(metadata, dict) else None
        if not isinstance(metadata, dict) or metadata.get("trainer") != "v4b" or not isinstance(config, dict):
            raise ValueError("V4B online generation requires train_high_value_scene_generator_v4b.py output")
        if int(config.get("num_generate_agents", -1)) != 2:
            raise ValueError("V4B checkpoint does not declare the fixed-two-target contract")
        try:
            trained_forward_min_x = float(config["forward_min_x"])
            trained_margin = float(config["target_fov_margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("V4B checkpoint is missing the forward-sector training contract") from exc
        if not math.isfinite(trained_forward_min_x) or abs(trained_forward_min_x - self.forward_min_x) > 1e-6:
            raise ValueError("V4B online --hv_generation_forward_min_x differs from training")
        if not math.isfinite(trained_margin) or abs(trained_margin - self.target_fov_margin) > 1e-6:
            raise ValueError("V4B online --hv_generation_target_fov_margin differs from training")
        self.v4b_stage = str(config.get("v4b_stage") or "")
        if self.v4b_stage not in self._VALID_STAGES:
            raise ValueError("V4B checkpoint has no recognized isolated ablation stage")
        # The parent records this field in its generic contract diagnostics.
        self.v4a_stage = "v4b:" + self.v4b_stage

    def generate(self, snapshot):
        generated = super().generate(snapshot)
        generated["source"] = "scenario_dreamer_route_conditioned_ldm_v4b_full_fov_forward"
        generated["v4b_checkpoint_stage"] = self.v4b_stage
        return generated
