"""Online V4C entry with a strict V4C checkpoint provenance contract.

V4C adds target-only training losses but no network parameters, no online
features, and no sampling-time guidance.  It therefore reuses the V4A raw
sampler while refusing to mislabel V4A/V2/V3 checkpoints as V4C.
"""

import math

import torch

from high_value_init.scene_generator_v4a import ScenarioDreamerHighValueSceneGeneratorV4A


class ScenarioDreamerHighValueSceneGeneratorV4C(ScenarioDreamerHighValueSceneGeneratorV4A):
    """RIFT-facing V4C sampler; RIFT remains the only normal CBV route."""

    def _require_v4a_checkpoint(self, checkpoint_path: str) -> None:
        """Validate V4C's immutable inference and forward-sector contracts."""
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        config = metadata.get("config") if isinstance(metadata, dict) else None
        if not isinstance(metadata, dict) or metadata.get("trainer") != "v4c" or not isinstance(config, dict):
            raise ValueError(
                "V4C online generation requires a checkpoint written by "
                "train_high_value_scene_generator_v4c.py"
            )
        if int(config.get("num_generate_agents", -1)) != 2:
            raise ValueError("V4C checkpoint does not declare the fixed-two-target contract")
        try:
            trained_forward_min_x = float(config["forward_min_x"])
            trained_fov_margin = float(config["target_fov_margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("V4C checkpoint is missing its forward-sector training contract") from exc
        if not math.isfinite(trained_forward_min_x) or abs(trained_forward_min_x - self.forward_min_x) > 1e-6:
            raise ValueError("V4C online --hv_generation_forward_min_x differs from checkpoint training contract")
        if not math.isfinite(trained_fov_margin) or abs(trained_fov_margin - self.target_fov_margin) > 1e-6:
            raise ValueError("V4C online --hv_generation_target_fov_margin differs from checkpoint training contract")
        self.v4c_stage = str(metadata.get("v4c_stage") or config.get("v4c_stage") or "")
        if self.v4c_stage not in {
                "c0_target_metadata_only", "c1_conflict_anchor", "c2_route_progress_sync"}:
            raise ValueError("V4C checkpoint is missing a recognized isolated stage contract")
        self.v4a_stage = self.v4c_stage

    def generate(self, snapshot):
        generated = super().generate(snapshot)
        generated.update({
            "source": "scenario_dreamer_route_conditioned_ldm_v4c_full_fov_forward",
            "v4c_checkpoint_stage": self.v4c_stage,
        })
        generated.pop("v4a_checkpoint_stage", None)
        return generated
