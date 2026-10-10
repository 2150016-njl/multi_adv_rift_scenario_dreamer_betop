"""V4-A geometry losses on the unchanged V3 Scenario Dreamer architecture.

This module adds losses only.  It does not add input features, output heads,
topology embeddings, state postprocessing, or an online sample selector.
"""

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from cfgs.config import BEFORE_PARTITION
from high_value_init.high_value_ldm_v3 import RouteConditionedHighValueLDMV3


class RouteConditionedHighValueLDMV4A(RouteConditionedHighValueLDMV3):
    """V3 LDM plus optional V4-A forward-support and OBB-SAT losses."""

    def __init__(
            self,
            *args,
            forward_support_loss_weight: float = 0.0,
            obb_clearance_loss_weight: float = 0.0,
            forward_min_x: float = 0.0,
            target_fov_margin: float = 2.0,
            min_obb_axis_clearance: float = 0.5,
            heading_unit_loss_weight: float = 0.0,
            speed_limit_loss_weight: float = 0.0,
            min_generated_speed: float = 0.0,
            max_generated_speed: float = 30.0,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_support_loss_weight = float(forward_support_loss_weight)
        self.obb_clearance_loss_weight = float(obb_clearance_loss_weight)
        self.forward_min_x = float(forward_min_x)
        self.target_fov_margin = float(target_fov_margin)
        self.min_obb_axis_clearance = float(min_obb_axis_clearance)
        self.heading_unit_loss_weight = float(heading_unit_loss_weight)
        self.speed_limit_loss_weight = float(speed_limit_loss_weight)
        self.min_generated_speed = float(min_generated_speed)
        self.max_generated_speed = float(max_generated_speed)
        fov = float(self.cfg_ldm.dataset.fov)
        self.forward_limit = fov / 2.0 - self.target_fov_margin
        if not 0.0 <= self.forward_min_x <= self.forward_limit:
            raise ValueError("V4-A forward/FOV support interval is empty or invalid")
        if self.min_obb_axis_clearance < 0.0:
            raise ValueError("min_obb_axis_clearance must be non-negative")
        if self.heading_unit_loss_weight < 0.0 or self.speed_limit_loss_weight < 0.0:
            raise ValueError("heading/speed auxiliary loss weights must be non-negative")
        if self.min_generated_speed > self.max_generated_speed:
            raise ValueError("generated speed interval is invalid")

    def training_loss(self, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.build_training_batch(records)
        loss_dict = self.ldm.loss(batch)
        if self.decoded_state_loss_weight <= 0.0:
            return loss_dict
        decoded = self._decoded_state_loss_v4a(batch, records)
        loss_dict["loss"] = loss_dict["loss"] + self.decoded_state_loss_weight * decoded["decoded_state_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.lane_corridor_loss_weight * decoded["lane_corridor_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.lane_heading_loss_weight * decoded["lane_heading_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.initial_clearance_loss_weight * decoded["initial_clearance_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.forward_support_loss_weight * decoded["forward_support_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.obb_clearance_loss_weight * decoded["obb_clearance_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.heading_unit_loss_weight * decoded["heading_unit_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.speed_limit_loss_weight * decoded["speed_limit_loss"]
        loss_dict.update(decoded)
        return loss_dict

    def _decoded_state_loss_v4a(self, data, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # This is the V3 decoded branch with two additional losses computed
        # from the same decoded prediction.  There are no indexed in-place
        # writes in this autograd path.
        x_agent = data["agent"].latents.unsqueeze(1)
        x_lane = data["lane"].latents.unsqueeze(1)
        agent_batch, lane_batch = data["agent"].batch, data["lane"].batch
        timesteps = torch.randint(0, self.ldm.n_timesteps, (data.batch_size,), device=x_agent.device).long()
        t_agent, t_lane = timesteps[agent_batch], timesteps[lane_batch]
        agent_noise, lane_noise = torch.randn_like(x_agent), torch.randn_like(x_lane)
        x_agent_noisy = self.ldm.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)
        x_lane_noisy = self.ldm.q_sample(x_start=x_lane, t=t_lane, noise=lane_noise)
        agent_condition_mask = data["agent"].partition_mask == BEFORE_PARTITION
        lane_condition_mask = data["lane"].partition_mask == BEFORE_PARTITION
        x_agent_noisy = torch.where(agent_condition_mask.view(-1, 1, 1), x_agent, x_agent_noisy)
        x_lane_noisy = torch.where(lane_condition_mask.view(-1, 1, 1), x_lane, x_lane_noisy)
        agent_noise_pred, lane_noise_pred = self.ldm.model(x_agent_noisy, x_lane_noisy, data, t_agent, t_lane)
        pred_agent_latents = self.ldm.predict_start_from_noise(x_agent_noisy, t=t_agent, noise=agent_noise_pred)[:, 0].clone()
        pred_lane_latents = self.ldm.predict_start_from_noise(x_lane_noisy, t=t_lane, noise=lane_noise_pred)[:, 0].clone()
        pred_agent_latents = torch.where(agent_condition_mask.view(-1, 1), data["agent"].latents, pred_agent_latents)
        pred_lane_latents = torch.where(lane_condition_mask.view(-1, 1), data["lane"].latents, pred_lane_latents)
        agent_latents, lane_latents = self._unnormalize_latents(pred_agent_latents, pred_lane_latents)
        agent_samples, _lane_samples, _, _, _ = self.autoencoder.forward_decoder(agent_latents, lane_latents, data)
        state_metrics = self._set_matched_state_loss(agent_samples, data)
        raw_agents_v3 = self._unnormalize_agents_for_geometry(agent_samples)
        topology_metrics = self._topology_losses(raw_agents_v3, data, records)
        raw_agents = self._unnormalize_agents_for_v4a_losses(agent_samples)
        extra_metrics = self._v4a_geometry_losses(raw_agents, data, records)
        return {**state_metrics, **topology_metrics, **extra_metrics}

    def _unnormalize_agents_for_v4a_losses(self, agent_samples: torch.Tensor) -> torch.Tensor:
        """Unnormalize without clipping x/y so support violations retain gradients."""
        dataset = self.cfg_ldm.dataset
        fov = float(dataset.fov)
        min_speed, max_speed = float(dataset.min_speed), float(dataset.max_speed)
        min_length, max_length = float(dataset.min_length), float(dataset.max_length)
        min_width, max_width = float(dataset.min_width), float(dataset.max_width)
        return torch.stack((
            ((agent_samples[:, 0] + 1.0) / 2.0) * fov - fov / 2.0,
            ((agent_samples[:, 1] + 1.0) / 2.0) * fov - fov / 2.0,
            ((agent_samples[:, 2] + 1.0) / 2.0) * (max_speed - min_speed) + min_speed,
            agent_samples[:, 3], agent_samples[:, 4],
            ((agent_samples[:, 5] + 1.0) / 2.0) * (max_length - min_length) + min_length,
            ((agent_samples[:, 6] + 1.0) / 2.0) * (max_width - min_width) + min_width,
        ), dim=1)

    @staticmethod
    def _state_to_obb(state: torch.Tensor):
        heading_vector = state[3:5]
        unit_heading = heading_vector / torch.linalg.norm(heading_vector).clamp_min(1e-5)
        lateral = torch.stack((-unit_heading[1], unit_heading[0]))
        # Physical-bound loss remains responsible for raw size validity.  The
        # smooth positive proxy here only prevents undefined SAT radii.
        half_length = F.softplus(state[5]) / 2.0
        half_width = F.softplus(state[6]) / 2.0
        return state[:2], unit_heading, lateral, half_length, half_width

    @classmethod
    def _sat_signed_axis_clearance(cls, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_center, left_u, left_v, left_l, left_w = cls._state_to_obb(left)
        right_center, right_u, right_v, right_l, right_w = cls._state_to_obb(right)
        delta = right_center - left_center
        gaps = []
        for axis in (left_u, left_v, right_u, right_v):
            left_radius = left_l * torch.abs(torch.dot(axis, left_u)) + left_w * torch.abs(torch.dot(axis, left_v))
            right_radius = right_l * torch.abs(torch.dot(axis, right_u)) + right_w * torch.abs(torch.dot(axis, right_v))
            gaps.append(torch.abs(torch.dot(delta, axis)) - left_radius - right_radius)
        return torch.stack(gaps).max()

    @staticmethod
    def _external_state_to_raw(state, dtype, device) -> torch.Tensor:
        values = torch.tensor(state[:6], dtype=dtype, device=device)
        yaw = values[2]
        return torch.stack((values[0], values[1], values[3], torch.cos(yaw), torch.sin(yaw), values[4], values[5]))

    def _v4a_geometry_losses(self, raw_agents: torch.Tensor, data, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        target_mask, batch = data["agent"].target_state_mask, data["agent"].batch
        support_losses: List[torch.Tensor] = []
        obb_losses: List[torch.Tensor] = []
        heading_unit_losses: List[torch.Tensor] = []
        speed_limit_losses: List[torch.Tensor] = []
        for scene_idx, record in enumerate(records):
            pred = raw_agents[target_mask & (batch == scene_idx)]
            if pred.numel() == 0:
                continue
            if self.forward_support_loss_weight > 0.0:
                support_losses.append((
                    torch.relu(self.forward_min_x - pred[:, 0]) +
                    torch.relu(pred[:, 0] - self.forward_limit) +
                    torch.relu(torch.abs(pred[:, 1]) - self.forward_limit)
                ).mean())
            if self.heading_unit_loss_weight > 0.0:
                heading_unit_losses.append(((pred[:, 3] ** 2 + pred[:, 4] ** 2 - 1.0) ** 2).mean())
            if self.speed_limit_loss_weight > 0.0:
                speed_limit_losses.append((
                    torch.relu(self.min_generated_speed - pred[:, 2]) +
                    torch.relu(pred[:, 2] - self.max_generated_speed)
                ).mean())
            if self.obb_clearance_loss_weight > 0.0:
                condition = record.get("condition") or {}
                externals = [condition.get("ego_state") or [0.0, 0.0, 0.0, 0.0, 5.176, 2.297, 1.0]]
                externals.extend(condition.get("observed_agent_states") or [])
                for item in externals:
                    if not isinstance(item, (list, tuple)) or len(item) < 6:
                        continue
                    other = self._external_state_to_raw(item, pred.dtype, pred.device)
                    for state in pred:
                        clearance = self._sat_signed_axis_clearance(state, other)
                        obb_losses.append(torch.relu(self.min_obb_axis_clearance - clearance))
                for left in range(pred.shape[0]):
                    for right in range(left + 1, pred.shape[0]):
                        clearance = self._sat_signed_axis_clearance(pred[left], pred[right])
                        obb_losses.append(torch.relu(self.min_obb_axis_clearance - clearance))
        zero = raw_agents.sum() * 0.0
        return {
            "forward_support_loss": torch.stack(support_losses).mean() if support_losses else zero,
            "obb_clearance_loss": torch.stack(obb_losses).mean() if obb_losses else zero,
            "heading_unit_loss": torch.stack(heading_unit_losses).mean() if heading_unit_losses else zero,
            "speed_limit_loss": torch.stack(speed_limit_losses).mean() if speed_limit_losses else zero,
        }
