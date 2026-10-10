"""V3 map-aware losses layered on the unchanged Scenario Dreamer LDM."""

import math
from typing import Dict, List, Sequence

import torch

from cfgs.config import BEFORE_PARTITION
from high_value_init.high_value_ldm import RouteConditionedHighValueLDM


class RouteConditionedHighValueLDMV3(RouteConditionedHighValueLDM):
    """Adds raw-state map and clearance losses without changing v2 modules."""

    def __init__(
            self,
            *args,
            lane_corridor_loss_weight: float = 0.0,
            lane_heading_loss_weight: float = 0.0,
            initial_clearance_loss_weight: float = 0.0,
            default_lane_half_width: float = 1.75,
            min_initial_clearance: float = 2.5,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.lane_corridor_loss_weight = float(lane_corridor_loss_weight)
        self.lane_heading_loss_weight = float(lane_heading_loss_weight)
        self.initial_clearance_loss_weight = float(initial_clearance_loss_weight)
        self.default_lane_half_width = float(default_lane_half_width)
        self.min_initial_clearance = float(min_initial_clearance)

    def training_loss(self, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.build_training_batch(records)
        loss_dict = self.ldm.loss(batch)
        if self.decoded_state_loss_weight <= 0.0:
            return loss_dict
        decoded = self._decoded_state_loss_v3(batch, records)
        loss_dict["loss"] = loss_dict["loss"] + self.decoded_state_loss_weight * decoded["decoded_state_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.lane_corridor_loss_weight * decoded["lane_corridor_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.lane_heading_loss_weight * decoded["lane_heading_loss"]
        loss_dict["loss"] = loss_dict["loss"] + self.initial_clearance_loss_weight * decoded["initial_clearance_loss"]
        loss_dict.update(decoded)
        return loss_dict

    def _decoded_state_loss_v3(self, data, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        x_agent = data["agent"].latents.unsqueeze(1)
        x_lane = data["lane"].latents.unsqueeze(1)
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        timesteps = torch.randint(0, self.ldm.n_timesteps, (data.batch_size,), device=x_agent.device).long()
        t_agent = timesteps[agent_batch]
        t_lane = timesteps[lane_batch]
        agent_noise = torch.randn_like(x_agent)
        lane_noise = torch.randn_like(x_lane)
        x_agent_noisy = self.ldm.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)
        x_lane_noisy = self.ldm.q_sample(x_start=x_lane, t=t_lane, noise=lane_noise)
        agent_condition_mask = data["agent"].partition_mask == BEFORE_PARTITION
        lane_condition_mask = data["lane"].partition_mask == BEFORE_PARTITION
        # Preserve conditioning latents without mutating tensors retained by
        # q_sample/predict_start_from_noise for the backward pass.
        x_agent_noisy = torch.where(
            agent_condition_mask.view(-1, 1, 1), x_agent, x_agent_noisy,
        )
        x_lane_noisy = torch.where(
            lane_condition_mask.view(-1, 1, 1), x_lane, x_lane_noisy,
        )
        agent_noise_pred, lane_noise_pred = self.ldm.model(x_agent_noisy, x_lane_noisy, data, t_agent, t_lane)
        pred_agent_latents = self.ldm.predict_start_from_noise(x_agent_noisy, t=t_agent, noise=agent_noise_pred)[:, 0].clone()
        pred_lane_latents = self.ldm.predict_start_from_noise(x_lane_noisy, t=t_lane, noise=lane_noise_pred)[:, 0].clone()
        pred_agent_latents = torch.where(
            agent_condition_mask.view(-1, 1),
            data["agent"].latents,
            pred_agent_latents,
        )
        pred_lane_latents = torch.where(
            lane_condition_mask.view(-1, 1),
            data["lane"].latents,
            pred_lane_latents,
        )
        agent_latents, lane_latents = self._unnormalize_latents(pred_agent_latents, pred_lane_latents)
        agent_samples, lane_samples, _, _, _ = self.autoencoder.forward_decoder(agent_latents, lane_latents, data)
        state_metrics = self._set_matched_state_loss(agent_samples, data)
        raw_agents = self._unnormalize_agents_for_geometry(agent_samples)
        topology_metrics = self._topology_losses(raw_agents, data, records)
        return {**state_metrics, **topology_metrics}

    def _unnormalize_agents_for_geometry(self, agent_samples: torch.Tensor) -> torch.Tensor:
        """Functional agent-state inverse scaling for differentiable V3 geometry losses."""
        dataset = self.cfg_ldm.dataset
        fov = float(dataset.fov)
        min_speed, max_speed = float(dataset.min_speed), float(dataset.max_speed)
        min_length, max_length = float(dataset.min_length), float(dataset.max_length)
        min_width, max_width = float(dataset.min_width), float(dataset.max_width)
        clipped = lambda index: torch.clamp(agent_samples[:, index], -1.0, 1.0)
        return torch.stack((
            ((clipped(0) + 1.0) / 2.0) * fov - fov / 2.0,
            ((clipped(1) + 1.0) / 2.0) * fov - fov / 2.0,
            ((clipped(2) + 1.0) / 2.0) * (max_speed - min_speed) + min_speed,
            clipped(3),
            clipped(4),
            ((clipped(5) + 1.0) / 2.0) * (max_length - min_length) + min_length,
            ((clipped(6) + 1.0) / 2.0) * (max_width - min_width) + min_width,
        ), dim=1)

    def _topology_losses(self, raw_agents, data, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        corridor_losses: List[torch.Tensor] = []
        heading_losses: List[torch.Tensor] = []
        clearance_losses: List[torch.Tensor] = []
        target_mask = data["agent"].target_state_mask
        batch = data["agent"].batch
        for scene_idx, record in enumerate(records):
            mask = target_mask & (batch == scene_idx)
            pred = raw_agents[mask]
            if pred.numel() == 0:
                continue
            target = torch.tensor(
                (record.get("target") or {}).get("generated_agent_states") or [],
                dtype=pred.dtype,
                device=pred.device,
            )
            if target.numel() == 0:
                continue
            pred = self._match_prediction_order(pred, target[:, :2])
            lanes = ((record.get("condition") or {}).get("map") or {}).get("lanes_v3") or []
            associations = (record.get("target") or {}).get("map_associations_v3") or []
            for idx in range(min(pred.shape[0], len(associations))):
                association = associations[idx] or {}
                lane_idx = association.get("lane_index")
                if lane_idx is None or int(lane_idx) < 0 or int(lane_idx) >= len(lanes):
                    continue
                points = torch.tensor(lanes[int(lane_idx)].get("centerline") or [], dtype=pred.dtype, device=pred.device)
                if points.ndim != 2 or points.shape[0] < 2:
                    continue
                segment_start = points[:-1, :2]
                segment_delta = points[1:, :2] - segment_start
                segment_norm_sq = (segment_delta * segment_delta).sum(dim=-1).clamp_min(1e-9)
                point_delta = pred[idx, :2].unsqueeze(0) - segment_start
                segment_ratio = (point_delta * segment_delta).sum(dim=-1) / segment_norm_sq
                segment_ratio = segment_ratio.clamp(0.0, 1.0)
                closest = segment_start + segment_ratio.unsqueeze(-1) * segment_delta
                distances = torch.linalg.norm(pred[idx, :2].unsqueeze(0) - closest, dim=-1)
                nearest_segment = int(torch.argmin(distances).detach().cpu())
                distance = distances[nearest_segment]
                half_width = float(association.get("lane_half_width", self.default_lane_half_width))
                corridor_losses.append(torch.relu(distance - half_width))
                tangent_vector = segment_delta[nearest_segment]
                tangent = torch.atan2(tangent_vector[1], tangent_vector[0])
                heading = torch.atan2(pred[idx, 4], pred[idx, 3])
                heading_losses.append(torch.abs(torch.atan2(torch.sin(heading - tangent), torch.cos(heading - tangent))))
            ego_distance = torch.linalg.norm(pred[:, :2], dim=-1)
            clearance_losses.append(torch.relu(self.min_initial_clearance - ego_distance).mean())
            observed = (record.get("condition") or {}).get("observed_agent_states") or []
            if observed:
                observed_xy = torch.tensor([row[:2] for row in observed], dtype=pred.dtype, device=pred.device)
                nearest = torch.cdist(pred[:, :2], observed_xy).min(dim=1).values
                clearance_losses.append(torch.relu(self.min_initial_clearance - nearest).mean())
            if pred.shape[0] > 1:
                pair = torch.pdist(pred[:, :2])
                clearance_losses.append(torch.relu(self.min_initial_clearance - pair).mean())
        zero = raw_agents.sum() * 0.0
        return {
            "lane_corridor_loss": torch.stack(corridor_losses).mean() if corridor_losses else zero,
            "lane_heading_loss": torch.stack(heading_losses).mean() if heading_losses else zero,
            "initial_clearance_loss": torch.stack(clearance_losses).mean() if clearance_losses else zero,
        }
