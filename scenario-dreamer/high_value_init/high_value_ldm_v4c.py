"""V4C target-only RIFT route-interaction supervision.

V4C preserves the frozen Scenario Dreamer architecture and V4A graph input.
RIFT's interaction waypoint and planner route are labels available only after
decoding. Generated x/y/yaw define the same two-metre preview point used by
RIFT; V4C constrains that point to the selected target's RIFT trace_route, its
remaining route distance to conflict, and ego-route synchronization. None of
those labels is a graph feature, queried online, or used to alter CARLA state.
"""

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from high_value_init.high_value_ldm_v4a import RouteConditionedHighValueLDMV4A


class RouteConditionedHighValueLDMV4C(RouteConditionedHighValueLDMV4A):
    """V4A plus target-only RIFT preview-route interaction losses."""

    def __init__(
            self,
            *args,
            interaction_anchor_along_loss_weight: float = 0.0,
            interaction_anchor_lateral_loss_weight: float = 0.0,
            interaction_route_distance_loss_weight: float = 0.0,
            interaction_route_sync_loss_weight: float = 0.0,
            interaction_route_corridor_loss_weight: float = 0.0,
            interaction_route_tangent_loss_weight: float = 0.0,
            interaction_strength_weight_power: float = 0.0,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction_anchor_along_loss_weight = float(interaction_anchor_along_loss_weight)
        self.interaction_anchor_lateral_loss_weight = float(interaction_anchor_lateral_loss_weight)
        self.interaction_route_distance_loss_weight = float(interaction_route_distance_loss_weight)
        self.interaction_route_sync_loss_weight = float(interaction_route_sync_loss_weight)
        self.interaction_route_corridor_loss_weight = float(interaction_route_corridor_loss_weight)
        self.interaction_route_tangent_loss_weight = float(interaction_route_tangent_loss_weight)
        self.interaction_strength_weight_power = float(interaction_strength_weight_power)
        if self.decoded_state_loss_weight <= 0.0:
            raise ValueError("V4C requires decoded_state_loss_weight > 0 for conflict-anchor supervision")
        if self.interaction_anchor_along_loss_weight < 0.0:
            raise ValueError("interaction_anchor_along_loss_weight must be non-negative")
        if self.interaction_anchor_lateral_loss_weight < 0.0:
            raise ValueError("interaction_anchor_lateral_loss_weight must be non-negative")
        if min(
                self.interaction_route_distance_loss_weight,
                self.interaction_route_sync_loss_weight,
                self.interaction_route_corridor_loss_weight,
                self.interaction_route_tangent_loss_weight,
        ) < 0.0:
            raise ValueError("interaction route-distance/synchronization weights must be non-negative")
        if not 0.0 <= self.interaction_strength_weight_power <= 2.0:
            raise ValueError("interaction_strength_weight_power must be within [0, 2]")

    def training_loss(self, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        loss_dict = super().training_loss(records)
        # _decoded_state_loss_v4a dispatches to our overridden geometry hook,
        # so these terms share exactly the same differentiable decoded sample.
        loss_dict["loss"] = loss_dict["loss"] + (
            self.interaction_anchor_along_loss_weight * loss_dict["interaction_anchor_along_loss"] +
            self.interaction_anchor_lateral_loss_weight * loss_dict["interaction_anchor_lateral_loss"] +
            self.interaction_route_distance_loss_weight * loss_dict["interaction_route_distance_loss"] +
            self.interaction_route_sync_loss_weight * loss_dict["interaction_route_sync_loss"] +
            self.interaction_route_corridor_loss_weight * loss_dict["interaction_route_corridor_loss"] +
            self.interaction_route_tangent_loss_weight * loss_dict["interaction_route_tangent_loss"]
        )
        return loss_dict

    def _v4a_geometry_losses(self, raw_agents: torch.Tensor, data, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        metrics = super()._v4a_geometry_losses(raw_agents, data, records)
        metrics.update(self._route_interaction_losses(raw_agents, data, records))
        return metrics

    @staticmethod
    def _supervision(record: Dict):
        return list((record.get("target") or {}).get("route_interaction_supervision") or [])

    @staticmethod
    def _scalar(item: Dict, key: str):
        try:
            value = float(item[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _route_interaction_losses(self, raw_agents: torch.Tensor, data, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Compare decoded states to the target conflict-anchor relation.

        ``raw_agents`` columns are ``x, y, speed, cos(yaw), sin(yaw), L, W``.
        Predictions are matched to the supervised target order before reading
        its anchor.  No RIFT field participates in graph construction.
        """
        along_losses: List[torch.Tensor] = []
        lateral_losses: List[torch.Tensor] = []
        along_errors: List[torch.Tensor] = []
        lateral_errors: List[torch.Tensor] = []
        route_distance_losses: List[torch.Tensor] = []
        route_sync_losses: List[torch.Tensor] = []
        route_distance_errors: List[torch.Tensor] = []
        route_sync_gaps: List[torch.Tensor] = []
        route_sync_errors: List[torch.Tensor] = []
        route_corridor_losses: List[torch.Tensor] = []
        route_tangent_losses: List[torch.Tensor] = []
        route_corridor_errors: List[torch.Tensor] = []
        route_tangent_errors: List[torch.Tensor] = []
        target_mask, batch = data["agent"].target_state_mask, data["agent"].batch
        for scene_idx, record in enumerate(records):
            supervision = self._supervision(record)
            target_states = (record.get("target") or {}).get("generated_agent_states") or []
            pred = raw_agents[target_mask & (batch == scene_idx)]
            if pred.numel() == 0 or len(supervision) != len(target_states):
                continue
            target_xy = torch.tensor(
                [state[:2] for state in target_states], dtype=pred.dtype, device=pred.device,
            )
            pred = self._match_prediction_order(pred, target_xy)
            count = min(int(pred.shape[0]), len(supervision))
            for index in range(count):
                item = supervision[index] or {}
                anchor = item.get("conflict_point_local_xy")
                target_along = self._scalar(item, "target_to_conflict_along_heading_m")
                target_lateral = self._scalar(item, "target_to_conflict_lateral_heading_m")
                strength = self._scalar(item, "interaction_strength")
                if (not isinstance(anchor, (list, tuple)) or len(anchor) < 2 or
                        target_along is None or target_lateral is None or strength is None):
                    continue
                try:
                    anchor_x, anchor_y = float(anchor[0]), float(anchor[1])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(anchor_x) or not math.isfinite(anchor_y):
                    continue
                state = pred[index]
                heading = state[3:5] / torch.linalg.norm(state[3:5]).clamp_min(1e-5)
                lateral = torch.stack((-heading[1], heading[0]))
                anchor_tensor = torch.tensor((anchor_x, anchor_y), dtype=pred.dtype, device=pred.device)
                delta = anchor_tensor - state[:2]
                predicted_along = torch.dot(delta, heading)
                predicted_lateral = torch.dot(delta, lateral)
                along_error = F.smooth_l1_loss(predicted_along, torch.tensor(target_along, dtype=pred.dtype, device=pred.device))
                lateral_error = F.smooth_l1_loss(predicted_lateral, torch.tensor(target_lateral, dtype=pred.dtype, device=pred.device))
                # This is a static target importance weight, never an input or
                # online score.  A zero exponent makes every selected target
                # contribute equally.
                # Retain support for every selected RIFT-positive target.  A
                # score of zero must not silently remove its geometry gradient.
                importance = (0.25 + 0.75 * max(min(strength, 1.0), 0.0)) ** self.interaction_strength_weight_power
                along_losses.append(along_error * importance)
                lateral_losses.append(lateral_error * importance)
                along_errors.append(torch.abs(predicted_along - target_along).detach())
                lateral_errors.append(torch.abs(predicted_lateral - target_lateral).detach())
                route_metrics = self._route_progress_metrics(state[:2], state[3:5], item)
                if route_metrics is not None:
                    (
                        predicted_route_distance, ego_route_distance, target_route_distance, threshold,
                        preview_to_route_distance, heading_tangent_misalignment,
                    ) = route_metrics
                    route_distance_loss = F.smooth_l1_loss(predicted_route_distance, target_route_distance)
                    route_sync_gap = torch.abs(predicted_route_distance - ego_route_distance)
                    # The RIFT threshold is a recognition admissibility
                    # criterion, not a useful regression target: almost all
                    # labeled positives are already below it, so a hinge loss
                    # supplies zero gradient.  Reconstruct the labeled
                    # target/ego route-distance gap instead.  This directly
                    # preserves the route-level timing relation while keeping
                    # RIFT labels target-only and offline.
                    target_sync_gap = torch.abs(target_route_distance - ego_route_distance)
                    route_distance_losses.append(route_distance_loss * importance)
                    route_sync_losses.append(F.smooth_l1_loss(route_sync_gap, target_sync_gap) * importance)
                    route_distance_errors.append(torch.abs(predicted_route_distance - target_route_distance).detach())
                    route_sync_gaps.append(route_sync_gap.detach())
                    route_sync_errors.append(torch.abs(route_sync_gap - target_sync_gap).detach())
                    route_corridor_losses.append(F.smooth_l1_loss(
                        preview_to_route_distance, torch.zeros_like(preview_to_route_distance),
                    ) * importance)
                    route_tangent_losses.append(heading_tangent_misalignment * importance)
                    route_corridor_errors.append(preview_to_route_distance.detach())
                    route_tangent_errors.append(heading_tangent_misalignment.detach())
        zero = raw_agents.sum() * 0.0
        return {
            "interaction_anchor_along_loss": torch.stack(along_losses).mean() if along_losses else zero,
            "interaction_anchor_lateral_loss": torch.stack(lateral_losses).mean() if lateral_losses else zero,
            "interaction_anchor_along_abs_error_m": torch.stack(along_errors).mean() if along_errors else zero.detach(),
            "interaction_anchor_lateral_abs_error_m": torch.stack(lateral_errors).mean() if lateral_errors else zero.detach(),
            "interaction_route_distance_loss": torch.stack(route_distance_losses).mean() if route_distance_losses else zero,
            "interaction_route_sync_loss": torch.stack(route_sync_losses).mean() if route_sync_losses else zero,
            "interaction_route_distance_abs_error_m": torch.stack(route_distance_errors).mean() if route_distance_errors else zero.detach(),
            "interaction_route_sync_gap_m": torch.stack(route_sync_gaps).mean() if route_sync_gaps else zero.detach(),
            "interaction_route_sync_abs_error_m": torch.stack(route_sync_errors).mean() if route_sync_errors else zero.detach(),
            # Backward-compatible history key; it is now the supervision
            # error rather than the raw predicted gap.
            "interaction_route_sync_abs_gap_m": torch.stack(route_sync_errors).mean() if route_sync_errors else zero.detach(),
            "interaction_route_corridor_loss": torch.stack(route_corridor_losses).mean() if route_corridor_losses else zero,
            "interaction_route_tangent_loss": torch.stack(route_tangent_losses).mean() if route_tangent_losses else zero,
            "interaction_route_preview_lateral_error_m": torch.stack(route_corridor_errors).mean() if route_corridor_errors else zero.detach(),
            "interaction_route_heading_tangent_error": torch.stack(route_tangent_errors).mean() if route_tangent_errors else zero.detach(),
        }

    @staticmethod
    def _route_progress_metrics(position: torch.Tensor, heading_xy: torch.Tensor, item: Dict):
        """Return target-specific RIFT route distances for one decoded state.

        The selected segment is a detached nearest-segment decision; projection
        within that segment remains differentiable.  This is a standard
        piecewise-smooth projection, not an Euclidean-distance surrogate.
        """
        polyline = item.get("target_route_polyline_local_xy") or []
        try:
            points = torch.tensor(polyline, dtype=position.dtype, device=position.device)
            offset = float(item["target_route_distance_offset_m"])
            ego_distance = float(item["ego_route_distance_to_conflict_m"])
            target_distance = float(item["target_route_distance_to_conflict_m"])
            threshold = float(item["route_distance_difference_threshold_m"])
        except (KeyError, TypeError, ValueError):
            return None
        if (points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2 or
                not all(math.isfinite(value) for value in (offset, ego_distance, target_distance, threshold)) or
                target_distance < 0.0 or ego_distance < 0.0 or threshold <= 0.0):
            return None
        preview_distance = float(item.get("rift_preview_distance_m", 2.0))
        if not math.isfinite(preview_distance) or preview_distance <= 0.0:
            return None
        starts = points[:-1, :2]
        deltas = points[1:, :2] - starts
        lengths = torch.linalg.norm(deltas, dim=-1)
        valid = lengths > 1e-5
        if not bool(valid.any()):
            return None
        norm_sq = (deltas * deltas).sum(dim=-1).clamp_min(1e-9)
        heading = heading_xy / torch.linalg.norm(heading_xy).clamp_min(1e-5)
        preview_position = position + preview_distance * heading
        ratios = ((preview_position.unsqueeze(0) - starts) * deltas).sum(dim=-1) / norm_sq
        ratios = ratios.clamp(0.0, 1.0)
        projections = starts + ratios.unsqueeze(-1) * deltas
        distance_sq = ((preview_position.unsqueeze(0) - projections) ** 2).sum(dim=-1)
        distance_sq = torch.where(valid, distance_sq, torch.full_like(distance_sq, float("inf")))
        index = int(torch.argmin(distance_sq.detach()).item())
        prefix = torch.cat((torch.zeros((1,), dtype=position.dtype, device=position.device), torch.cumsum(lengths, dim=0)))
        remaining = prefix[-1] - (prefix[index] + ratios[index] * lengths[index])
        predicted_distance = remaining + offset
        tangent = deltas[index] / lengths[index].clamp_min(1e-5)
        preview_to_route_distance = torch.sqrt(distance_sq[index].clamp_min(1e-9))
        heading_tangent_misalignment = 1.0 - torch.dot(heading, tangent).clamp(-1.0, 1.0)
        return (
            predicted_distance,
            torch.tensor(ego_distance, dtype=position.dtype, device=position.device),
            torch.tensor(target_distance, dtype=position.dtype, device=position.device),
            torch.tensor(threshold, dtype=position.dtype, device=position.device),
            preview_to_route_distance,
            heading_tangent_misalignment,
        )
