#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
RIFT-native path interaction labels for high-value initial-state generation.

This module intentionally delegates route-level interaction reasoning to the
existing RIFT CBV recognition policy. It only serializes the resulting labels
and metadata for a downstream generator.
"""

import math
from typing import Dict, Iterable, List, Optional

import carla

from rift.scenario.tools.carla_data_provider import CarlaDataProvider


def get_rift_targetable_actors(
        ego_vehicle: carla.Vehicle,
        existing_cbv_ids: Optional[Iterable[int]] = None) -> Dict[int, carla.Actor]:
    """Return the same actor universe used by RIFT CBV candidate selection.

    This only aligns serialization with RIFT's original candidate source
    (CarlaDataProvider.get_actors). It does not implement any route/conflict
    filtering here.
    """
    existing_cbv_ids = set(existing_cbv_ids or [])
    reached_cbv_ids = set(CarlaDataProvider.get_CBVs_reach_goal_by_ego(ego_vehicle.id).keys())
    actors = {}
    for actor_id, actor in CarlaDataProvider.get_actors().items():
        if actor is None or not actor.is_alive:
            continue
        if actor_id == ego_vehicle.id or actor_id in existing_cbv_ids or actor_id in reached_cbv_ids:
            continue
        if not getattr(actor, "type_id", "").startswith("vehicle."):
            continue
        actors[int(actor_id)] = actor
    return actors


def _safe_float(value, default=None):
    if value is None:
        return default
    return float(value)


def actor_state_to_dict(actor: carla.Actor) -> Dict:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    extent = actor.bounding_box.extent
    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return {
        "actor_id": int(actor.id),
        "type_id": actor.type_id,
        "role_name": actor.attributes.get("role_name", ""),
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "roll_deg": float(transform.rotation.roll),
        "pitch_deg": float(transform.rotation.pitch),
        "yaw_deg": float(transform.rotation.yaw),
        "vx": float(velocity.x),
        "vy": float(velocity.y),
        "vz": float(velocity.z),
        "speed_mps": float(speed),
        "length": float(extent.x * 2.0),
        "width": float(extent.y * 2.0),
        "height": float(extent.z * 2.0),
    }


def waypoint_to_dict(waypoint: Optional[carla.Waypoint]) -> Optional[Dict]:
    if waypoint is None:
        return None
    transform = waypoint.transform
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "roll_deg": float(transform.rotation.roll),
        "pitch_deg": float(transform.rotation.pitch),
        "yaw_deg": float(transform.rotation.yaw),
        "road_id": int(waypoint.road_id),
        "lane_id": int(waypoint.lane_id),
        "section_id": int(waypoint.section_id),
        "s": float(waypoint.s),
        "is_junction": bool(waypoint.is_junction),
    }


def route_elements_to_waypoints(route_elements) -> List[Dict]:
    """Serialize RIFT's already-planned target route to the conflict point.

    This is an additive, offline-label field.  It does not influence candidate
    recognition, CBV planning, actor control, or online state generation.  A
    later dataset builder may use it as target-only route-progress supervision
    instead of incorrectly replacing a planner distance with straight-line
    distance.
    """
    points = []
    for element in route_elements or []:
        waypoint = element[0] if isinstance(element, (tuple, list)) and element else element
        point = waypoint_to_dict(waypoint)
        if point is None:
            continue
        if points:
            previous = points[-1]
            if abs(float(previous["x"]) - float(point["x"])) < 1e-4 and abs(float(previous["y"]) - float(point["y"])) < 1e-4:
                continue
        points.append(point)
    return points


class RiftInteractionLabeler:
    """Serialize labels produced by RIFT's original path interaction logic."""

    def __init__(self, logger=None):
        self.logger = logger
        self.last_debug = {}

    def build_labels(
            self,
            ego_vehicle: carla.Vehicle,
            cbv_recog_policy,
            cbv_route_planner,
            ego_rest_route_waypoints: List[carla.Waypoint],
            existing_cbv_ids: Optional[Iterable[int]] = None,
            targetable_actors: Optional[Dict[int, carla.Actor]] = None) -> List[Dict]:
        existing_cbv_ids = set(existing_cbv_ids or [])
        scenario_actors = CarlaDataProvider.get_scenario_actors_by_ego(ego_vehicle.id)
        scenario_bvs = {
            actor_id: actor
            for actor_id, actor in scenario_actors.get("BVs", {}).items()
            if actor is not None and actor.is_alive
        }
        registered_actors = CarlaDataProvider.get_actors()
        bvs = targetable_actors or get_rift_targetable_actors(
            ego_vehicle,
            existing_cbv_ids=existing_cbv_ids,
        )

        for actor_id in bvs:
            cbv_route_planner.inter_info.pop(actor_id, None)

        candidates = cbv_recog_policy.get_CBV_candidates(
            ego_vehicle,
            existing_cbv_ids,
            ego_rest_route_waypoints
        )
        candidate_ids = {actor.id for actor in candidates}

        labels = []
        for actor_id in sorted(bvs):
            inter_info = cbv_route_planner.inter_info.get(actor_id, {})
            labels.append(self._build_actor_label(actor_id, actor_id in candidate_ids, inter_info))
        self.last_debug = {
            "actor_source": "CarlaDataProvider.get_actors",
            "num_registered_actors": len(registered_actors),
            "num_scenario_bvs": len(scenario_bvs),
            "num_targetable_actors": len(bvs),
            "num_rift_candidates": len(candidate_ids),
            "num_interactive_labels": sum(1 for label in labels if label.get("is_interactive", False)),
            "targetable_actor_ids": sorted(int(actor_id) for actor_id in bvs),
            "scenario_bv_actor_ids": sorted(int(actor_id) for actor_id in scenario_bvs),
            "rift_candidate_actor_ids": sorted(int(actor_id) for actor_id in candidate_ids),
            "unlabeled_rift_candidate_actor_ids": sorted(int(actor_id) for actor_id in candidate_ids - set(bvs)),
            "rift_candidate_debug": getattr(cbv_recog_policy, "last_candidate_debug", {}),
        }
        return labels

    def _build_actor_label(self, actor_id: int, is_candidate: bool, inter_info: Dict) -> Dict:
        inter_waypoint = inter_info.get("inter_waypoint")
        interaction_meta = dict(inter_info.get("interaction_meta", {}))
        distance_diff = interaction_meta.get("distance_diff")
        threshold = interaction_meta.get("min_interaction_dis_diff")
        score = 0.0
        if distance_diff is not None and threshold:
            score = max(0.0, 1.0 - float(distance_diff) / float(threshold))

        is_interactive = bool(is_candidate and inter_waypoint is not None)
        # `inter_CBV_route_elements` originates from the original RIFT
        # recognition planner.  Keep it as label evidence only; no caller in
        # this module feeds it back into recognition or control.
        target_route = route_elements_to_waypoints(inter_info.get("inter_CBV_route_elements")) if is_interactive else []
        return {
            "actor_id": int(actor_id),
            "is_interactive": is_interactive,
            "score": float(score) if is_interactive else 0.0,
            "ego_route_index": interaction_meta.get("ego_route_index"),
            "ego_distance": _safe_float(interaction_meta.get("ego_distance")),
            "cbv_distance": _safe_float(interaction_meta.get("CBV_distance")),
            "distance_diff": _safe_float(distance_diff),
            "min_interaction_dis_diff": _safe_float(threshold),
            "interaction_waypoint": waypoint_to_dict(inter_waypoint),
            "target_route_to_interaction": target_route,
        }
