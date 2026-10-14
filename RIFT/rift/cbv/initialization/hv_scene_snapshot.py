#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""RIFT-native scene snapshots for high-value initial-state generation.

This module only serializes the live CARLA/RIFT state and, when requested,
delegates interaction labeling to RIFT's original CBV-recognition logic. It
does not generate actors, optimize actor states, select CBVs, or implement a
separate route/conflict approximation.
"""

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import carla
from shapely.geometry import Point

from nuplan_plugin.maps.maps_datatypes import SemanticMapLayer
from rift.cbv.initialization.rift_interaction_labeler import (
    RiftInteractionLabeler,
    actor_state_to_dict,
    waypoint_to_dict,
)
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


def _actor_distance(ego_vehicle: carla.Vehicle, actor: carla.Actor) -> float:
    ego_loc = CarlaDataProvider.get_location(ego_vehicle) or ego_vehicle.get_location()
    actor_loc = CarlaDataProvider.get_location(actor) or actor.get_location()
    return float(ego_loc.distance(actor_loc))


def _safe_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


class HighValueSceneSnapshotter:
    """Serialize RIFT/CARLA state for generator training and evaluation."""

    def __init__(
            self,
            actor_radius: float = 80.0,
            map_radius: float = 64.0,
            max_route_points: int = 160,
            max_map_lanes: int = 120,
            map_coordinate_mode: str = "legacy",
            candidate_fov: float = 64.0,
            forward_min_x: float = 0.0,
            target_fov_margin: float = 2.0,
            native_map_export: bool = False,
            logger=None):
        self.actor_radius = float(actor_radius)
        self.map_radius = float(map_radius)
        self.max_route_points = int(max_route_points)
        self.max_map_lanes = int(max_map_lanes)
        self.candidate_fov = float(candidate_fov)
        self.forward_min_x = float(forward_min_x)
        self.target_fov_margin = float(target_fov_margin)
        if self.candidate_fov <= 0.0 or self.target_fov_margin < 0.0:
            raise ValueError("candidate FOV must be positive and its margin non-negative")
        if not 0.0 <= self.forward_min_x <= self.candidate_fov / 2.0 - self.target_fov_margin:
            raise ValueError("forward candidate sector is empty or invalid")
        if map_coordinate_mode not in {"legacy", "right_handed"}:
            raise ValueError("map_coordinate_mode must be 'legacy' or 'right_handed'")
        self.map_coordinate_mode = map_coordinate_mode
        self.logger = logger
        self.labeler = RiftInteractionLabeler(logger=logger)
        self._last_map_lane_diagnostics = {}
        self.native_map_export=bool(native_map_export)
        self._native_geometry=None
        self._native_world_identity=None

    def build(
            self,
            ego_vehicle: carla.Vehicle,
            route_config,
            ego_rest_route_waypoints: List[carla.Waypoint],
            cbv_recog_policy=None,
            cbv_route_planner=None,
            existing_cbv_ids: Optional[Iterable[int]] = None,
            include_rift_labels: bool = True,
            frame_id: Optional[int] = None,
            extra: Optional[Dict] = None) -> Dict:
        """Build one snapshot in CARLA world coordinates.

        Dataset builders convert this into ego-local Scenario Dreamer tensors.
        Keeping the raw dump in world coordinates makes RIFT audit reproducible.
        """
        actors = self._nearby_vehicle_states(ego_vehicle)
        labels = []
        label_debug = {}
        if include_rift_labels:
            if cbv_recog_policy is None or cbv_route_planner is None:
                raise ValueError("RIFT labels require cbv_recog_policy and cbv_route_planner")
            targetable = {
                int(actor["actor_id"]): CarlaDataProvider.get_actor_by_id(int(actor["actor_id"]))
                for actor in actors
            }
            targetable = {
                actor_id: actor
                for actor_id, actor in targetable.items()
                if actor is not None and actor.is_alive
            }
            labels = self.labeler.build_labels(
                ego_vehicle=ego_vehicle,
                cbv_recog_policy=cbv_recog_policy,
                cbv_route_planner=cbv_route_planner,
                ego_rest_route_waypoints=ego_rest_route_waypoints,
                existing_cbv_ids=existing_cbv_ids or [],
                targetable_actors=targetable,
            )
            labels = self._attach_local_ids(actors, labels)
            label_debug = dict(self.labeler.last_debug)

        route = [
            waypoint_to_dict(wp)
            for wp in (ego_rest_route_waypoints or [])[:self.max_route_points]
            if wp is not None
        ]
        num_positive = sum(1 for label in labels if label.get("is_interactive", False))
        forward_diagnostics = self._forward_interaction_diagnostics(ego_vehicle, actors, labels)
        map_lanes = self._map_lanes(ego_vehicle)
        payload = {
            "schema_version": "hv_init_rift_snapshot_v1" if not include_rift_labels else "hv_init_rift_labeled_v1",
            "scene_id": self._scene_id(route_config, frame_id),
            "route_id": getattr(route_config, "name", None),
            "route_index": getattr(route_config, "index", None),
            "repetition_index": getattr(route_config, "repetition_index", None),
            "town": getattr(route_config, "town", None),
            "frame_id": int(frame_id) if frame_id is not None else None,
            "coordinate_frame": "carla_world",
            "ego": actor_state_to_dict(ego_vehicle),
            "ego_route": route,
            "map_lanes": map_lanes,
            "agents": actors,
            "rift_labels": labels,
            "diagnostics": {
                "num_agents": len(actors),
                "actor_radius": self.actor_radius,
                "map_radius": self.map_radius,
                "map_coordinate_mode": self.map_coordinate_mode,
                **self._last_map_lane_diagnostics,
                "num_rift_positive_agents": num_positive,
                **forward_diagnostics,
                "rift_labeler": label_debug,
            },
        }
        if extra:
            payload["extra"] = extra
        if self.native_map_export:
            from high_value_init.audit_physics_geometry import MapGeometry
            world=ego_vehicle.get_world();world_map=world.get_map()
            identity=(world_map.name,getattr(world,'id',None))
            if self._native_world_identity!=identity:
                self._native_geometry=MapGeometry(world_map,1.)
                self._native_world_identity=identity
                self._native_map_sha=hashlib.sha256(world_map.to_opendrive().encode()).hexdigest()
            snapshot=world.get_snapshot()
            payload['native_map_lanes']=self._native_geometry.local_lanes(payload['ego'],radius=100.)
            payload['native_map_metadata']={'coordinate_frame':'carla_world_xyz','resolution_m':1.,
                'opendrive_sha256':self._native_map_sha,'map_name':world_map.name,
                'sim_frame_id':int(snapshot.frame),'elapsed_seconds':float(snapshot.timestamp.elapsed_seconds)}
            payload['collection_feasibility']={'saved_forward_positive_count':
                forward_diagnostics.get('num_forward_rift_positive_agents'),
                'absence_of_saved_positives_is_not_proven_infeasible':True}
        return payload

    def _forward_interaction_diagnostics(self, ego_vehicle, actors: List[Dict], labels: List[Dict]) -> Dict:
        """Observation-only V4-A candidate coverage in the same ego frame as dataset building."""
        ego = actor_state_to_dict(ego_vehicle)
        actor_by_id = {int(actor["actor_id"]): actor for actor in actors if actor.get("actor_id") is not None}
        half_fov = self.candidate_fov / 2.0
        forward_limit = half_fov - self.target_fov_margin
        counts = {
            "rear_half": 0,
            "forward_target_sector": 0,
            "forward_or_side_excluded_by_margin_or_min_x": 0,
            "outside_candidate_fov": 0,
            "unmatched_label_actor": 0,
        }
        for label in labels:
            if not label.get("is_interactive", False):
                continue
            try:
                actor_id = int(label.get("actor_id"))
            except (TypeError, ValueError):
                actor_id = None
            actor = actor_by_id.get(actor_id)
            if actor is None:
                counts["unmatched_label_actor"] += 1
                continue
            x, y = self._ego_local_xy(ego, actor)
            if abs(x) > half_fov or abs(y) > half_fov:
                counts["outside_candidate_fov"] += 1
            elif self.forward_min_x <= x <= forward_limit and abs(y) <= forward_limit:
                counts["forward_target_sector"] += 1
            elif x < 0.0:
                counts["rear_half"] += 1
            else:
                counts["forward_or_side_excluded_by_margin_or_min_x"] += 1
        return {
            "v4a_candidate_fov": self.candidate_fov,
            "v4a_forward_min_x": self.forward_min_x,
            "v4a_target_fov_margin": self.target_fov_margin,
            "num_forward_rift_positive_agents": counts["forward_target_sector"],
            "rift_positive_region_counts": counts,
        }

    @staticmethod
    def _ego_local_xy(ego: Dict, actor: Dict) -> Tuple[float, float]:
        dx = float(actor["x"]) - float(ego["x"])
        dy = float(actor["y"]) - float(ego["y"])
        yaw = math.radians(float(ego.get("yaw_deg", 0.0)))
        return (
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
        )

    def dump(self, snapshot: Dict, output_dir: str, prefix: Optional[str] = None) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        scene_id = str(snapshot.get("scene_id", "scene")).replace("/", "_").replace(":", "_")
        filename = f"{prefix}_{scene_id}.json" if prefix else f"{scene_id}.json"
        path = Path(output_dir) / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        return str(path)

    def _nearby_vehicle_states(self, ego_vehicle: carla.Vehicle) -> List[Dict]:
        states = []
        for actor_id, actor in CarlaDataProvider.get_actors().items():
            if actor is None or not actor.is_alive or int(actor_id) == int(ego_vehicle.id):
                continue
            if not getattr(actor, "type_id", "").startswith("vehicle."):
                continue
            distance = _actor_distance(ego_vehicle, actor)
            if distance > self.actor_radius:
                continue
            state = actor_state_to_dict(actor)
            state["ego_distance"] = distance
            states.append(state)
        states.sort(key=lambda item: (float(item.get("ego_distance", 0.0)), int(item["actor_id"])))
        for local_id, state in enumerate(states):
            state["local_id"] = int(local_id)
            state["source"] = "carla_actor"
        return states

    def _map_lanes(self, ego_vehicle: carla.Vehicle) -> List[Dict]:
        map_api = CarlaDataProvider.get_map_api()
        if map_api is None:
            self._last_map_lane_diagnostics = {
                "map_api_available": False,
                "map_lane_query_coordinate_frame": "carla_world",
                "map_lane_query_raw_count": 0,
                "map_lane_serialized_count": 0,
            }
            return []
        ego_loc = CarlaDataProvider.get_location(ego_vehicle) or ego_vehicle.get_location()
        # CarlaMap indexes the RIFT HD map in its right-handed coordinate
        # system.  V3/V4 datasets query that same index with (x, -y), then
        # convert returned lane geometry back to CARLA world coordinates for
        # the snapshot adapter.
        map_y = -float(ego_loc.y) if self.map_coordinate_mode == "right_handed" else float(ego_loc.y)
        proximal = map_api.query_proximal_map_data(
            Point(float(ego_loc.x), map_y), self.map_radius
        )
        lanes = []
        raw_count = 0
        for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
            for obj in proximal.get(layer, []) or []:
                raw_count += 1
                lane = self._serialize_lane(
                    obj,
                    str(layer).split(".")[-1].lower(),
                    map_y_to_carla=self.map_coordinate_mode == "right_handed",
                )
                if lane is not None:
                    lanes.append(lane)
        lanes.sort(key=lambda item: item.get("distance", 0.0))
        lanes = lanes[:self.max_map_lanes]
        self._last_map_lane_diagnostics = {
            "map_api_available": True,
            "map_lane_query_coordinate_frame": (
                "rift_hd_map_right_handed"
                if self.map_coordinate_mode == "right_handed"
                else "legacy"
            ),
            "map_lane_query_x": float(ego_loc.x),
            "map_lane_query_y": map_y,
            "map_lane_query_raw_count": raw_count,
            "map_lane_serialized_count": len(lanes),
        }
        return lanes

    def _serialize_lane(self, obj, layer_name: str, map_y_to_carla: bool = False) -> Optional[Dict]:
        centerline = self._centerline_points(obj, map_y_to_carla=map_y_to_carla)
        if len(centerline) < 2:
            return None
        topology = []
        for item in _safe_list(getattr(obj, "topology", None)):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                topology.append([int(item[0]), int(item[1])])
        return {
            "layer": layer_name,
            "token_id": int(getattr(obj, "token_id", -1)),
            "road_id": int(getattr(obj, "road_id", -1)) if getattr(obj, "road_id", None) is not None else None,
            "lane_id": int(getattr(obj, "lane_id", 0)) if getattr(obj, "lane_id", None) is not None else None,
            "lane_type": str(getattr(obj, "lane_type", "")),
            "topology_type": str(getattr(obj, "topology_type", "")),
            "topology": topology,
            "distance": float(getattr(obj, "distance", 0.0)),
            "centerline": centerline,
        }

    def _centerline_points(self, obj, map_y_to_carla: bool = False) -> List[List[float]]:
        raw = getattr(obj, "centerline", None)
        if raw is None:
            raw = getattr(obj, "center_coords", None)
        points = []
        for point in _safe_list(raw):
            if hasattr(point, "x") and hasattr(point, "y"):
                x, y, heading = float(point.x), float(point.y), float(getattr(point, "heading", 0.0))
            else:
                # CarlaMap exposes centerline rows as NumPy arrays, while
                # serialized snapshots may contain lists or tuples.  Accept
                # every indexable coordinate row, matching the V3/V4 offline
                # HD-map extractor.
                try:
                    if len(point) < 2:
                        continue
                    x, y = float(point[0]), float(point[1])
                    heading = float(point[2]) if len(point) >= 3 else 0.0
                except (TypeError, IndexError, ValueError):
                    continue
            points.append([x, -y if map_y_to_carla else y, -heading if map_y_to_carla else heading])
        return points

    def _attach_local_ids(self, actors: List[Dict], labels: List[Dict]) -> List[Dict]:
        actor_to_local = {int(actor["actor_id"]): int(actor["local_id"]) for actor in actors}
        aligned = []
        for label in labels:
            label = dict(label)
            label["local_id"] = actor_to_local.get(int(label["actor_id"]))
            aligned.append(label)
        aligned.sort(key=lambda item: (item.get("local_id") is None, item.get("local_id", 10**9)))
        return aligned

    def _scene_id(self, route_config, frame_id: Optional[int]) -> str:
        name = getattr(route_config, "name", "RouteScenario")
        index = getattr(route_config, "index", "idx")
        rep = getattr(route_config, "repetition_index", 0)
        frame = "none" if frame_id is None else int(frame_id)
        return f"{name}_idx{index}_rep{rep}_frame{frame}"
