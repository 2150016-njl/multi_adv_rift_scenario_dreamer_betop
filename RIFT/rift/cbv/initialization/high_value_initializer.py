#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
RIFT-native high-value initial-state generation pipeline.

RIFT owns route-level interaction labeling and online auditing. The external
Scenario Dreamer generator scores all actor slots, generates full-scene
initial states, and asks RIFT to apply those generated states only to top-k
existing actors. The first ``hv_init_target_count`` selected actors are
registered as adversarial CBVs.
"""

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import carla
import numpy as np
from shapely.geometry import Point

from rift.cbv.initialization.generator import (
    HighValueInitGeneratorError,
    build_high_value_generator,
)
from rift.cbv.initialization.rift_interaction_labeler import (
    RiftInteractionLabeler,
    actor_state_to_dict,
    get_rift_targetable_actors,
    waypoint_to_dict,
)
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from nuplan_plugin.maps.maps_datatypes import SemanticMapLayer


class HighValueCBVInitializer:
    def __init__(self, config: Dict, world: carla.World, logger=None):
        self.config = config
        self.world = world
        self.logger = logger
        self.cbv_count = int(config.get("hv_init_target_count", 2))
        self.target_count = self.cbv_count
        self.apply_top_k = int(config.get("hv_init_apply_top_k", self.cbv_count))
        self.apply_top_k = max(self.apply_top_k, self.cbv_count)
        self.max_attempts = int(config.get("hv_init_max_attempts", 1))
        self.collect_only = bool(config.get("hv_init_collect_only", False))
        self.require_target_mask = bool(config.get("hv_init_require_target_mask", True))
        self.fail_on_invalid = bool(config.get("hv_init_fail_on_invalid", True))
        self.validation_mode = str(config.get("hv_init_validation_mode", "audit"))
        if self.validation_mode not in {"audit", "strict"}:
            raise ValueError("hv_init_validation_mode must be either 'audit' or 'strict'")
        self.lock_selected_cbvs = bool(config.get("hv_init_lock_selected_cbvs", True))
        self.reject_extra_rift_positive = bool(config.get("hv_init_reject_extra_rift_positive", False))
        self.tick_after_apply = bool(config.get("hv_init_tick_after_apply", True))
        self.dump_dir = config.get("hv_init_dump_dir")
        self.save_viz = bool(config.get("hv_init_save_viz", True))
        self.min_selected_pair_distance = float(config.get("hv_init_min_selected_pair_distance", 0.0))
        self.max_abs_xy_delta = float(config.get("hv_init_max_abs_xy_delta", 30.0))
        self.max_abs_yaw_delta = float(config.get("hv_init_max_abs_yaw_delta", 180.0))
        self.max_speed_mps = float(config.get("hv_init_max_speed_mps", 40.0))
        self.max_route_points = int(config.get("hv_init_max_route_points", 160))
        self.map_radius = float(config.get("hv_init_map_radius", 64.0))
        self.max_map_lanes = int(config.get("hv_init_max_map_lanes", 120))

        self.labeler = RiftInteractionLabeler(logger=logger)
        self.generator = None if self.collect_only else build_high_value_generator(config)

        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

    def initialize(
            self,
            ego_vehicle: carla.Vehicle,
            route_config,
            cbv_recog_policy,
            cbv_route_planner,
            ego_rest_route_waypoints: List[carla.Waypoint],
            existing_cbv_ids: Optional[Iterable[int]] = None) -> Dict:
        existing_cbv_ids = set(existing_cbv_ids or [])
        actors = self._get_targetable_bvs(ego_vehicle, existing_cbv_ids=existing_cbv_ids)
        if len(actors) < self.target_count:
            return self._failure("not_enough_targetable_bvs", selected_actor_ids=[])

        last_error = None
        last_failure_result = None
        original_states = {actor_id: self._snapshot_actor(actor) for actor_id, actor in actors.items()}

        for attempt in range(self.max_attempts):
            scene = None
            proposal = {}
            try:
                pre_labels = self.labeler.build_labels(
                    ego_vehicle,
                    cbv_recog_policy,
                    cbv_route_planner,
                    ego_rest_route_waypoints,
                    existing_cbv_ids=existing_cbv_ids,
                    targetable_actors=actors,
                )
                scene = self._build_scene(
                    ego_vehicle,
                    route_config,
                    actors,
                    ego_rest_route_waypoints,
                    pre_labels,
                )
                if self.collect_only:
                    result = {
                        "success": False,
                        "collect_only": True,
                        "attempt": attempt,
                        "selected_actor_ids": [],
                        "cbv_actor_ids": [],
                        "pre_interaction_labels": pre_labels,
                        "num_interactive_labels": sum(
                            1 for label in pre_labels if label.get("is_interactive", False)
                        ),
                        "target_count": self.cbv_count,
                        "apply_top_k": self.apply_top_k,
                    }
                    self._dump_attempt(scene, proposal, result, attempt)
                    self._log(
                        ">> High-value init collect-only dumped "
                        f"{result['num_interactive_labels']} interactive labels",
                        color="yellow",
                    )
                    return result

                proposal = self.generator.generate(scene)
                selected_actor_ids, cbv_actor_ids, updates = self._normalize_proposal(proposal, scene)

                self._validate_selected_actor_ids(selected_actor_ids, actors)
                self._apply_updates(selected_actor_ids, updates, actors, original_states)

                if self.tick_after_apply:
                    self.world.tick()
                    CarlaDataProvider.on_carla_tick()

                post_labels = self.labeler.build_labels(
                    ego_vehicle,
                    cbv_recog_policy,
                    cbv_route_planner,
                    ego_rest_route_waypoints,
                    existing_cbv_ids=existing_cbv_ids,
                    targetable_actors=actors,
                )
                validation_error = self._validate_rift_result(cbv_actor_ids, post_labels)
                if validation_error is None or self.validation_mode == "audit":
                    selected_actors = [actors[actor_id] for actor_id in selected_actor_ids]
                    cbv_actors = [actors[actor_id] for actor_id in cbv_actor_ids]
                    result = {
                        "success": True,
                        "rift_validation_passed": validation_error is None,
                        "rift_validation_error": validation_error,
                        "attempt": attempt,
                        "selected_actor_ids": selected_actor_ids,
                        "cbv_actor_ids": cbv_actor_ids,
                        "selected_actors": selected_actors,
                        "cbv_actors": cbv_actors,
                        "pre_interaction_labels": pre_labels,
                        "post_interaction_labels": post_labels,
                        "proposal": proposal,
                        "target_count": self.cbv_count,
                        "apply_top_k": self.apply_top_k,
                    }
                    self._dump_attempt(scene, proposal, result, attempt)
                    if validation_error is None:
                        self._log(
                            f">> High-value init accepted CBV ids: {cbv_actor_ids}; updated actor ids: {selected_actor_ids}",
                            color="yellow",
                        )
                    else:
                        self._log(
                            f">> High-value init kept model output but RIFT audit failed: {validation_error}",
                            color="yellow",
                        )
                    return result

                last_error = validation_error
                self._restore_updates(selected_actor_ids, actors, original_states)
                self._dump_attempt(scene, proposal, self._failure(validation_error, selected_actor_ids), attempt)

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._restore_all(actors, original_states)
                last_failure_result = self._failure(last_error, selected_actor_ids=[])
                if scene is not None:
                    self._dump_attempt(
                        scene,
                        proposal,
                        last_failure_result,
                        attempt,
                    )
                if isinstance(exc, HighValueInitGeneratorError):
                    break

        return last_failure_result or self._failure(last_error or "unknown_failure", selected_actor_ids=[])

    def should_lock_dynamic_cbv(self, init_result: Optional[Dict]) -> bool:
        return bool(self.lock_selected_cbvs and init_result and init_result.get("success", False))

    def result_for_info(self, init_result: Optional[Dict]) -> Optional[Dict]:
        if init_result is None:
            return None
        pre_labels = init_result.get("pre_interaction_labels") or []
        post_labels = init_result.get("post_interaction_labels") or []
        pre_positive_ids = [
            int(label["actor_id"])
            for label in pre_labels
            if label.get("is_interactive", False)
        ]
        post_positive_ids = [
            int(label["actor_id"])
            for label in post_labels
            if label.get("is_interactive", False)
        ]
        return {
            "success": bool(init_result.get("success", False)),
            "collect_only": bool(init_result.get("collect_only", False)),
            "rift_validation_passed": init_result.get("rift_validation_passed"),
            "rift_validation_error": init_result.get("rift_validation_error"),
            "attempt": init_result.get("attempt"),
            "selected_actor_ids": init_result.get("selected_actor_ids", []),
            "cbv_actor_ids": init_result.get("cbv_actor_ids", []),
            "interaction_actor_ids": init_result.get("selected_actor_ids", []),
            "failure_reason": init_result.get("failure_reason"),
            "num_interactive_labels": init_result.get("num_interactive_labels"),
            "num_pre_interactive_labels": len(pre_positive_ids),
            "num_post_interactive_labels": len(post_positive_ids) if post_labels else None,
            "pre_interactive_actor_ids": pre_positive_ids,
            "post_interactive_actor_ids": post_positive_ids if post_labels else None,
            "target_count": init_result.get("target_count", self.cbv_count),
            "apply_top_k": init_result.get("apply_top_k", self.apply_top_k),
        }

    def _build_scene(
            self,
            ego_vehicle: carla.Vehicle,
            route_config,
            actors: Dict[int, carla.Actor],
            ego_rest_route_waypoints: List[carla.Waypoint],
            interaction_labels: List[Dict]) -> Dict:
        ordered_actor_ids = self._ordered_actor_ids_by_ego_distance(ego_vehicle, actors)
        labels_by_actor_id = {label["actor_id"]: label for label in interaction_labels}
        actor_states = [actor_state_to_dict(actors[actor_id]) for actor_id in ordered_actor_ids]
        labels = [
            labels_by_actor_id.get(actor_id, self._empty_label(actor_id))
            for actor_id in ordered_actor_ids
        ]

        return {
            "schema_version": "rift_high_value_init_v1",
            "target_count": self.cbv_count,
            "apply_top_k": self.apply_top_k,
            "coordinate_frame": "carla_world",
            "scenario": {
                "name": getattr(route_config, "name", None),
                "index": getattr(route_config, "index", None),
                "repetition_index": getattr(route_config, "repetition_index", None),
                "town": getattr(route_config, "town", None),
            },
            "ego": actor_state_to_dict(ego_vehicle),
            "ego_route": [
                waypoint_to_dict(waypoint)
                for waypoint in ego_rest_route_waypoints[:self.max_route_points]
            ],
            "actors": actor_states,
            "interaction_labels": labels,
            "interaction_diagnostics": dict(self.labeler.last_debug),
            "map_lanes": self._serialize_map_lanes(ego_vehicle),
        }

    def _ordered_actor_ids_by_ego_distance(
            self,
            ego_vehicle: carla.Vehicle,
            actors: Dict[int, carla.Actor]) -> List[int]:
        ego_location = CarlaDataProvider.get_location(ego_vehicle)
        ranked = []
        for actor_id, actor in actors.items():
            actor_location = CarlaDataProvider.get_location(actor)
            ranked.append((ego_location.distance(actor_location), int(actor_id)))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [actor_id for _distance, actor_id in ranked]

    def _serialize_map_lanes(self, ego_vehicle: carla.Vehicle) -> List[Dict]:
        map_api = CarlaDataProvider.get_map_api()
        if map_api is None:
            return []

        ego_loc = ego_vehicle.get_location()
        map_objects = map_api.query_proximal_map_data(Point(ego_loc.x, ego_loc.y), self.map_radius)
        lanes = []
        for layer, layer_name in (
                (SemanticMapLayer.LANE, "lane"),
                (SemanticMapLayer.LANE_CONNECTOR, "lane_connector")):
            for lane_obj in map_objects.get(layer, []):
                centerline = getattr(lane_obj, "centerline", None)
                if centerline is None:
                    centerline = getattr(lane_obj, "center_coords", None)
                points = self._to_xy_points(centerline)
                if len(points) < 2:
                    continue
                lanes.append({
                    "road_id": getattr(lane_obj, "road_id", None),
                    "lane_id": getattr(lane_obj, "lane_id", None),
                    "layer": layer_name,
                    "centerline": points,
                    "topology": self._topology_to_pairs(getattr(lane_obj, "topology", [])),
                })
                if len(lanes) >= self.max_map_lanes:
                    return lanes
        return lanes

    def _to_xy_points(self, centerline) -> List[List[float]]:
        if centerline is None:
            return []
        arr = np.asarray(centerline)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return []
        return [[float(x), float(y)] for x, y in arr[:, :2]]

    def _topology_to_pairs(self, topology) -> List[List[int]]:
        pairs = []
        for item in topology or []:
            if item is None or len(item) < 2:
                continue
            pairs.append([int(item[0]), int(item[1])])
        return pairs

    def _normalize_proposal(self, proposal: Dict, scene: Dict) -> Tuple[List[int], List[int], Dict[int, Dict]]:
        if not isinstance(proposal, dict):
            raise ValueError("high-value generator must return a dict proposal")

        actor_ids = [actor["actor_id"] for actor in scene["actors"]]
        if proposal.get("selected_actor_ids"):
            selected_actor_ids = [int(actor_id) for actor_id in proposal["selected_actor_ids"]]
        elif "actor_scores" in proposal:
            scores = list(proposal["actor_scores"])
            if len(scores) != len(actor_ids):
                raise ValueError(
                    f"actor_scores length {len(scores)} does not match actor count {len(actor_ids)}"
                )
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda idx: float(scores[idx]),
                reverse=True,
            )
            selected_actor_ids = [
                int(actor_ids[idx])
                for idx in ranked_indices[:min(self.apply_top_k, len(actor_ids))]
            ]
        elif "target_mask" not in proposal:
            if self.require_target_mask:
                raise ValueError(
                    "proposal must contain target_mask or actor_scores aligned with scene['actors']"
                )
            selected_actor_ids = []
        else:
            target_mask = self._normalize_target_mask(proposal["target_mask"], actor_ids)
            selected_actor_ids = [
                int(actor_id) for actor_id, selected in zip(actor_ids, target_mask) if selected
            ]

        if len(selected_actor_ids) < self.cbv_count:
            raise ValueError(
                f"proposal selected {len(selected_actor_ids)} actors; expected at least {self.cbv_count}"
            )

        if proposal.get("cbv_actor_ids"):
            cbv_actor_ids = [int(actor_id) for actor_id in proposal["cbv_actor_ids"]]
        elif "cbv_mask" in proposal:
            cbv_mask = self._normalize_target_mask(proposal["cbv_mask"], actor_ids)
            cbv_actor_ids = [
                int(actor_id) for actor_id, selected in zip(actor_ids, cbv_mask) if selected
            ]
        else:
            cbv_actor_ids = selected_actor_ids[:self.cbv_count]

        if len(cbv_actor_ids) != self.cbv_count:
            raise ValueError(
                f"proposal selected {len(cbv_actor_ids)} CBVs; expected {self.cbv_count}"
            )
        non_updated_cbv_ids = set(cbv_actor_ids) - set(selected_actor_ids)
        if non_updated_cbv_ids:
            raise ValueError(f"CBV ids are not included in selected actor ids: {sorted(non_updated_cbv_ids)}")

        updates = {}
        for actor_id in selected_actor_ids:
            actor_index = actor_ids.index(actor_id)
            update = self._lookup_actor_update(proposal.get("state_updates"), actor_id, actor_index)
            if update is None:
                raise ValueError(f"missing absolute state update for selected actor {actor_id}")
            updates[actor_id] = update

        return selected_actor_ids, cbv_actor_ids, updates

    def _normalize_target_mask(self, raw_mask, actor_ids: List[int]) -> List[bool]:
        if isinstance(raw_mask, dict):
            return [
                bool(raw_mask.get(actor_id, raw_mask.get(str(actor_id), False)))
                for actor_id in actor_ids
            ]
        if len(raw_mask) != len(actor_ids):
            raise ValueError(
                f"target_mask length {len(raw_mask)} does not match actor count {len(actor_ids)}"
            )
        return [bool(value) for value in raw_mask]

    def _lookup_actor_update(self, container, actor_id: int, actor_index: int):
        if container is None:
            return None
        if isinstance(container, dict):
            return container.get(actor_id, container.get(str(actor_id)))
        if len(container) != 0 and isinstance(container, list):
            return container[actor_index]
        raise ValueError("state_updates must be a dict or an actor-aligned list")

    def _validate_selected_actor_ids(self, selected_actor_ids: List[int], actors: Dict[int, carla.Actor]):
        missing_ids = [actor_id for actor_id in selected_actor_ids if actor_id not in actors]
        if missing_ids:
            raise ValueError(f"proposal selected non-targetable actor ids: {missing_ids}")

    def _apply_updates(
            self,
            selected_actor_ids: List[int],
            updates: Dict[int, Dict],
            actors: Dict[int, carla.Actor],
            original_states: Dict[int, Dict]):
        updated_locations = []
        for actor_id in selected_actor_ids:
            actor = actors[actor_id]
            transform, velocity = self._build_updated_state(original_states[actor_id], updates[actor_id])
            actor.set_transform(transform)
            actor.set_target_velocity(velocity)
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            updated_locations.append(transform.location)

        if self.min_selected_pair_distance > 0.0 and len(updated_locations) >= 2:
            for i, loc_i in enumerate(updated_locations):
                for loc_j in updated_locations[i + 1:]:
                    if loc_i.distance(loc_j) < self.min_selected_pair_distance:
                        raise ValueError("selected actors are too close after generated update")

    def _build_updated_state(self, original_state: Dict, update: Dict) -> Tuple[carla.Transform, carla.Vector3D]:
        if not isinstance(update, dict):
            raise ValueError("each state update must be a dict")

        required_keys = ("x", "y", "yaw_deg")
        missing = [key for key in required_keys if key not in update]
        if missing:
            raise ValueError(f"state update missing required keys: {missing}")

        x = float(update["x"])
        y = float(update["y"])
        z = float(update.get("z", original_state["z"]))
        if abs(x - original_state["x"]) > self.max_abs_xy_delta or abs(y - original_state["y"]) > self.max_abs_xy_delta:
            raise ValueError("generated xy move exceeds hv_init_max_abs_xy_delta")

        if "heading_rad" in update:
            yaw_deg = math.degrees(float(update["heading_rad"]))
        else:
            yaw_deg = float(update["yaw_deg"])
        yaw_delta = abs((yaw_deg - original_state["yaw_deg"] + 180.0) % 360.0 - 180.0)
        if yaw_delta > self.max_abs_yaw_delta:
            raise ValueError("generated yaw move exceeds hv_init_max_abs_yaw_delta")

        speed_mps = update.get("speed_mps")
        if speed_mps is None:
            speed_mps = original_state["speed_mps"]
        speed_mps = max(0.0, min(float(speed_mps), self.max_speed_mps))

        if all(key in update for key in ("vx", "vy")):
            vx = float(update["vx"])
            vy = float(update["vy"])
            vz = float(update.get("vz", original_state["vz"]))
        else:
            yaw_rad = math.radians(yaw_deg)
            vx = speed_mps * math.cos(yaw_rad)
            vy = speed_mps * math.sin(yaw_rad)
            vz = float(update.get("vz", original_state["vz"]))

        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(
                roll=float(update.get("roll_deg", original_state["roll_deg"])),
                pitch=float(update.get("pitch_deg", original_state["pitch_deg"])),
                yaw=yaw_deg,
            )
        )
        velocity = carla.Vector3D(vx, vy, vz)
        return transform, velocity

    def _validate_rift_result(self, cbv_actor_ids: List[int], post_labels: List[Dict]) -> Optional[str]:
        labels_by_actor_id = {label["actor_id"]: label for label in post_labels}
        invalid_ids = [
            actor_id
            for actor_id in cbv_actor_ids
            if not labels_by_actor_id.get(actor_id, {}).get("is_interactive", False)
        ]
        if invalid_ids:
            return f"CBV actors failed RIFT interaction validation: {invalid_ids}"

        if self.reject_extra_rift_positive:
            positive_ids = {
                actor_id
                for actor_id, label in labels_by_actor_id.items()
                if label.get("is_interactive", False)
            }
            extra_ids = positive_ids - set(cbv_actor_ids)
            if extra_ids:
                return f"extra RIFT-positive actors after generation: {sorted(extra_ids)}"

        return None

    def _restore_updates(
            self,
            actor_ids: Iterable[int],
            actors: Dict[int, carla.Actor],
            original_states: Dict[int, Dict]):
        for actor_id in actor_ids:
            if actor_id in actors and actor_id in original_states:
                self._restore_actor(actors[actor_id], original_states[actor_id])
        if self.tick_after_apply:
            self.world.tick()
            CarlaDataProvider.on_carla_tick()

    def _restore_all(self, actors: Dict[int, carla.Actor], original_states: Dict[int, Dict]):
        self._restore_updates(actors.keys(), actors, original_states)

    def _restore_actor(self, actor: carla.Actor, state: Dict):
        transform = carla.Transform(
            carla.Location(x=state["x"], y=state["y"], z=state["z"]),
            carla.Rotation(
                roll=state["roll_deg"],
                pitch=state["pitch_deg"],
                yaw=state["yaw_deg"],
            )
        )
        actor.set_transform(transform)
        actor.set_target_velocity(carla.Vector3D(state["vx"], state["vy"], state["vz"]))

    def _snapshot_actor(self, actor: carla.Actor) -> Dict:
        return actor_state_to_dict(actor)

    def _get_targetable_bvs(
            self,
            ego_vehicle: carla.Vehicle,
            existing_cbv_ids: Optional[Iterable[int]] = None) -> Dict[int, carla.Actor]:
        return get_rift_targetable_actors(ego_vehicle, existing_cbv_ids=existing_cbv_ids)

    def _empty_label(self, actor_id: int) -> Dict:
        return {
            "actor_id": int(actor_id),
            "is_interactive": False,
            "score": 0.0,
            "ego_route_index": None,
            "ego_distance": None,
            "cbv_distance": None,
            "distance_diff": None,
            "min_interaction_dis_diff": None,
            "interaction_waypoint": None,
        }

    def _failure(self, reason, selected_actor_ids: Optional[List[int]] = None) -> Dict:
        result = {
            "success": False,
            "failure_reason": str(reason),
            "selected_actor_ids": selected_actor_ids or [],
            "cbv_actor_ids": [],
            "target_count": self.cbv_count,
            "apply_top_k": self.apply_top_k,
        }
        self._log(f">> High-value init failed: {reason}", color="red")
        return result

    def _dump_attempt(self, scene: Dict, proposal: Dict, result: Dict, attempt: int):
        if not self.dump_dir:
            return
        scenario_name = scene["scenario"].get("name") or "scenario"
        safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(scenario_name))
        scenario_index = scene["scenario"].get("index")
        repetition_index = scene["scenario"].get("repetition_index")
        if scenario_index is not None:
            safe_name = f"{safe_name}_idx{scenario_index}"
        if repetition_index is not None:
            safe_name = f"{safe_name}_rep{repetition_index}"
        path = os.path.join(self.dump_dir, f"{safe_name}_attempt_{attempt:02d}.json")
        payload = {
            "scene": scene,
            "proposal": self._to_jsonable(proposal),
            "result": self._to_jsonable(self.result_for_info(result)),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if self.save_viz:
            self._dump_viz(scene, proposal, result, attempt, os.path.splitext(path)[0] + ".png")

    def _dump_viz(self, scene: Dict, proposal: Dict, result: Dict, attempt: int, path: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self._log(f">> High-value init viz skipped: {exc}", color="yellow")
            return

        actors = scene.get("actors", [])
        selected_ids = set(result.get("selected_actor_ids", []))
        cbv_ids = set(result.get("cbv_actor_ids", []))
        route = [point for point in scene.get("ego_route", []) if point is not None]
        interaction_by_id = {
            label.get("actor_id"): label
            for label in scene.get("interaction_labels", [])
        }

        fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
        if route:
            ax.plot([p["x"] for p in route], [p["y"] for p in route], color="black", linewidth=2.0, label="ego route")

        ego = scene.get("ego")
        if ego:
            self._plot_actor_box(ax, ego, edgecolor="#1f77b4", linewidth=2.0, label="ego")

        for actor in actors:
            actor_id = actor["actor_id"]
            is_selected = actor_id in selected_ids
            is_cbv = actor_id in cbv_ids
            label = interaction_by_id.get(actor_id, {})
            if is_cbv:
                edgecolor = "#d62728"
            elif is_selected:
                edgecolor = "#ff7f0e"
            elif label.get("is_interactive"):
                edgecolor = "#9467bd"
            else:
                edgecolor = "#c7c7c7"
            linewidth = 2.0 if is_selected else 1.0
            self._plot_actor_box(ax, actor, edgecolor=edgecolor, linewidth=linewidth)

            wp = label.get("interaction_waypoint")
            if wp and label.get("is_interactive"):
                ax.plot([actor["x"], wp["x"]], [actor["y"], wp["y"]], "--", color=edgecolor, linewidth=0.8, alpha=0.55)
                ax.scatter([wp["x"]], [wp["y"]], color=edgecolor, marker="*", s=40)

            if is_selected:
                after_actor = self._apply_proposal_to_actor_state(actor, proposal, actors)
                ax.arrow(
                    actor["x"], actor["y"],
                    after_actor["x"] - actor["x"], after_actor["y"] - actor["y"],
                    color=edgecolor, width=0.15, head_width=1.2, length_includes_head=True, alpha=0.85
                )
                self._plot_actor_box(ax, after_actor, edgecolor="#2ca02c", linewidth=2.0, linestyle="--")
                role = "CBV" if is_cbv else "topK"
                ax.text(after_actor["x"], after_actor["y"], f"{role}:{actor_id}", color=edgecolor, fontsize=7)

        title = scene["scenario"].get("name") or "scenario"
        ax.set_title(
            f"{title} | attempt={attempt} | cbv={sorted(cbv_ids)} | updated={sorted(selected_ids)}"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _plot_actor_box(self, ax, actor: Dict, edgecolor: str, linewidth: float, label=None, linestyle="-"):
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D

        length = max(float(actor.get("length", 4.5)), 0.1)
        width = max(float(actor.get("width", 2.0)), 0.1)
        x = float(actor["x"])
        y = float(actor["y"])
        yaw = float(actor.get("yaw_deg", 0.0))
        rect = Rectangle(
            (-length / 2.0, -width / 2.0),
            length,
            width,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )
        rect.set_transform(Affine2D().rotate_deg(yaw).translate(x, y) + ax.transData)
        ax.add_patch(rect)
        heading = math.radians(yaw)
        ax.arrow(
            x, y,
            0.5 * length * math.cos(heading),
            0.5 * length * math.sin(heading),
            color=edgecolor,
            width=0.05,
            head_width=0.5,
            length_includes_head=True,
            alpha=0.75,
        )

    def _apply_proposal_to_actor_state(self, actor: Dict, proposal: Dict, actors: List[Dict]) -> Dict:
        actor_ids = [item["actor_id"] for item in actors]
        actor_index = actor_ids.index(actor["actor_id"])
        update = self._lookup_actor_update(proposal.get("state_updates"), actor["actor_id"], actor_index) or {}
        after = dict(actor)
        after["x"] = float(update.get("x", after["x"]))
        after["y"] = float(update.get("y", after["y"]))
        if "heading_rad" in update:
            after["yaw_deg"] = math.degrees(float(update["heading_rad"]))
        elif "yaw_deg" in update:
            after["yaw_deg"] = float(update["yaw_deg"])
        return after

    def _to_jsonable(self, value):
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _log(self, message: str, color: Optional[str] = None):
        if self.logger is not None:
            self.logger.log(message, color=color)
