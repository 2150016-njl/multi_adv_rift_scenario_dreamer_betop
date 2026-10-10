import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils.data_container import ScenarioDreamerData
from utils.data_helpers import (
    get_lane_connection_type_onehot_nuplan,
    get_lane_type_onehot_nuplan,
    get_object_type_onehot_nuplan,
    normalize_scene,
)
from utils.lane_graph_helpers import resample_polyline
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph


NUPLAN_STATS = {
    "fov": 64.0,
    "min_speed": 0.0,
    "max_speed": 29.19,
    "min_length": 0.09,
    "max_length": 19.77,
    "min_width": 0.14,
    "max_width": 7.20,
    "min_lane_x": -32.0,
    "max_lane_x": 32.0,
    "min_lane_y": -32.0,
    "max_lane_y": 32.0,
}


class RiftToScenarioDreamerAdapter:
    """Convert RIFT online scene packages into Scenario Dreamer graph data.

    RIFT owns map/route/conflict computation. This adapter only changes data
    representation so Scenario Dreamer modules can consume the same scene.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg or {}
        self.num_points_per_lane = int(self.cfg.get("hv_sd_num_points_per_lane", 20))
        self.max_num_lanes = int(self.cfg.get("hv_sd_max_num_lanes", 100))
        self.fov = float(self.cfg.get("hv_sd_fov", NUPLAN_STATS["fov"]))

    def build(self, scene: Dict) -> Tuple[ScenarioDreamerData, List[int]]:
        ego = scene["ego"]
        actor_ids = [int(actor["actor_id"]) for actor in scene["actors"]]
        agent_states, agent_types = self._build_agent_arrays(ego, scene["actors"])
        road_points, lane_types, road_connection_types = self._build_lane_arrays(ego, scene)

        if road_points.shape[0] == 0:
            road_points = np.zeros((1, self.num_points_per_lane, 2), dtype=np.float32)
            lane_types = np.asarray([get_lane_type_onehot_nuplan("lane")], dtype=np.float32)

        num_agents = agent_states.shape[0]
        num_lanes = road_points.shape[0]
        edge_index_lane_to_lane = get_edge_index_complete_graph(num_lanes)
        edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents)
        edge_index_lane_to_agent = get_edge_index_bipartite(num_lanes, num_agents)
        if road_connection_types is None:
            road_connection_types = self._connection_types(num_lanes)

        agent_states_norm, road_points_norm = normalize_scene(
            agent_states.copy(),
            road_points.copy(),
            fov=self.fov,
            min_speed=NUPLAN_STATS["min_speed"],
            max_speed=NUPLAN_STATS["max_speed"],
            min_length=NUPLAN_STATS["min_length"],
            max_length=NUPLAN_STATS["max_length"],
            min_width=NUPLAN_STATS["min_width"],
            max_width=NUPLAN_STATS["max_width"],
            min_lane_x=NUPLAN_STATS["min_lane_x"],
            max_lane_x=NUPLAN_STATS["max_lane_x"],
            min_lane_y=NUPLAN_STATS["min_lane_y"],
            max_lane_y=NUPLAN_STATS["max_lane_y"],
        )

        data = ScenarioDreamerData()
        data["idx"] = 0
        data["num_agents"] = num_agents
        data["num_lanes"] = num_lanes
        data["lg_type"] = 0
        data["map_id"] = int(self.cfg.get("hv_sd_map_id", 0))
        data["agent"].x = torch.from_numpy(agent_states_norm).float()
        data["agent"].type = torch.from_numpy(agent_types).float()
        data["agent"].batch = torch.zeros(num_agents, dtype=torch.long)
        data["lane"].x = torch.from_numpy(road_points_norm).float()
        data["lane"].type = torch.from_numpy(lane_types).float()
        data["lane"].batch = torch.zeros(num_lanes, dtype=torch.long)
        data["lane"].partition_mask = torch.zeros(num_lanes, dtype=torch.bool)
        data["num_agents_after_origin"] = 0
        data["num_lanes_after_origin"] = 0
        data["batch_size"] = 1

        data["lane", "to", "lane"].edge_index = edge_index_lane_to_lane.long()
        data["lane", "to", "lane"].type = torch.from_numpy(road_connection_types).float()
        data["agent", "to", "agent"].edge_index = edge_index_agent_to_agent.long()
        data["lane", "to", "agent"].edge_index = edge_index_lane_to_agent.long()
        data["lane", "to", "lane"].encoder_mask = torch.ones(edge_index_lane_to_lane.shape[1], dtype=torch.bool)
        data["lane", "to", "agent"].encoder_mask = torch.ones(edge_index_lane_to_agent.shape[1], dtype=torch.bool)
        data["agent", "to", "agent"].encoder_mask = torch.ones(edge_index_agent_to_agent.shape[1], dtype=torch.bool)

        return data, actor_ids

    def build_ldm_seed(self, scene: Dict, agent_latent_dim: int, lane_latent_dim: int) -> Tuple[ScenarioDreamerData, List[int]]:
        """Build a Scenario Dreamer LDM seed with the same actor/lane counts.

        The LDM fills ``agent.x`` and ``lane.x`` with sampled latents, so these
        tensors are placeholders whose first dimension encodes the desired
        generated scene cardinality. Actor ids stay aligned with
        ``scene["actors"]`` after the ego slot.
        """
        ae_data, actor_ids = self.build(scene)
        num_agents = int(ae_data["num_agents"])
        num_lanes = int(ae_data["num_lanes"])

        data = ScenarioDreamerData()
        data["idx"] = 0
        data["num_agents"] = torch.tensor([num_agents], dtype=torch.long)
        data["num_lanes"] = torch.tensor([num_lanes], dtype=torch.long)
        data["lg_type"] = torch.tensor([0], dtype=torch.long)
        data["map_id"] = torch.tensor([int(self.cfg.get("hv_sd_map_id", 0))], dtype=torch.long)
        data["agent"].x = torch.empty((num_agents, int(agent_latent_dim)), dtype=torch.float32)
        data["lane"].x = torch.empty((num_lanes, int(lane_latent_dim)), dtype=torch.float32)
        data["agent"].batch = torch.zeros(num_agents, dtype=torch.long)
        data["lane"].batch = torch.zeros(num_lanes, dtype=torch.long)
        data["batch_size"] = 1

        data["lane", "to", "lane"].edge_index = ae_data["lane", "to", "lane"].edge_index.clone().long()
        data["agent", "to", "agent"].edge_index = ae_data["agent", "to", "agent"].edge_index.clone().long()
        data["lane", "to", "agent"].edge_index = ae_data["lane", "to", "agent"].edge_index.clone().long()
        return data, actor_ids

    def actor_context_features(self, scene: Dict) -> Tuple[torch.Tensor, List[int]]:
        """Return actor-aligned geometric context features without RIFT labels."""
        ego = scene["ego"]
        rows = []
        actor_ids = []
        for actor in scene["actors"]:
            actor_ids.append(int(actor["actor_id"]))
            rows.append(self._actor_context_row(ego, actor))
        return torch.tensor(rows, dtype=torch.float32).unsqueeze(0), actor_ids

    def local_actor_states(self, scene: Dict) -> Tuple[np.ndarray, List[int]]:
        """Return actor states in the Scenario Dreamer local state convention."""
        ego = scene["ego"]
        rows = []
        actor_ids = []
        for actor in scene["actors"]:
            actor_ids.append(int(actor["actor_id"]))
            rel_x, rel_y = self._world_to_ego_local(ego, actor["x"], actor["y"])
            rel_heading = math.radians(float(actor.get("yaw_deg", 0.0)) - float(ego.get("yaw_deg", 0.0)))
            rows.append([
                rel_x,
                rel_y,
                float(actor.get("speed_mps", 0.0)),
                math.cos(rel_heading),
                math.sin(rel_heading),
                float(actor.get("length", 4.5)),
                float(actor.get("width", 2.0)),
            ])
        return np.asarray(rows, dtype=np.float32), actor_ids

    def _actor_context_row(self, ego: Dict, actor: Dict) -> List[float]:
        rel_x, rel_y = self._world_to_ego_local(ego, actor["x"], actor["y"])
        yaw = math.radians(float(actor.get("yaw_deg", 0.0)) - float(ego.get("yaw_deg", 0.0)))
        vx = float(actor.get("vx", 0.0))
        vy = float(actor.get("vy", 0.0))
        rel_vx, rel_vy = self._rotate_to_ego_local(ego, vx, vy)
        distance = math.sqrt(rel_x ** 2 + rel_y ** 2)
        speed = float(actor.get("speed_mps", math.sqrt(vx ** 2 + vy ** 2)))
        return [
            rel_x / max(self.fov, 1.0),
            rel_y / max(self.fov, 1.0),
            distance / max(self.fov, 1.0),
            math.cos(yaw),
            math.sin(yaw),
            speed / max(NUPLAN_STATS["max_speed"], 1.0),
            rel_vx / max(NUPLAN_STATS["max_speed"], 1.0),
            rel_vy / max(NUPLAN_STATS["max_speed"], 1.0),
            float(actor.get("length", 4.5)) / max(NUPLAN_STATS["max_length"], 1.0),
            float(actor.get("width", 2.0)) / max(NUPLAN_STATS["max_width"], 1.0),
            1.0 if "vehicle" in actor.get("type_id", "") else 0.0,
            1.0,
        ]

    def _build_agent_arrays(self, ego: Dict, actors: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        rows = [self._agent_state_row(ego, ego, is_ego=True)]
        types = [get_object_type_onehot_nuplan("vehicle")]
        for actor in actors:
            rows.append(self._agent_state_row(ego, actor, is_ego=False))
            types.append(get_object_type_onehot_nuplan(self._actor_type(actor)))
        return np.asarray(rows, dtype=np.float32), np.asarray(types, dtype=np.float32)

    def _agent_state_row(self, ego: Dict, actor: Dict, is_ego: bool) -> List[float]:
        if is_ego:
            rel_x, rel_y = 0.0, 0.0
            rel_heading = 0.0
        else:
            rel_x, rel_y = self._world_to_ego_local(ego, actor["x"], actor["y"])
            rel_heading = math.radians(float(actor.get("yaw_deg", 0.0)) - float(ego.get("yaw_deg", 0.0)))
        speed = float(actor.get("speed_mps", 0.0))
        return [
            rel_x,
            rel_y,
            speed,
            math.cos(rel_heading),
            math.sin(rel_heading),
            float(actor.get("length", 4.5)),
            float(actor.get("width", 2.0)),
        ]

    def _actor_type(self, actor: Dict) -> str:
        type_id = actor.get("type_id", "")
        if "walker" in type_id or "pedestrian" in type_id:
            return "pedestrian"
        if "static" in type_id:
            return "static_object"
        return "vehicle"

    def _build_lane_arrays(self, ego: Dict, scene: Dict):
        lane_rows = []
        lane_type_rows = []
        kept_lanes = []
        for lane in scene.get("map_lanes", []):
            points = lane.get("centerline") or []
            if len(points) < 2:
                continue
            local_points = np.asarray(
                [self._world_to_ego_local(ego, point[0], point[1]) for point in points],
                dtype=np.float32,
            )
            if not self._within_fov(local_points):
                continue
            lane_rows.append(resample_polyline(local_points, num_points=self.num_points_per_lane)[:, :2])
            lane_type_rows.append(get_lane_type_onehot_nuplan("lane"))
            kept_lanes.append(lane)

        if len(lane_rows) == 0:
            route = scene.get("ego_route", [])
            route_points = [
                self._world_to_ego_local(ego, point["x"], point["y"])
                for point in route
                if point is not None
            ]
            if len(route_points) >= 2:
                lane_rows.append(resample_polyline(np.asarray(route_points, dtype=np.float32), self.num_points_per_lane)[:, :2])
                lane_type_rows.append(get_lane_type_onehot_nuplan("lane"))

        if len(lane_rows) == 0:
            return (
                np.zeros((0, self.num_points_per_lane, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                None,
            )
        road_points = np.asarray(lane_rows, dtype=np.float32)
        lane_types = np.asarray(lane_type_rows, dtype=np.float32)
        kept_lanes = list(kept_lanes)
        if road_points.shape[0] > self.max_num_lanes:
            distances = np.linalg.norm(road_points, axis=-1).min(axis=1)
            keep_indices = np.argsort(distances)[:self.max_num_lanes]
            road_points = road_points[keep_indices]
            lane_types = lane_types[keep_indices]
            kept_lanes = [kept_lanes[int(idx)] for idx in keep_indices]
        return road_points, lane_types, self._connection_types(road_points.shape[0], kept_lanes)

    def _connection_types(self, num_lanes: int, lanes: List[Dict] = None) -> np.ndarray:
        lanes = lanes or []
        id_to_index = {}
        for idx, lane in enumerate(lanes):
            if lane.get("road_id") is None or lane.get("lane_id") is None:
                continue
            id_to_index[(int(lane["road_id"]), int(lane["lane_id"]))] = idx
        successors = set()
        for src_idx, lane in enumerate(lanes):
            for road_id, lane_id in lane.get("topology", []):
                dst_idx = id_to_index.get((int(road_id), int(lane_id)))
                if dst_idx is not None:
                    successors.add((src_idx, dst_idx))

        edge_index = get_edge_index_complete_graph(num_lanes)
        rows = []
        for src, dst in edge_index.t().tolist():
            if src == dst:
                conn_type = "self"
            elif (src, dst) in successors:
                conn_type = "succ"
            elif (dst, src) in successors:
                conn_type = "pred"
            else:
                conn_type = "none"
            rows.append(get_lane_connection_type_onehot_nuplan(conn_type))
        return np.asarray(rows, dtype=np.float32)

    def _within_fov(self, points: np.ndarray) -> bool:
        half = self.fov / 2.0
        return bool(np.any((np.abs(points[:, 0]) < half) & (np.abs(points[:, 1]) < half)))

    def _world_to_ego_local(self, ego: Dict, x: float, y: float) -> Tuple[float, float]:
        dx = float(x) - float(ego["x"])
        dy = float(y) - float(ego["y"])
        return self._rotate_to_ego_local(ego, dx, dy)

    def _rotate_to_ego_local(self, ego: Dict, x: float, y: float) -> Tuple[float, float]:
        yaw = math.radians(float(ego.get("yaw_deg", 0.0)))
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        return (
            cos_yaw * float(x) - sin_yaw * float(y),
            sin_yaw * float(x) + cos_yaw * float(y),
        )
