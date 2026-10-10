"""High-value scene dataset utilities for RIFT -> Scenario Dreamer.

The training records here are scene-generation records. They intentionally do
not contain target masks, actor scores, CBV ids, or delta targets.
"""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cfgs.config import BEFORE_PARTITION, PARTITIONED
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


def read_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: Iterable[Dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def state_generation_schema_is_clean(record: Dict) -> bool:
    forbidden = ("target_mask", "actor_scores", "cbv_actor_ids", "selected_actor_ids", "state_delta", "state_deltas")
    text = json.dumps(record, ensure_ascii=False)
    return not any(name in text for name in forbidden)


class HighValueSceneJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.path = jsonl_path
        self.records = read_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


class HighValueSceneGraphBuilder:
    """Build Scenario Dreamer graph objects from high-value generation records."""

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.fov = float(cfg.get("fov", cfg.get("hv_sd_fov", NUPLAN_STATS["fov"])))
        self.num_points_per_lane = int(cfg.get("num_points_per_lane", cfg.get("hv_sd_num_points_per_lane", 20)))
        self.max_num_lanes = int(cfg.get("max_num_lanes", cfg.get("hv_sd_max_num_lanes", 100)))
        self.max_num_agents = int(cfg.get("max_num_agents", cfg.get("hv_sd_max_num_agents", 61)))
        # Legacy datasets may retain the historical nearest-actor truncation.
        # New contracts can opt in to rejection so context is never silently
        # changed between their audited JSONL and the LDM graph.
        self.reject_observed_over_capacity = bool(cfg.get("reject_observed_over_capacity", False))
        self.route_num_points = int(cfg.get("route_num_points", cfg.get("hv_route_num_points", 80)))
        self.map_id = int(cfg.get("map_id", cfg.get("hv_sd_map_id", 0)))

    def build_ae_data(self, record: Dict, include_targets: bool = True) -> Tuple[ScenarioDreamerData, Dict]:
        condition = record["condition"]
        observed = list(condition.get("observed_agent_states", []))
        target = self._target_states(record) if include_targets else []
        if 1 + len(target) > self.max_num_agents:
            raise ValueError(
                f"target agent count {len(target)} exceeds Scenario Dreamer capacity "
                f"{self.max_num_agents - 1}"
            )
        max_observed = max(0, self.max_num_agents - 1 - len(target))
        if self.reject_observed_over_capacity and len(observed) > max_observed:
            raise ValueError(
                f"observed context ({len(observed)}) exceeds capacity {max_observed} after reserving "
                f"ego and {len(target)} target agents; refusing to truncate context"
            )
        observed = observed[:max_observed]
        ego_state = condition.get("ego_state", [0.0, 0.0, 0.0, 0.0, 5.176, 2.297, 1])
        agent_states, agent_types = self._agent_arrays(ego_state, observed + target)
        lane_states, lane_types, lane_conn = self._lane_arrays(condition)

        agent_states_norm, lane_states_norm = normalize_scene(
            agent_states.copy(),
            lane_states.copy(),
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

        num_agents = int(agent_states_norm.shape[0])
        num_lanes = int(lane_states_norm.shape[0])
        data = ScenarioDreamerData()
        data["idx"] = int(record.get("idx", 0))
        data["num_agents"] = torch.tensor(num_agents, dtype=torch.long)
        data["num_lanes"] = torch.tensor(num_lanes, dtype=torch.long)
        data["lg_type"] = torch.tensor(PARTITIONED, dtype=torch.long)
        data["map_id"] = torch.tensor(self.map_id, dtype=torch.long)
        data["num_agents_after_origin"] = torch.tensor(len(target), dtype=torch.long)
        data["num_lanes_after_origin"] = torch.tensor(0, dtype=torch.long)
        data["agent"].x = torch.from_numpy(agent_states_norm).float()
        data["agent"].type = torch.from_numpy(agent_types).float()
        data["agent"].partition_mask = torch.ones(num_agents, dtype=torch.bool)
        data["lane"].x = torch.from_numpy(lane_states_norm).float()
        data["lane"].type = torch.from_numpy(lane_types).float()
        data["lane"].partition_mask = torch.ones(num_lanes, dtype=torch.bool)

        l2l = get_edge_index_complete_graph(num_lanes).long()
        a2a = get_edge_index_complete_graph(num_agents).long()
        l2a = get_edge_index_bipartite(num_lanes, num_agents).long()
        data["lane", "to", "lane"].edge_index = l2l
        data["lane", "to", "lane"].type = torch.from_numpy(lane_conn).float()
        data["agent", "to", "agent"].edge_index = a2a
        data["lane", "to", "agent"].edge_index = l2a
        data["lane", "to", "lane"].encoder_mask = torch.ones(l2l.shape[1], dtype=torch.bool)
        data["lane", "to", "agent"].encoder_mask = torch.ones(l2a.shape[1], dtype=torch.bool)
        data["agent", "to", "agent"].encoder_mask = torch.ones(a2a.shape[1], dtype=torch.bool)

        route_features = self.route_features(condition)
        data["route"].x = torch.from_numpy(route_features).float()

        meta = {
            "context_agent_count": 1 + len(observed),
            "target_agent_count": len(target),
            "num_agents": num_agents,
            "num_lanes": num_lanes,
            "mode": self.mode_from_record(record),
            "observed_agent_count_kept": len(observed),
        }
        return data, meta

    def build_ldm_seed(self, record: Dict, agent_latent_dim: int, lane_latent_dim: int) -> Tuple[ScenarioDreamerData, Dict]:
        """Build an LDM seed for generation from Gaussian noise.

        Context agents and lanes are conditioned through ``mask=True`` and
        generated slots are appended as anonymous latent slots with ``mask=False``.
        """
        condition_only, meta = self.build_ae_data(record, include_targets=False)
        context_agent_count = int(meta["context_agent_count"])
        num_generate = int(record["condition"].get("num_generate_agents", len(self._target_states(record))))
        total_agents = context_agent_count + num_generate
        if total_agents > self.max_num_agents:
            raise ValueError(
                f"context ({context_agent_count}) + generated ({num_generate}) agents exceed "
                f"Scenario Dreamer capacity {self.max_num_agents}"
            )
        num_lanes = int(condition_only["num_lanes"])

        data = ScenarioDreamerData()
        data["idx"] = int(record.get("idx", 0))
        data["num_agents"] = torch.tensor(total_agents, dtype=torch.long)
        data["num_lanes"] = torch.tensor(num_lanes, dtype=torch.long)
        data["lg_type"] = torch.tensor(PARTITIONED, dtype=torch.long)
        data["map_id"] = torch.tensor(self.map_id, dtype=torch.long)
        data["agent"].x = torch.zeros((total_agents, int(agent_latent_dim)), dtype=torch.float32)
        data["lane"].x = torch.zeros((num_lanes, int(lane_latent_dim)), dtype=torch.float32)
        data["agent"].mask = torch.zeros(total_agents, dtype=torch.bool)
        data["agent"].mask[:context_agent_count] = True
        data["lane"].mask = torch.ones(num_lanes, dtype=torch.bool)
        data["agent"].partition_mask = torch.zeros(total_agents, dtype=torch.bool)
        data["agent"].partition_mask[:context_agent_count] = BEFORE_PARTITION
        data["lane"].partition_mask = torch.ones(num_lanes, dtype=torch.bool)

        data["lane", "to", "lane"].edge_index = get_edge_index_complete_graph(num_lanes).long()
        data["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(total_agents).long()
        data["lane", "to", "agent"].edge_index = get_edge_index_bipartite(num_lanes, total_agents).long()
        data["route"].x = condition_only["route"].x.clone()

        meta.update({
            "context_agent_count": context_agent_count,
            "target_agent_count": num_generate,
            "num_agents": total_agents,
            "num_lanes": num_lanes,
        })
        return data, meta

    def route_features(self, condition: Dict) -> np.ndarray:
        route = condition.get("ego_route", {})
        if isinstance(route, dict):
            points = route.get("polyline") or route.get("points") or []
        else:
            points = route or []
        xy = []
        for point in points:
            if isinstance(point, dict):
                xy.append([float(point["x"]), float(point["y"])])
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                xy.append([float(point[0]), float(point[1])])
        if len(xy) < 2:
            xy = [[0.0, 0.0], [1.0, 0.0]]
        xy = resample_polyline(np.asarray(xy, dtype=np.float32), self.route_num_points)[:, :2]
        delta = np.gradient(xy, axis=0)
        heading = np.arctan2(delta[:, 1], delta[:, 0])
        s = np.zeros((xy.shape[0],), dtype=np.float32)
        if xy.shape[0] > 1:
            dist = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
            s[1:] = np.cumsum(dist)
            if s[-1] > 1e-6:
                s = s / s[-1]
        return np.stack([xy[:, 0], xy[:, 1], np.cos(heading), np.sin(heading), s], axis=-1).astype(np.float32)

    def target_local_states_tensor(self, record: Dict) -> torch.Tensor:
        return torch.tensor(self._target_states(record), dtype=torch.float32)

    def mode_from_record(self, record: Dict) -> str:
        schema = record.get("schema_version", "")
        if "full_generation" in schema:
            return "full_generation"
        return "inpainting"

    def _target_states(self, record: Dict) -> List[List[float]]:
        target = record.get("target", {})
        return list(target.get("generated_agent_states") or target.get("agent_states") or [])

    def _agent_arrays(self, ego_state: List[float], non_ego_states: List[List[float]]):
        rows = [self._state_to_sd(ego_state, is_ego=True)]
        types = [get_object_type_onehot_nuplan("vehicle")]
        for state in non_ego_states:
            rows.append(self._state_to_sd(state, is_ego=False))
            types.append(get_object_type_onehot_nuplan("vehicle"))
        return np.asarray(rows, dtype=np.float32), np.asarray(types, dtype=np.float32)

    def _state_to_sd(self, state: List[float], is_ego: bool = False) -> List[float]:
        if len(state) < 6:
            raise ValueError("agent state must be [x, y, heading, speed, length, width, ...]")
        x = 0.0 if is_ego else float(state[0])
        y = 0.0 if is_ego else float(state[1])
        heading = 0.0 if is_ego else float(state[2])
        speed = float(state[3])
        length = float(state[4])
        width = float(state[5])
        return [x, y, speed, math.cos(heading), math.sin(heading), length, width]

    def _lane_arrays(self, condition: Dict):
        lane_polylines = list(condition.get("map", {}).get("lane_polylines", []))
        if not lane_polylines:
            lane_polylines = [condition.get("ego_route", {}).get("polyline", [])]
        lanes = []
        lane_types = []
        for polyline in lane_polylines:
            xy = []
            for point in polyline:
                if isinstance(point, dict):
                    xy.append([float(point["x"]), float(point["y"])])
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    xy.append([float(point[0]), float(point[1])])
            if len(xy) < 2:
                continue
            lane = resample_polyline(np.asarray(xy, dtype=np.float32), self.num_points_per_lane)[:, :2]
            if self._lane_inside_fov(lane):
                lanes.append(lane)
                lane_types.append(get_lane_type_onehot_nuplan("lane"))
        if not lanes:
            lanes = [np.zeros((self.num_points_per_lane, 2), dtype=np.float32)]
            lane_types = [get_lane_type_onehot_nuplan("lane")]
        if len(lanes) > self.max_num_lanes:
            distances = np.asarray([np.linalg.norm(lane, axis=-1).min() for lane in lanes])
            keep = np.argsort(distances)[:self.max_num_lanes]
            lanes = [lanes[int(i)] for i in keep]
            lane_types = [lane_types[int(i)] for i in keep]
        lane_states = np.asarray(lanes, dtype=np.float32)
        lane_types_arr = np.asarray(lane_types, dtype=np.float32)
        lane_conn = self._lane_connections(lane_states.shape[0])
        return lane_states, lane_types_arr, lane_conn

    def _lane_connections(self, num_lanes: int) -> np.ndarray:
        edges = get_edge_index_complete_graph(num_lanes).t().tolist()
        rows = []
        for src, dst in edges:
            rows.append(get_lane_connection_type_onehot_nuplan("self" if src == dst else "none"))
        return np.asarray(rows, dtype=np.float32)

    def _lane_inside_fov(self, lane: np.ndarray) -> bool:
        half = self.fov / 2.0
        return bool(np.any((np.abs(lane[:, 0]) <= half) & (np.abs(lane[:, 1]) <= half)))


def decoded_sd_agents_to_hv_states(agent_states: torch.Tensor, start: int, count: int) -> torch.Tensor:
    """Convert decoded Scenario Dreamer agent states to HV local states."""
    selected = agent_states[start:start + count]
    heading = torch.atan2(selected[:, 4], selected[:, 3]).unsqueeze(-1)
    return torch.cat([
        selected[:, 0:2],
        heading,
        selected[:, 2:3],
        selected[:, 5:7],
        torch.ones((selected.shape[0], 1), dtype=selected.dtype, device=selected.device),
    ], dim=-1)
