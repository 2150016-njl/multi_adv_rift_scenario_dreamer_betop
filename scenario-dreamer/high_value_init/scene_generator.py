"""Scenario Dreamer generator entry point used by RIFT online evaluation."""

import math
from typing import Dict, List, Tuple

from high_value_init.high_value_ldm import RouteConditionedHighValueLDM


def _local_xy(ego: Dict, x: float, y: float) -> Tuple[float, float]:
    dx = float(x) - float(ego["x"])
    dy = float(y) - float(ego["y"])
    yaw = math.radians(float(ego.get("yaw_deg", 0.0)))
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    return cos_yaw * dx - sin_yaw * dy, sin_yaw * dx + cos_yaw * dy


def _local_heading(ego: Dict, yaw_deg: float) -> float:
    return math.atan2(
        math.sin(math.radians(float(yaw_deg) - float(ego.get("yaw_deg", 0.0)))),
        math.cos(math.radians(float(yaw_deg) - float(ego.get("yaw_deg", 0.0)))),
    )


def _actor_to_state(ego: Dict, actor: Dict) -> List[float]:
    x, y = _local_xy(ego, actor["x"], actor["y"])
    return [
        float(x),
        float(y),
        float(_local_heading(ego, actor.get("yaw_deg", 0.0))),
        float(actor.get("speed_mps", 0.0)),
        float(actor.get("length", 4.5)),
        float(actor.get("width", 2.0)),
        1.0,
    ]


def _point_to_local(ego: Dict, point) -> List[float]:
    if isinstance(point, dict):
        return list(_local_xy(ego, point["x"], point["y"]))
    return list(_local_xy(ego, point[0], point[1]))


def _snapshot_to_record(
        snapshot: Dict,
        mode: str,
        num_generate_agents: int,
        actor_radius: float,
        fov: float) -> Dict:
    # ``full_fov_inpainting`` is a V4-A request name.  It has the same graph
    # representation as inpainting, but makes the no-hidden/full-FOV online
    # contract explicit in telemetry and configuration.
    model_mode = "inpainting" if mode in ("sparse_inpainting", "full_fov_inpainting") else mode
    ego = snapshot["ego"]
    observed = []
    half_fov = float(fov) / 2.0
    if model_mode == "inpainting":
        for actor in snapshot.get("agents", []):
            distance = float(actor.get("ego_distance", math.hypot(actor["x"] - ego["x"], actor["y"] - ego["y"])))
            if distance > actor_radius:
                continue
            state = _actor_to_state(ego, actor)
            # Training records retain only the fov=64 ego-local square.  Keep
            # the online condition in precisely that same support.
            if abs(state[0]) <= half_fov and abs(state[1]) <= half_fov:
                observed.append(state)

    route_polyline = [_point_to_local(ego, point) for point in snapshot.get("ego_route", [])]
    if len(route_polyline) < 2:
        route_polyline = [[0.0, 0.0], [1.0, 0.0]]

    lanes = []
    for lane in snapshot.get("map_lanes", []):
        polyline = [_point_to_local(ego, point) for point in lane.get("centerline", [])]
        if len(polyline) >= 2:
            lanes.append(polyline)

    condition = {
        "ego_state": [
            0.0,
            0.0,
            0.0,
            float(ego.get("speed_mps", 0.0)),
            float(ego.get("length", 5.176)),
            float(ego.get("width", 2.297)),
            1.0,
        ],
        "fov": float(fov),
        "ego_route": {"polyline": route_polyline},
        "map": {"lane_polylines": lanes},
        "num_generate_agents": int(num_generate_agents),
    }
    if model_mode == "inpainting":
        condition["observed_agent_states"] = observed
        condition["observed_agent_types"] = [1 for _ in observed]

    return {
        "schema_version": f"hv_init_{mode}_inference_v1",
        "scene_id": snapshot.get("scene_id", "online_scene"),
        "town": snapshot.get("town"),
        "condition": condition,
        "target": {},
        "source": "rift_online_inference",
    }


class ScenarioDreamerHighValueSceneGenerator:
    """RIFT-facing wrapper around the route-conditioned high-value LDM."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", config: Dict = None):
        config = config or {}
        self.mode = config.get("hv_generation_mode", "inpainting")
        self.model_mode = "inpainting" if self.mode == "sparse_inpainting" else self.mode
        self.num_generate_agents = int(config.get("hv_num_generate_agents", 2))
        self.actor_radius = float(config.get("hv_actor_radius", 80.0))
        self.fov = float(config.get("hv_sd_fov", 64.0))
        autoencoder_checkpoint = config.get("hv_autoencoder_checkpoint")
        ldm_checkpoint = config.get("hv_ldm_checkpoint")
        route_num_points = int(config.get("hv_route_num_points", config.get("hv_max_route_points", 80)))
        clamp_generated_states = bool(config.get("hv_generation_clamp_generated_states", False))
        if checkpoint_path is None:
            raise ValueError("--hv_generation_checkpoint is required for online generation")
        self.model = RouteConditionedHighValueLDM(
            checkpoint_path=checkpoint_path,
            autoencoder_checkpoint=autoencoder_checkpoint,
            ldm_checkpoint=ldm_checkpoint,
            device=device,
            mode=self.model_mode,
            num_generate_agents=self.num_generate_agents,
            route_num_points=route_num_points,
            use_route_condition=True,
            min_generated_speed=float(config.get("hv_min_generated_speed", 0.0)),
            max_generated_speed=float(config.get("hv_max_generated_speed", 30.0)),
            min_generated_length=float(config.get("hv_min_generated_length", 2.0)),
            max_generated_length=float(config.get("hv_max_generated_length", 8.0)),
            min_generated_width=float(config.get("hv_min_generated_width", 1.4)),
            max_generated_width=float(config.get("hv_max_generated_width", 3.2)),
            clamp_generated_states=clamp_generated_states,
        )

    def generate(self, snapshot: Dict) -> Dict:
        record = _snapshot_to_record(
            snapshot,
            mode=self.mode,
            num_generate_agents=self.num_generate_agents,
            actor_radius=self.actor_radius,
            fov=self.fov,
        )
        generated = self.model.generate(record, num_generate_agents=self.num_generate_agents)
        generated.update({
            "scene_id": snapshot.get("scene_id"),
            "coordinate_frame": "ego_local_rh_x_forward",
            "requested_generation_mode": self.mode,
            "model_generation_mode": self.model_mode,
            "rift_labels_used_as_input": False,
            "source": "scenario_dreamer_route_conditioned_ldm",
        })
        return generated
