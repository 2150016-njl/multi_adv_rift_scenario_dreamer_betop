from typing import Any, Dict, List, Optional, Set

import matplotlib
from torch import Tensor
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely import Point, Polygon
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from nuplan_plugin.actor_state.state_representation import StateSE2
from nuplan_plugin.actor_state.tracked_objects_types import TrackedObjectType
from nuplan_plugin.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusType,
)
from nuplan_plugin.trajectory.interpolated_trajectory import InterpolatedTrajectory
from .visualize import *

AGENT_COLOR_MAPPING = {
    TrackedObjectType.VEHICLE: "#001eff",  # blue
    TrackedObjectType.PEDESTRIAN: "#FFD700",  # gold
    TrackedObjectType.BICYCLE: "#001eff",  # blue
}

GENERATED_ACTOR_COLOR = "#ff8c00"  # orange
GENERATED_CBV_COLOR = "#00a65a"  # green

TRAFFIC_LIGHT_COLOR_MAPPING = {
    TrafficLightStatusType.GREEN: "#2ca02c",
    TrafficLightStatusType.YELLOW: "#ff7f0e",
    TrafficLightStatusType.RED: "#d62728",
}


def is_obj_in_route(obj, route_ids_set: Set[Tuple[int, int]]) -> bool:
    """
    Check if the road_id of the given obj exists in any set of road_ids
    and whether the corresponding lane_id in lane_ids has the same sign
    as obj.lane_id.

    :param route_ids_set: Set[Tuple[int, int]]
    :return: bool
    """
    for road_id, lane_id in route_ids_set:
        if obj.road_id == road_id and (lane_id * obj.lane_id > 0):
            return True
    return False


class NuplanScenarioRender:
    def __init__(
        self,
        env_params,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
        bounds = 50,
        offset = 15,
        map_radius = 120,

    ) -> None:
        super().__init__()

        self.img_size = env_params["img_size"]
        self.future_horizon = future_horizon
        self.future_samples = int(self.future_horizon / sample_interval)
        self.sample_interval = sample_interval
        self.bounds = float(env_params.get("video_render_bounds", bounds))
        self.offset = float(env_params.get("video_render_offset", offset))
        self.map_radius = float(env_params.get("video_render_map_radius", map_radius))

        self.candidate_index = None
        self._history_trajectory = {}

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.static_objects_types = [
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.BARRIER,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        self.road_elements = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
        ]

    def render(
        self,
        ego_states: Dict[int, np.array],
        nearby_agents_states: Dict[int, np.array],
        CBV_states: Dict[int, np.array] = {},
        route_ids_list: List[Dict[str, Set[int]]]= [],
        reference_lines_list: List[List[np.ndarray]] = [],
        route_waypoints_list: List[List[StateSE2]] = [],
        interaction_wp_list: List[StateSE2] = [],
        planning_trajectory_list: List[np.ndarray] = [],
        candidate_trajectories_list: List[np.ndarray] = [],
        candidate_index_list:List[Any] = [],
        rollout_trajectories_list: List[Any] = None,  # currently is None
        predictions_list: List[np.ndarray] = [],
        CBV_teacher_infos: Dict[int, torch.Tensor] = {},
        scenario_dreamer_actor_ids: List[int] = None,
        scenario_dreamer_cbv_actor_ids: List[int] = None,
        generated_actor_ids: List[int] = None,
        generated_cbv_actor_ids: List[int] = None,
    ):  
        # render for each small scenario
        if len(ego_states) == 1:
            ego_id, ego_state = next(iter(ego_states.items()))
        else:
            raise ValueError("Ego id should be the same for different CBV")
        generated_actor_id_set = {int(actor_id) for actor_id in (generated_actor_ids or [])}
        generated_cbv_id_set = {int(actor_id) for actor_id in (generated_cbv_actor_ids or [])}
        scenario_dreamer_actor_id_set = {int(actor_id) for actor_id in (scenario_dreamer_actor_ids or [])}
        scenario_dreamer_cbv_id_set = {int(actor_id) for actor_id in (scenario_dreamer_cbv_actor_ids or [])}

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.set_size_inches(self.img_size[1] / fig.dpi, self.img_size[1] / fig.dpi)

        self.candidate_index = candidate_index_list
        if ego_id not in self._history_trajectory:
            self._history_trajectory[ego_id] = {ego_id: [ego_state.rear_axle.array]}
        else:
            self._history_trajectory[ego_id][ego_id].append(ego_state.rear_axle.array)

        # the agent state is already under right-hand system
        self.origin = ego_state.rear_axle.array
        self.angle = ego_state.rear_axle.heading
        self.rot_mat = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ],
            dtype=np.float64,
        )

        self._plot_map(
            ax,
            CarlaDataProvider.get_map_api(),
            Point(self.origin),
            route_ids_list,
        )

        # plot the ego vehicle
        self._plot_ego(ax, ego_state)            

        # plot all the nearby agents (near ego and CBVs) across the scenario
        for agent_id, agent_state in nearby_agents_states.items():
            if agent_id not in CBV_states.keys() and agent_id != ego_id:
                self._plot_tracked_object(
                    ax,
                    agent_state,
                    agent_id=agent_id,
                    generated_source=(
                        "scenario_dreamer_background" if agent_id in scenario_dreamer_actor_id_set else
                        "high_value" if agent_id in generated_actor_id_set else None
                    ),
                )
        
        # plot all the CBV
        for CBV_id, CBV_state in CBV_states.items():
            self._plot_CBV(
                ax,
                CBV_state,
                CBV_id,
                ego_id,
                generated_source=(
                    "scenario_dreamer_background" if CBV_id in scenario_dreamer_cbv_id_set else
                    "high_value" if (CBV_id in generated_cbv_id_set or CBV_id in generated_actor_id_set) else None
                ),
            )

        # plot the teacher info when necessary
        self._plot_CBV_teacher_info(ax, CBV_teacher_infos, CBV_states)
        
        for (CBV_id, CBV_state), planning_trajs, candidate_trajs in zip(CBV_states.items(), planning_trajectory_list, candidate_trajectories_list):
            # plot the planning trajectories
            transformed_planning = self._transform_trajectories(planning_trajs[None, ...], CBV_state)
            self._plot_planning(ax, transformed_planning)

            # plot the candidates trajectories
            transformed_candidate_trajs = self._transform_trajectories(candidate_trajs, CBV_state)
            self._plot_candidate_trajectories(ax, transformed_candidate_trajs)

        # plot prediciton (transform the predictions from CBV coord to ego coord)
        if predictions_list:
            for (CBV_id, CBV_state), predictions in zip(CBV_states.items(), predictions_list):
                if predictions is not None:
                    transformed_predictions = self._transform_predictions(predictions, CBV_state)
                    self._plot_prediction(ax, transformed_predictions)
        
        if rollout_trajectories_list:
            for idx, ((CBV_id, CBV_state), rollout_trajs) in enumerate(zip(CBV_states.items(), rollout_trajectories_list)):
                if rollout_trajs is not None:
                    transformed_rollout_trajs = self._transform_trajectories(rollout_trajs, CBV_state)
                    self._plot_rollout_trajectories(ax, transformed_rollout_trajs, idx)

        # plot the reference lines
        for reference_lines in reference_lines_list:
            self._plot_reference_lines(ax, reference_lines)

        # # plot the CBV route
        # for route_waypoints in route_waypoints_list:
        #     self._plot_route_waypoints(ax, route_waypoints[:-1])
        
        # plot the interaction waypoints
        for interaction_wp in interaction_wp_list:
            self._plot_interaction_waypoint(ax, interaction_wp)

        # plot the mission point
        for route_waypoints in route_waypoints_list:
            self._plot_mission_point(ax, route_waypoints[-1])
        
        # plot the history of the CBV and Ego
        self._plot_history(
            ax,
            CBV_states,
            ego_id,
            high_value_cbv_ids=generated_cbv_id_set | generated_actor_id_set,
            scenario_dreamer_cbv_ids=scenario_dreamer_cbv_id_set | scenario_dreamer_actor_id_set,
        )

        ax.axis("equal")
        ax.set_xlim(xmin=-self.bounds + self.offset, xmax=self.bounds + self.offset)
        ax.set_ylim(ymin=-self.bounds, ymax=self.bounds)
        ax.axis("off")
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            int(height), int(width), 3
        )
        plt.close(fig)
        return img

    def _plot_map(
        self,
        ax,
        map_api: CarlaMap,
        query_point: Point,
        route_ids_list: List[Dict[str, Set[int]]] = [],
    ):
        map_objects = map_api.query_proximal_map_data(query_point, self.map_radius)
        road_objects = (
            map_objects[SemanticMapLayer.LANE]
            + map_objects[SemanticMapLayer.LANE_CONNECTOR]
        )

        road_ids_list, lane_ids_list = [], []
        for route_ids in route_ids_list:
            road_ids_list.extend(route_ids["road_ids"])
            lane_ids_list.extend(route_ids["lane_ids"])
        route_ids_set = set(zip(road_ids_list, lane_ids_list))
        for obj in road_objects:

            kwargs = {"color": "lightgray", "alpha": 0.4, "ec": None, "zorder": 0}
            if is_obj_in_route(obj, route_ids_set):
                kwargs["color"] = "dodgerblue"
                kwargs["alpha"] = 0.1
                kwargs["zorder"] = 1
            if obj.polygon is not None:
                ax.add_artist(self._polygon_to_patch(obj.polygon, **kwargs))

            # for stopline in obj.stop_lines:
            #     if stopline.id in plotted_stopline:
            #         continue
            #     kwargs = {"color": "k", "alpha": 0.3, "ec": None, "zorder": 1}
            #     ax.add_artist(self._polygon_to_patch(stopline.polygon, **kwargs))
            #     plotted_stopline.add(stopline.id)

            cl_color, linewidth = "gray", 1.0
            cl = np.array([[s.x, s.y] for s in obj.center_states])
            cl = np.matmul(cl - self.origin, self.rot_mat)
            
            # handle the seperated center states and plot each segment
            threshold_sq = 20 ** 2
            dx = np.diff(cl[:, 0])
            dy = np.diff(cl[:, 1])
            squared_distances = dx * dx + dy * dy
            split_indices = np.where(squared_distances > threshold_sq)[0]
            split_points = np.concatenate([[0], split_indices + 1, [len(cl)]])
            for i in range(len(split_points) - 1):
                start = split_points[i]
                end = split_points[i + 1]
                if end - start >= 2:
                    segment = cl[start:end]
                    ax.plot(
                        segment[:, 0],
                        segment[:, 1],
                        color=cl_color,
                        alpha=0.5,
                        linestyle="--",
                        zorder=1,
                        linewidth=linewidth,
                    )

            # cl = np.array([[s.x, s.y] for s in obj.center_states])
            # cl = np.matmul(cl - self.origin, self.rot_mat)
            # ax.plot(
            #     cl[:, 0],
            #     cl[:, 1],
            #     color=cl_color,
            #     alpha=0.5,
            #     linestyle="--",
            #     zorder=1,
            #     linewidth=linewidth,
            # )

        for obj in map_objects[SemanticMapLayer.CROSSWALK]:
            xys = np.array(obj.polygon.exterior.coords.xy).T
            xys = np.matmul(xys - self.origin, self.rot_mat)
            polygon = Polygon(
                xys, color="gray", alpha=0.4, ec=None, zorder=3, hatch="///"
            )
            ax.add_patch(polygon)

    def _plot_ego(self, ax, ego_state: CarlaAgentState, gt=False):
        kwargs = {"lw": 1.5}
        if gt:
            ax.add_patch(
                self._polygon_to_patch(
                    ego_state.car_footprint.geometry,
                    color="gray",
                    alpha=0.3,
                    zorder=9,
                    **kwargs,
                )
            )
        else:
            ax.add_patch(
                self._polygon_to_patch(
                    ego_state.car_footprint.geometry,
                    ec="#FF0000",  # red
                    alpha=0.8,
                    fill=False,
                    zorder=10,
                    **kwargs,
                )
            )

        ax.plot(
            [1.69, 1.69 + ego_state.car_footprint.length * 0.75],
            [0, 0],
            color="#FF0000",  # red
            alpha=0.8,
            linewidth=1.5,
            zorder=11,
        )

    def _plot_CBV(
            self,
            ax,
            agent_state: CarlaAgentState,
            agent_id: int,
            ego_id,
            generated_source: Optional[str] = None):
        if agent_id not in self._history_trajectory[ego_id]:
            self._history_trajectory[ego_id][agent_id] = [agent_state.rear_axle.array]
        else:
            self._history_trajectory[ego_id][agent_id].append(agent_state.rear_axle.array)

        # the agent state already under right hand system
        center, angle = agent_state.center.array, agent_state.center.heading
        center = np.matmul(center - self.origin, self.rot_mat)
        angle = angle - self.angle

        direct = np.array([np.cos(angle), np.sin(angle)]) * agent_state.car_footprint.length / 1.5
        direct = np.stack([center, center + direct], axis=0)

        color = GENERATED_CBV_COLOR if generated_source == "high_value" else "#9500ff"
        linewidth = 2.5 if generated_source else 1.5
        # draw the bbox
        ax.add_patch(
            self._polygon_to_patch(
                agent_state.car_footprint.geometry, ec=color, fill=False, alpha=1.0, zorder=4, lw=linewidth
            )
        )
        # draw the arrow
        ax.plot(direct[:, 0], direct[:, 1], color=color, linewidth=1.5 if generated_source else 1, zorder=4)
        if generated_source:
            ax.text(
                center[0],
                center[1],
                f"{'HV' if generated_source == 'high_value' else 'SD'}-CBV {agent_id}",
                color=color,
                fontsize=7,
                fontweight="bold",
                zorder=12,
            )

    def _plot_tracked_object(
            self,
            ax,
            agent_state: CarlaAgentState,
            agent_id: int = None,
            generated_source: Optional[str] = None):
        center, angle = agent_state.center.array, agent_state.center.heading  # the agent state already under right hand system

        center = np.matmul(center - self.origin, self.rot_mat)
        angle = angle - self.angle

        direct = np.array([np.cos(angle), np.sin(angle)]) * agent_state.car_footprint.length / 1.5
        direct = np.stack([center, center + direct], axis=0)

        color = (
            GENERATED_ACTOR_COLOR if generated_source == "high_value" else
            AGENT_COLOR_MAPPING.get(agent_state.agent_state.tracked_object_type, "k")
        )
        ax.add_patch(
            self._polygon_to_patch(
                agent_state.car_footprint.geometry,
                ec=color,
                fill=False,
                alpha=1.0,
                zorder=4,
                lw=2.0 if generated_source else 1.5,
                linestyle="--" if generated_source else "-",
            )
        )

        if color != "k":
            ax.plot(direct[:, 0], direct[:, 1], color=color, linewidth=1, zorder=4)
        if generated_source and agent_id is not None:
            label = "HV" if generated_source == "high_value" else "SD"
            ax.text(center[0], center[1], f"{label} {agent_id}", color=color, fontsize=7, fontweight="bold", zorder=12)

    def _polygon_to_patch(self, polygon: shapely.geometry.Polygon, **kwargs):
        polygon = np.array(polygon.exterior.xy).T
        polygon = np.matmul(polygon - self.origin, self.rot_mat)
        return patches.Polygon(polygon, **kwargs)

    def _plot_planning(self, ax, planning_trajectory: np.ndarray):
        plot_polyline(
            ax,
            [planning_trajectory[0]],
            linewidth=4,
            arrow=False,
            zorder=6,
            alpha=1.0,
            cmap="spring",
        )

    def _plot_candidate_trajectories(self, ax, candidate_trajectories: np.ndarray):
        for traj in candidate_trajectories:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="gray",
                alpha=0.5,
                zorder=5,
                linewidth=2,
            )
            ax.scatter(traj[-1, 0], traj[-1, 1], color="gray", zorder=5, s=10)

    def _plot_rollout_trajectories(self, ax, candidate_trajectories: np.ndarray, idx: int):
        for i, traj in enumerate(candidate_trajectories):
            kwargs = {"lw": 1.5, "zorder": 5, "color": "cyan"}
            if self.candidate_index[idx] is not None and i == self.candidate_index[idx]:
                kwargs = {"lw": 5, "zorder": 6, "color": "red"}
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, **kwargs)
            ax.scatter(traj[-1, 0], traj[-1, 1], color="cyan", zorder=5, s=10)

    def _plot_prediction(self, ax, predictions: np.ndarray):
        kwargs = {"lw": 3}
        for pred in predictions:
            pred = pred[:40, ..., :2]
            self._plot_polyline(ax, pred, cmap="winter", **kwargs)

    def _plot_polyline(self, ax, polyline, cmap="spring", **kwargs) -> None:
        arc = get_polyline_arc_length(polyline)
        polyline = polyline.reshape(-1, 1, 2)
        segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
        norm = plt.Normalize(arc.min(), arc.max())
        lc = LineCollection(
            segment,
            cmap=cmap,
            norm=norm,
            array=arc,
            **kwargs,
        )
        ax.add_collection(lc)

    def _plot_reference_lines(self, ax, ref_lines):
        for ref_line in ref_lines:
            ref_line_pos = np.matmul(ref_line[::20, :2] - self.origin, self.rot_mat)
            ref_line_angle = ref_line[::20, 2] - self.angle
            for p, angle in zip(ref_line_pos, ref_line_angle):
                ax.arrow(
                    p[0],
                    p[1],
                    np.cos(angle) * 1.5,
                    np.sin(angle) * 1.5,
                    color="magenta",
                    width=0.2,
                    head_width=0.8,
                    zorder=6,
                    alpha=0.2,
                )

    def _plot_route_waypoints(self, ax, route_waypoints: List[StateSE2]):
        for waypoint in route_waypoints:
            point = np.matmul(waypoint.point.array - self.origin, self.rot_mat)
            ax.plot(point[0], point[1], marker="*", markersize=5, color="gold", zorder=6)

    def _plot_interaction_waypoint(self, ax, interaction_wp: StateSE2):
        point = np.matmul(interaction_wp.point.array - self.origin, self.rot_mat)
        ax.plot(point[0], point[1], marker="*", markersize=10, color="red", zorder=6)

    def _plot_mission_point(self, ax, mission_point:StateSE2):
        point = np.matmul(mission_point.point.array - self.origin, self.rot_mat)
        ax.plot(point[0], point[1], marker="*", markersize=10, color="gold", zorder=6)

    def _plot_plant_wps(self, ax, plant_wps: Tensor):
        '''
        plant_wps: Tensor, shape (4, 2)
        '''
        ax.plot(plant_wps[:, 0], plant_wps[:, 1], marker="o", markersize=5, color="gold", zorder=6)

    def _plot_CBV_teacher_info(self, ax, CBV_teacher_infos: Dict[int, torch.Tensor], CBV_states: Dict[int, np.array]):
        for CBV_id, CBV_state in CBV_states.items():
            if CBV_id in CBV_teacher_infos:
                target_speed, x, y, heading, _ = CBV_teacher_infos[CBV_id]

                # escape the case when the target speed is too small
                if target_speed < 0.2:
                    target_speed = 0.2

                x, y = np.matmul([x, y] - self.origin, self.rot_mat)
                heading = heading - self.angle
                ax.arrow(
                    x,
                    y,
                    np.cos(heading) * target_speed,
                    np.sin(heading) * target_speed,
                    color="#001eff",
                    width=0.8,
                    head_width=1.0,
                    zorder=7,
                    alpha=0.3,
                )

    def _plot_history(
            self,
            ax,
            CBV_states: Dict[int, np.array],
            ego_id: int,
            high_value_cbv_ids: Set[int] = None,
            scenario_dreamer_cbv_ids: Set[int] = None,
            interval: int = 10):
        CBV_id_set = set(CBV_states.keys())
        high_value_cbv_ids = set() if high_value_cbv_ids is None else high_value_cbv_ids
        scenario_dreamer_cbv_ids = set() if scenario_dreamer_cbv_ids is None else scenario_dreamer_cbv_ids
        id_need_clean = []
        for agent_id, history_trajectory in self._history_trajectory[ego_id].items():
            if agent_id == ego_id:
                color = "#FF0000"  # red
            elif agent_id in high_value_cbv_ids:
                color = GENERATED_CBV_COLOR
            elif agent_id in scenario_dreamer_cbv_ids:
                color = "#9500ff"
            elif agent_id in CBV_id_set:
                color = "#9500ff"  # purple
            else:
                id_need_clean.append(agent_id)
                continue
            history = np.array(history_trajectory)
            history = np.matmul(history - self.origin, self.rot_mat)
            # plot the trajectory lines
            ax.plot(
                history[:, 0],
                history[:, 1],
                color=color,
                alpha=0.4,
                zorder=6,
                linewidth=2,
            )
            # plot the scatter
            ax.scatter(
                history[::interval, 0], 
                history[::interval, 1], 
                color=color, 
                s=15, 
                alpha=0.4, 
                zorder=7,
            )

        
        for agent_id in id_need_clean:
            del self._history_trajectory[ego_id][agent_id]

    def _transform_predictions(self, predictions: np.ndarray, CBV_state: CarlaAgentState):
        # from CBV to global
        CBV_yaw = CBV_state.rear_axle.heading
        R_cbv = np.array([[np.cos(CBV_yaw), -np.sin(CBV_yaw)],
                        [np.sin(CBV_yaw),  np.cos(CBV_yaw)]])
        global_predictions = np.matmul(predictions[..., :2], R_cbv.T) + CBV_state.rear_axle.array

        # from global to ego
        transformed_predicitons = np.matmul(global_predictions - self.origin, self.rot_mat)

        return transformed_predicitons
    
    def _transform_plant_wps(self, plant_wps: Tensor, CBV_state: CarlaAgentState):
        '''
        plant_wps: Tensor, shape (4, 2)
        '''
        # from CBV to global
        CBV_yaw = CBV_state.rear_axle.heading
        R_cbv = np.array([[np.cos(CBV_yaw), -np.sin(CBV_yaw)],
                        [np.sin(CBV_yaw),  np.cos(CBV_yaw)]])
        global_predictions = np.matmul(plant_wps.numpy(), R_cbv.T) + CBV_state.rear_axle.array

        # from global to ego
        transformed_plant_wps = np.matmul(global_predictions - self.origin, self.rot_mat)

        return transformed_plant_wps

    def _transform_trajectories(self, global_trajectory: np.ndarray, CBV_state: CarlaAgentState):
        global_angle = global_trajectory[..., 2]
        # global to ego
        CBV_origin = CBV_state.rear_axle.array
        delta = CBV_origin - global_trajectory[..., 0, :2]
        adjusted_global_pos = global_trajectory[..., :2] + delta[:, np.newaxis, :]

        position = np.matmul(adjusted_global_pos - self.origin, self.rot_mat)
        heading = global_angle - self.angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def clean_up(self):
        self._history_trajectory = {}
