#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : nuplan_utils.py
@Date    : 2023/12/5
"""

from typing import List

from shapely import Point
from scipy.spatial import KDTree
import carla

import numba
import numpy as np
from distance3d import gjk, colliders

from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap, CarlaMapObject
from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.util.torch_util import CUDA, CPU
from rift.scenario.tools.timer import GameTime
from nuplan_plugin.actor_state.state_representation import StateSE2, TimePoint


@numba.njit
def linear_map(value, original_range, desired_range):
    """Linear map of value with original range to desired range."""
    return desired_range[0] + (value - original_range[0]) * (desired_range[1] - desired_range[0]) / (original_range[1] - original_range[0])


def store_agent_state(ego_vehicle):
    time_point = TimePoint(GameTime.get_time() * 1e6)  # micro seconds
    # get the all scenario actors managed by this ego
    all_scenario_actors = CarlaDataProvider.get_scenario_actors_by_ego(ego_vehicle.id)
    # store ego state info
    ego_parameters, ego_type = CarlaDataProvider.get_vehicle_info(ego_vehicle)
    CarlaDataProvider.add_history_state(ego_vehicle, CarlaAgentState.build_from_agent(ego_vehicle, ego_parameters, ego_type, time_point))
    
    # loop all type of scenario actors (BVs, CBVs, CBVs_reach_goals)
    for scenario_actors_dict in all_scenario_actors.values():
        for actor in scenario_actors_dict.values():
            # store each agent state into the history state buffer
            vehicle_parameters, vehicle_type = CarlaDataProvider.get_vehicle_info(actor)
            CarlaDataProvider.add_history_state(actor, CarlaAgentState.build_from_agent(actor, vehicle_parameters, vehicle_type, time_point))


def get_actor_off_road(actor):
    current_location = CarlaDataProvider.get_location(actor)
    # Get the waypoint at the current location to see if the actor is offroad
    drive_waypoint = CarlaDataProvider.get_map().get_waypoint(current_location, project_to_road=False)
    park_waypoint = CarlaDataProvider.get_map().get_waypoint(current_location, project_to_road=False, lane_type=carla.LaneType.Parking)
    if drive_waypoint or park_waypoint:
        off_road = False
    else:
        off_road = True
    return off_road


def get_ego_min_dis(ego, search_radius, bbox=True):
    ego_min_dis = search_radius
    ego_nearby_agents = CarlaDataProvider.get_ego_nearby_agents(ego.id)
    if ego_nearby_agents:
        for i, vehicle in enumerate(ego_nearby_agents):
            if i < 3:  # calculate only the closest three vehicles
                dis = get_min_distance_across_bboxes(ego, vehicle) if bbox else get_distance_across_centers(ego, vehicle)
                if dis < ego_min_dis:
                    ego_min_dis = dis
    return ego_min_dis


def filter_spawn_points(
    location_lists, 
    radius_list, 
    coord_interval: int = 30, 
    intensity=0.5, 
    low_limit=10, 
    upper_limit=30
):
    """Filter spawn points based on distance constraints and random sampling."""
    map_api: CarlaMap = CarlaDataProvider.get_map_api()
    cur_map = CarlaDataProvider.get_map()
    
    # Convert all locations to right-hand coordinate system
    locations_np = np.array([[loc.x, -loc.y] for loc in location_lists])
    start_loc_np = locations_np[0]  # Initial location
    radii = np.array(radius_list)

    # Collect all coordinates from proximal lanes
    coords_list = []
    for loc, r in zip(location_lists, radius_list):
        query_point = Point(loc.x, -loc.y)
        lane_objs = map_api.query_proximal_lane_data(query_point, r)
        coords_list.extend([lane_obj.center_coords[::coord_interval, :] for lane_obj in lane_objs])
    all_coords = np.concatenate(coords_list, axis=0)  # [N, 2]

    # Vectorized filtering: distance >=8m from start AND within any query radius
    dist_to_start = np.linalg.norm(all_coords - start_loc_np, axis=1)
    mask_start = dist_to_start >= 10.0

    dist_to_queries = np.linalg.norm(
        all_coords[:, np.newaxis, :] - locations_np, 
        axis=2
    )  # [N, M]
    mask_radius = np.any(dist_to_queries <= radii, axis=1)

    filtered_coords = all_coords[mask_start & mask_radius]

    # Randomly sample spawn points
    rng = np.random.default_rng(CarlaDataProvider.get_traffic_random_seed())
    sample_order = rng.permutation(len(filtered_coords))[:int(len(filtered_coords) * intensity)]

    # Convert back to Carla left-hand system and create spawn points
    spawn_points = []
    for idx in sample_order:
        loc = filtered_coords[idx]
        waypoint = cur_map.get_waypoint(
            carla.Location(loc[0], -loc[1]), 
            project_to_road=True, 
            lane_type=carla.LaneType.Driving
        )
        if waypoint:
            spawn_points.append(carla.Transform(
                waypoint.transform.location + carla.Location(z=0.2),
                waypoint.transform.rotation
            ))

    # Adjust final count based on intensity and limits
    final_count = max(
        low_limit, 
        min(len(spawn_points), upper_limit)
    )
    return spawn_points[:final_count]


def get_nearby_agents(center_vehicle, radius=25):
    '''
        return the nearby agents around the center vehicle
    '''
    center_location = CarlaDataProvider.get_location(center_vehicle)

    # get all the vehicles on the world using the actor dict on the CarlaDataProvider
    all_agents = CarlaDataProvider.get_actors()

    # store the nearby vehicle information [vehicle, distance]
    nearby_agents_info = []

    for vehicle_id, vehicle in all_agents.items():
        if vehicle_id != center_vehicle.id:  # except the center vehicle
            # the location of other vehicles
            vehicle_location = CarlaDataProvider.get_location(vehicle)
            distance = center_location.distance(vehicle_location)
            if distance <= radius:
                nearby_agents_info.append((vehicle, distance))

    # sort the nearby vehicles according to the distance in ascending order
    nearby_agents_info.sort(key=lambda x: x[1])

    # return the nearby vehicles list
    nearby_agents = [vehicle for vehicle, dis in nearby_agents_info]

    return nearby_agents


@numba.njit
def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def normalize_relative_angle(lhs, rhs):
    return normalize_angle(lhs - rhs)


def get_forward_speed(transform, velocity):
    """
        Convert the vehicle transform directly to forward speed
    """
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed


@numba.njit
def get_relative_transform(ego_matrix: np.ndarray, vehicle_matrix: np.ndarray):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    relative_pos = np.ascontiguousarray(ego_matrix[:3, :3].T) @ (vehicle_matrix[:3, 3] - ego_matrix[:3, 3])

    # transform to the right-handed system
    relative_pos[1] = - relative_pos[1]

    return relative_pos


def get_relative_info(actor, center_yaw, center_matrix):
    """
        get the relative actor info from the view of center vehicle
        info [x, y, bbox_x, bbox_y, yaw, forward speed]
    """
    actor_transform = CarlaDataProvider.get_transform(actor)
    actor_rotation = actor_transform.rotation
    actor_matrix = np.array(actor_transform.get_matrix())
    # actor bbox
    actor_extent = actor.bounding_box.extent
    dx = np.array([actor_extent.x, actor_extent.y]) * 2.
    # relative yaw angle
    yaw = actor_rotation.yaw / 180 * np.pi
    relative_yaw = normalize_angle(yaw - center_yaw)
    # relative pos
    relative_pos = get_relative_transform(ego_matrix=center_matrix, vehicle_matrix=actor_matrix)
    actor_velocity = CarlaDataProvider.get_velocity(actor)
    actor_speed = get_forward_speed(transform=actor_transform, velocity=actor_velocity)  # In m/s
    actor_info = [relative_pos[0], relative_pos[1], dx[0], dx[1], relative_yaw, actor_speed]
    return actor_info


def get_actor_polygons(filt):
    actor_poly_dict = {}
    for actor in CarlaDataProvider._world.get_actors().filter(filt):
        # Get x, y and yaw of the actor
        trans = actor.get_transform()
        x = trans.location.x
        y = trans.location.y
        yaw = trans.rotation.yaw / 180 * np.pi
        # Get length and width
        bb = actor.bounding_box
        l = bb.extent.x
        w = bb.extent.y
        # Get bounding box polygon in the actor's local coordinate
        poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
        # Get rotation matrix to transform to global coordinate
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        # Get global bounding box polygon
        poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
        actor_poly_dict[actor.id] = poly
    return actor_poly_dict


def get_min_distance_across_bboxes(veh1, veh2):
    box2origin_veh1, size_veh1 = compute_box2origin(veh1.bounding_box, CarlaDataProvider.get_transform(veh1))
    box2origin_veh2, size_veh2 = compute_box2origin(veh2.bounding_box, CarlaDataProvider.get_transform(veh2))
    # min distance
    box_collider_veh1 = colliders.Box(box2origin_veh1, size_veh1)
    box_collider_veh2 = colliders.Box(box2origin_veh2, size_veh2)
    dist, closest_point_box, closest_point_box2, _ = gjk.gjk(
        box_collider_veh1, box_collider_veh2)
    return dist


def get_distance_across_centers(veh1, veh2):
    veh1_loc = CarlaDataProvider.get_location(veh1)
    veh2_loc = CarlaDataProvider.get_location(veh2)
    return veh1_loc.distance(veh2_loc)


@numba.njit
def calculate_abs_velocity_array(vel_array):
    return np.sqrt(vel_array[0] ** 2 + vel_array[1] ** 2)


def calculate_abs_velocity(velocity):
    vel_array = np.array([velocity.x, velocity.y])
    return calculate_abs_velocity_array(vel_array)


@numba.njit
def calculate_abs_acc_array(acc_array):
    return np.sqrt(acc_array[0] ** 2 + acc_array[1] ** 2)


def calculate_abs_acc(acc):
    acc_array = np.array([acc.x, acc.y])
    return calculate_abs_acc_array(acc_array)


def compute_box2origin_numba(bbox_loc_array, vehicle_loc_array, pitch_rad, yaw_rad, roll_rad, extent_array):
    t = bbox_loc_array + vehicle_loc_array
    r = compute_R_numba(pitch_rad, yaw_rad, roll_rad)
    size = extent_array * np.float64(2.0)

    box2origin = np.zeros((4, 4), dtype=np.float64)
    box2origin[:3, :3] = r
    box2origin[:3, 3] = t
    box2origin[3, 3] = np.float64(1.0)

    return box2origin, size


def compute_box2origin(vehicle_box, vehicle_transform):
    vehicle_location = vehicle_transform.location
    bbox_location = vehicle_box.location
    vehicle_loc_array = np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z], dtype=np.float64)
    bbox_loc_array = np.array([bbox_location.x, bbox_location.y, bbox_location.z], dtype=np.float64)

    rotation = vehicle_transform.rotation
    pitch_rad = np.float64(np.radians(rotation.pitch))
    yaw_rad = np.float64(np.radians(rotation.yaw))
    roll_rad = np.float64(np.radians(rotation.roll))

    extent = vehicle_box.extent
    extent_array = np.array([extent.x, extent.y, extent.z], dtype=np.float64)

    return compute_box2origin_numba(bbox_loc_array, vehicle_loc_array, pitch_rad, yaw_rad, roll_rad, extent_array)


@numba.njit
def compute_R_numba(pitch_rad, yaw_rad, roll_rad):
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
                   [0.0, np.sin(roll_rad), np.cos(roll_rad)]], dtype=np.float64)
    Ry = np.array([[np.cos(pitch_rad), 0.0, np.sin(pitch_rad)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)]], dtype=np.float64)
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
                   [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

    rotation_matrix = Rx @ Ry @ Rz
    return rotation_matrix