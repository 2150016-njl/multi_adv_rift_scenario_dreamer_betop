#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : data_collect.py
@Date    : 2024/7/10
"""
import math

import numba
import numpy as np
import torch
import carla

from einops import rearrange
from rdp import rdp

from rift.gym_carla.utils.common import get_relative_info, get_min_distance_across_bboxes, get_distance_across_centers, normalize_angle
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.util.torch_util import CPU, CUDA

VIRTUAL_LIDAR_TRANS = np.array([1.3, 0.0, 2.5], dtype=np.float64)


def get_vehicle_infos(ego_vehicle, vehicle_ids):
    vehicle_infos = []
    assert ego_vehicle.id == vehicle_ids[0], "ego vehicle id should be the same"

    # the basic information about the ego (center vehicle)
    ego_transform = CarlaDataProvider.get_transform(ego_vehicle)
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_yaw = ego_transform.rotation.yaw / 180 * np.pi
    # the relative ego info
    ego_info = get_relative_info(actor=ego_vehicle, center_yaw=ego_yaw, center_matrix=ego_matrix)
    vehicle_infos.append(ego_info)
    for vehicle_id in vehicle_ids[1:]:
        vehicle = CarlaDataProvider.get_actor_by_id(vehicle_id)
        # the relative background vehicle info
        vehicle_infos.append(get_relative_info(actor=vehicle, center_yaw=ego_yaw, center_matrix=ego_matrix))

    return np.array(vehicle_infos, dtype=np.float32)


def get_bev_boxes(ego_veh, nearby_agents, local_route_waypoints):
    """
        modify from the PlanT
    """
    # -----------------------------------------------------------
    # Ego vehicle
    # -----------------------------------------------------------

    # add vehicle velocity and brake flag
    ego_transform = CarlaDataProvider.get_transform(ego_veh)
    ego_control = ego_veh.get_control()
    ego_velocity = ego_veh.get_velocity()
    ego_speed = get_forward_speed(transform=ego_transform, velocity=ego_velocity)  # In m/s
    ego_brake = ego_control.brake
    ego_rotation = ego_transform.rotation
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_extent = ego_veh.bounding_box.extent
    ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
    ego_yaw = ego_rotation.yaw / 180 * np.pi
    relative_yaw = 0
    relative_pos = get_relative_transform_for_lidar(ego_matrix, ego_matrix)

    results = []

    # add ego-vehicle to results list
    # the format is category, extent*3, position*3, yaw, points_in_bbox, distance, id
    # the position is in lidar coordinates
    result = {"class": "Car",
              "extent": [ego_dx[2], ego_dx[0], ego_dx[1]],  
              "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
              "yaw": relative_yaw,
              "num_points": -1,
              "distance": -1,
              "speed": ego_speed,
              "brake": ego_brake,
              "id": int(ego_veh.id),
              }
    results.append(result)

    # -----------------------------------------------------------
    # Other vehicles
    # -----------------------------------------------------------

    for vehicle in nearby_agents:
        vehicle_transform = CarlaDataProvider.get_transform(vehicle)
        vehicle_rotation = vehicle_transform.rotation

        vehicle_matrix = np.array(vehicle_transform.get_matrix())

        vehicle_extent = vehicle.bounding_box.extent
        dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
        yaw = vehicle_rotation.yaw / 180 * np.pi

        relative_yaw = normalize_angle(yaw - ego_yaw)
        relative_pos = get_relative_transform_for_lidar(ego_matrix, vehicle_matrix)

        vehicle_control = vehicle.get_control()
        vehicle_velocity = vehicle.get_velocity()
        vehicle_speed = get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)  # In m/s
        vehicle_brake = vehicle_control.brake

        # # filter bbox that didn't contain points of contains fewer points
        # if not lidar is None:
        #     num_in_bbox_points = get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
        #     # print("num points in bbox", num_in_bbox_points)
        # else:
        #     num_in_bbox_points = -1
        num_in_bbox_points = -1

        distance = np.linalg.norm(relative_pos)

        result = {
            "class": "Car",
            "extent": [dx[2], dx[0], dx[1]],
            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
            "yaw": relative_yaw,
            "num_points": int(num_in_bbox_points),
            "distance": distance,
            "speed": vehicle_speed,
            "brake": vehicle_brake,
            "id": int(vehicle.id),
        }
        results.append(result)

    # -----------------------------------------------------------
    # Route rdp
    # -----------------------------------------------------------
    max_len = 50
    waypoint_route = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in local_route_waypoints])
    if len(waypoint_route) < max_len:
        max_len = len(waypoint_route)
    shortened_route = rdp(waypoint_route[:max_len], epsilon=0.5)

    # convert points to vectors
    vectors = shortened_route[1:] - shortened_route[:-1]
    midpoints = shortened_route[:-1] + vectors / 2.
    norms = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    for i, midpoint in enumerate(midpoints):
        # find distance to center of waypoint
        center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
        transform = carla.Transform(center_bounding_box)
        route_matrix = np.array(transform.get_matrix())
        relative_pos = get_relative_transform_for_lidar(ego_matrix, route_matrix)
        distance = np.linalg.norm(relative_pos)

        # find distance to the beginning of bounding box
        starting_bounding_box = carla.Location(shortened_route[i][0], shortened_route[i][1], 0.0)
        st_transform = carla.Transform(starting_bounding_box)
        st_route_matrix = np.array(st_transform.get_matrix())
        st_relative_pos = get_relative_transform_for_lidar(ego_matrix, st_route_matrix)
        st_distance = np.linalg.norm(st_relative_pos)

        # only store route boxes that are near the ego vehicle
        if i > 0 and st_distance > 30:
            continue

        length_bounding_box = carla.Vector3D(norms[i] / 2., ego_extent.y, ego_extent.z)
        bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
        bounding_box.rotation = carla.Rotation(pitch=0.0,
                                               yaw=angles[i] * 180 / np.pi,
                                               roll=0.0)

        route_extent = bounding_box.extent
        dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
        relative_yaw = normalize_angle(angles[i] - ego_yaw)

        result = {
            "class": "Route",
            "extent": [dx[2], dx[0], dx[1]],
            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
            "yaw": relative_yaw,
            "centre_distance": distance,
            "starting_distance": st_distance,
            "id": i,
        }
        results.append(result)

    return results


def get_forward_speed(transform, velocity):
    """ Convert the vehicle transform directly to forward speed """

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed


@numba.njit
def get_relative_transform_for_lidar(ego_matrix: np.ndarray, vehicle_matrix: np.ndarray):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    relative_pos = np.ascontiguousarray(ego_matrix[:3, :3].T) @ (vehicle_matrix[:3, 3] - ego_matrix[:3, 3])

    # transform to right-handed system
    relative_pos[1] = - relative_pos[1]

    # transform relative pos to virtual lidar system
    relative_pos -= VIRTUAL_LIDAR_TRANS

    return relative_pos


def get_input_batch(label_raw, target_point, traffic_light_hazard, max_NextRouteBBs=2):
    sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}

    # all the vehicles' id
    data_car_ids = [x['id'] for x in label_raw if x['class'] == 'Car']

    data = label_raw[1:]  # remove first element (ego vehicle)

    data_car = [[
        1.,  # type indicator for cars
        float(x['position'][0]) - float(label_raw[0]['position'][0]),
        float(x['position'][1]) - float(label_raw[0]['position'][1]),
        float(x['yaw'] * 180 / 3.14159265359),  # in degrees
        float(x['speed'] * 3.6),  # in km/h
        float(x['extent'][2]),
        float(x['extent'][1]),
    ] for x in data if x['class'] == 'Car']
    # if we use the far_node as target waypoint we need the route as input
    data_route = [
        [
            2.,  # type indicator for route
            float(x['position'][0]) - float(label_raw[0]['position'][0]),
            float(x['position'][1]) - float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359),  # in degrees
            float(x['id']),
            float(x['extent'][2]),
            float(x['extent'][1]),
        ]
        for j, x in enumerate(data)
        if x['class'] == 'Route'
           and float(x['id']) < max_NextRouteBBs]

    # we split route segment longer than 10m into multiple segments
    # improves generalization
    data_route_split = []
    for route in data_route:
        if route[6] > 10:
            routes = split_large_BB(route, len(data_route_split))
            data_route_split.extend(routes)
        else:
            data_route_split.append(route)

    data_route = data_route_split[:max_NextRouteBBs]

    assert len(data_route) <= max_NextRouteBBs, 'Too many routes'

    features = data_car + data_route

    sample['input'] = features

    # dummy data
    sample['output'] = features
    sample['light'] = traffic_light_hazard

    local_command_point = np.array([target_point.transform.location.x, target_point.transform.location.x])
    sample['target_point'] = local_command_point

    batch = [sample]

    input_batch = generate_batch(batch)

    return input_batch, data_car_ids, data_car


def generate_batch(data_batch):
    input_batch, output_batch = [], []
    for element_id, sample in enumerate(data_batch):
        input_item = torch.tensor(sample["input"], dtype=torch.float32)
        output_item = torch.tensor(sample["output"], dtype=torch.float32)

        input_indices = torch.tensor([element_id] * len(input_item)).unsqueeze(1)
        output_indices = torch.tensor([element_id] * len(output_item)).unsqueeze(1)

        input_batch.append(CUDA(torch.cat([input_indices, input_item], dim=1)))
        output_batch.append(CUDA(torch.cat([output_indices, output_item], dim=1)))

    waypoints_batch = CUDA(torch.tensor([sample["waypoints"] for sample in data_batch]))
    tp_batch = CUDA(torch.tensor(
        [sample["target_point"] for sample in data_batch], dtype=torch.float32
    ))
    light_batch = CUDA(rearrange(
        torch.tensor([sample["light"] for sample in data_batch]), "b -> b 1"
    ))

    return input_batch, output_batch, waypoints_batch, tp_batch, light_batch


def split_large_BB(route, start_id):
    x = route[1]
    y = route[2]
    angle = route[3] - 90
    extent_x = route[5] / 2
    extent_y = route[6] / 2

    x1 = x - extent_y * math.sin(math.radians(angle))
    y1 = y - extent_y * math.cos(math.radians(angle))

    x0 = x + extent_y * math.sin(math.radians(angle))
    y0 = y + extent_y * math.cos(math.radians(angle))

    number_of_points = (
        math.ceil(extent_y * 2 / 10) - 1
    )  # 5 is the minimum distance between two points, we want to have math.ceil(extent_y / 5) and that minus 1 points
    xs = np.linspace(
        x0, x1, number_of_points + 2
    )  # +2 because we want to have the first and last point
    ys = np.linspace(y0, y1, number_of_points + 2)

    splitted_routes = []
    for i in range(len(xs) - 1):
        route_new = route.copy()
        route_new[1] = (xs[i] + xs[i + 1]) / 2
        route_new[2] = (ys[i] + ys[i + 1]) / 2
        route_new[4] = float(start_id + i)
        route_new[5] = extent_x * 2
        route_new[6] = route[6] / (
            number_of_points + 1
        )
        splitted_routes.append(route_new)

    return splitted_routes