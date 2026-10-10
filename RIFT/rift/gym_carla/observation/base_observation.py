#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : base_observation.py
@Date    : 2024/09/24
"""
import copy
import carla
import numpy as np
from rdp import rdp

from rift.cbv.planning.route_planner.base_planner import CBVBasePlanner
from rift.cbv.planning.route_planner.goal_planner import CBVGoalPlanner
from rift.ego.route_planner.route_planner import EgoRoutePlanner
from rift.gym_carla.reward.cbv_reward import CBVBaseReward, CBVFullTrainReward
from rift.gym_carla.reward.ego_reward import EgoReward
from rift.gym_carla.utils.common import calculate_abs_velocity, get_relative_info, get_relative_transform, normalize_angle
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class EgoBaseObservation(object):
    type = 'ego_no_obs'
    """
        Base Action
    """

    def __init__(self, env_params, EgoAction):
        self.EgoAction = EgoAction
        self.env_params = env_params
        self.ego_route_planner = EgoRoutePlanner(env_params)
        self.ego_reward = EgoReward(env_params['desired_speed'], env_params['out_lane_thres'])

    def get_obs(self, *args, **kwargs):
        return None
    
    def get_tran_obs(self, *args, **kwargs):
        return None


class CBVBaseObservation(object):
    type = 'cbv_no_obs'
    """
        Base Action
    """

    def __init__(self, env_params, CBVAction):
        self.CBVAction = CBVAction
        self.env_params = env_params
        self.CBV_route_planner = CBVBasePlanner(env_params)
        self.CBV_reward = CBVBaseReward(self.CBV_route_planner, env_params['desired_speed'], env_params['out_lane_thres'])

    def get_obs(self, *args, **kwargs):
        return None
    
    def get_tran_obs(self, *args, **kwargs):
        return None


class EgoSimpleObservation(EgoBaseObservation):
    type = 'ego_simple_obs'

    def __init__(self, env_params, EgoAction):
        super().__init__(env_params, EgoAction)

    def get_obs(self, ego_vehicle):
        # Ego state
        ego_trans = CarlaDataProvider.get_transform(ego_vehicle)
        ego_loc = ego_trans.location
        ego_pos = np.array([ego_loc.x, ego_loc.y])
        ego_speed = calculate_abs_velocity(CarlaDataProvider.get_velocity(ego_vehicle))  # m/s
        ego_compass = np.deg2rad(ego_trans.rotation.yaw)  # use API yaw instead of IMU, (need to minus 90 if using IMU compass)
        return {
            'gps': ego_pos,
            'speed': ego_speed,
            'compass': ego_compass
        }
    
    def get_tran_obs(self, ego_vehicle, next_obs):
        # ego vehicle share same next observation and transition observation
        return copy.deepcopy(next_obs)


class EgoNormalObservation(EgoBaseObservation):
    type = 'ego_normal_obs'

    def __init__(self, env_params, EgoAction):
        super().__init__(env_params, EgoAction)
        self.max_agent = 3

    def get_obs(self, ego_vehicle, ego_nearby_agents=None):
        '''
        safety network input state:
        all the rows are other bv's relative state
        '''
        infos = []
        # the basic information about the ego (center vehicle)
        ego_transform = CarlaDataProvider.get_transform(ego_vehicle)
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_yaw = ego_transform.rotation.yaw / 180 * np.pi
        ego_extent = ego_vehicle.bounding_box.extent
        # the relative CBV info
        ego_info = get_relative_info(actor=ego_vehicle, center_yaw=ego_yaw, center_matrix=ego_matrix)
        infos.append(ego_info)
        nearby_agents = ego_nearby_agents if ego_nearby_agents else CarlaDataProvider.get_ego_nearby_agents(ego_vehicle.id)
        for actor in nearby_agents:
            if len(infos) < self.max_agent:
                actor_info = get_relative_info(actor=actor, center_yaw=ego_yaw, center_matrix=ego_matrix)
                infos.append(actor_info)
            else:
                break
        while len(infos) < self.max_agent:  # if no enough nearby vehicles, padding with 0
            infos.append([0] * len(ego_info))

        # route information
        if self.ego_route_planner.local_route_waypoints is not None:
            route_info = self.get_relative_route_info(self.ego_route_planner.local_route_waypoints, center_yaw=ego_yaw, center_matrix=ego_matrix, center_extent=ego_extent)
            infos.append(route_info)

        # get the info of the ego vehicle and the other actors
        ego_obs = np.array(infos, dtype=np.float32)

        return ego_obs
    

    def get_relative_route_info(self, local_route_waypoints, center_yaw, center_matrix, center_extent):
        """
            get the relative route info from the view of center vehicle
            info [x, y, bbox_x, bbox_y, yaw, distance]
        """
        waypoint_route = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in local_route_waypoints])
        max_len = 12
        if len(waypoint_route) < max_len:
            max_len = len(waypoint_route)
        shortened_route = rdp(waypoint_route[:max_len], epsilon=0.5)

        # convert points to vectors
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors / 2.
        norms = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        i = 0  # only use the first midpoint
        midpoint = midpoints[i]
        # find distance to center of waypoint
        center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
        transform = carla.Transform(center_bounding_box)
        route_matrix = np.array(transform.get_matrix())
        relative_pos = get_relative_transform(center_matrix, route_matrix)
        distance = np.linalg.norm(relative_pos)

        length_bounding_box = carla.Vector3D(norms[i] / 2., center_extent.y, center_extent.z)
        bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
        bounding_box.rotation = carla.Rotation(pitch=0.0,
                                            yaw=angles[i] * 180 / np.pi,
                                            roll=0.0)

        route_extent = bounding_box.extent
        dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
        relative_yaw = normalize_angle(angles[i] - center_yaw)

        route_info = [relative_pos[0], relative_pos[1], dx[0], dx[1], relative_yaw, distance]

        return route_info
    
    def get_tran_obs(self, ego_vehicle, next_obs):
        # ego vehicle share same next observation and transition observation
        return copy.deepcopy(next_obs)


class CBVNormalObservation(CBVBaseObservation):
    type = 'cbv_normal_obs'

    def __init__(self, env_params, CBVAction):
        super().__init__(env_params, CBVAction)
        # normal observation uses the goal-based planner
        self.CBV_route_planner = CBVGoalPlanner(env_params)
        # init the reward function for CBV
        self.CBV_reward = CBVFullTrainReward(self.CBV_route_planner, env_params['desired_speed'], env_params['out_lane_thres'])

        self.max_agent = 3
        self.goal_radius = self.CBV_route_planner.goal_waypoints_radius

    def get_obs(self, ego_vehicle):
        """
            cbv normal obs:
            first row is CBV's relative state [x, y, bbox_x, bbox_y, yaw, forward speed]
            second row is ego's relative state [x, y, bbox_x, bbox_y, yaw, forward speed]
            rest row are other BV's relative state [x, y, bbox_x, bbox_y, yaw, forward speed]
        """
        CBVs_obs = {}
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(ego_vehicle.id).items():
            CBVs_obs[CBV_id] = self._get_obs(CBV_id, CBV, ego_vehicle)
        return CBVs_obs

    def _get_obs(self, CBV_id, CBV, ego_vehicle):
        actors_info = []
        # the basic information about the CBV (center vehicle)
        CBV_transform = CarlaDataProvider.get_transform(CBV)
        CBV_matrix = np.array(CBV_transform.get_matrix())
        CBV_yaw = CBV_transform.rotation.yaw / 180 * np.pi
        # the relative CBV info
        CBV_info = get_relative_info(actor=CBV, center_yaw=CBV_yaw, center_matrix=CBV_matrix)
        actors_info.append(CBV_info)
        # the relative ego info
        ego_info = get_relative_info(actor=ego_vehicle, center_yaw=CBV_yaw, center_matrix=CBV_matrix)
        actors_info.append(ego_info)

        for actor in CarlaDataProvider.get_CBV_nearby_agents(ego_vehicle.id, CBV_id):
            if actor.id == ego_vehicle.id:
                continue  # except the ego actor
            elif len(actors_info) < self.max_agent:
                actor_info = get_relative_info(actor=actor, center_yaw=CBV_yaw, center_matrix=CBV_matrix)
                actors_info.append(actor_info)
            else:
                # avoiding too many nearby vehicles
                break
        while len(actors_info) < self.max_agent:  # if no enough nearby vehicles, padding with 0
            actors_info.append([0] * len(CBV_info))

        # goal information
        goal_waypoint = self.CBV_route_planner.get_goal_waypoint()
        goal_info = self.get_relative_waypoint_info(goal_waypoint, CBV_yaw, CBV_matrix)
        actors_info.append(goal_info)

        return np.array(actors_info, dtype=np.float32)

    def get_relative_waypoint_info(self, goal_waypoint, center_yaw, center_matrix):
        """
        get the relative waypoint info from the view of center vehicle
        
        return:
        - info: [x, y, radius, radius, yaw, distance]
        """
        goal_transform = goal_waypoint.transform
        goal_matrix = np.array(goal_transform.get_matrix())
        # relative yaw angle
        yaw = goal_transform.rotation.yaw / 180 * np.pi
        relative_yaw = normalize_angle(yaw - center_yaw)
        # relative pos
        relative_pos = get_relative_transform(ego_matrix=center_matrix, vehicle_matrix=goal_matrix)
        distance = np.linalg.norm(relative_pos)

        goal_info = [relative_pos[0], relative_pos[1], self.goal_radius, self.goal_radius, relative_yaw, distance]

        return goal_info
    
    def get_tran_obs(self, ego_vehicle, next_obs):
        tran_obs = {}
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(ego_vehicle.id).items():
            if CBV_id in next_obs:
                tran_obs[CBV_id] = next_obs[CBV_id]
            else:
                tran_obs[CBV_id] = self._get_obs(CBV_id, CBV, ego_vehicle)
        return tran_obs



