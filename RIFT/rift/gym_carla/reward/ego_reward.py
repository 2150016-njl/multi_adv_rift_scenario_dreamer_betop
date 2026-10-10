#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : ego_reward.py
@Date    : 2024/10/13
'''
import numpy as np

from rift.gym_carla.utils.misc import get_lane_dis, get_pos
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class EgoReward():
    def __init__(self, desired_speed, out_lane_thres):
        self.ego_vehicle = None
        self.desired_speed = desired_speed
        self.out_lane_thres = out_lane_thres

    def set_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def get_reward(self, local_route_waypoints, ego_collide):
        """
        Get the ego step reward
        """
        r_collision = -1 if ego_collide else 0

        # reward for steering:
        r_steer = -self.ego_vehicle.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(local_route_waypoints, ego_x, ego_y)
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        # reward for speed tracking
        v = CarlaDataProvider.get_velocity(self.ego_vehicle)

        # cost for too fast
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)
        r_fast = -1 if lspeed_lon > self.desired_speed else 0

        # cost for lateral acceleration
        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon ** 2

        # combine all rewards
        # r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
        # reward from "Interpretable End-to-End Urban Autonomous Driving With Latent Deep Reinforcement Learning"
        r = 10 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1
        return r
    

class EgoRouteReward():
    def __init__(self, ego_route_planner):
        self.ego_vehicle = None
        self.ego_route_planner = ego_route_planner

    def set_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def get_reward(self, ego_collide):
        return {}