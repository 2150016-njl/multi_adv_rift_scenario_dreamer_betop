#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : base_planner.py
@Date    : 2024/10/12
'''
import carla
from typing import List


class CBVBasePlanner():
    def __init__(self, env_params):
        self.inter_info = {}

    def reset(self, ego_vehicle: carla.Vehicle, ego_global_route_waypoints: List[carla.Waypoint]):
        """
        Reset
        """
        self.ego_vehicle = ego_vehicle
        self.ego_global_route_waypoints = ego_global_route_waypoints
    
    def run_step(self, ego_local_route_waypoints: List[carla.Waypoint], ego_rest_route_waypoints: List[carla.Waypoint]):
        """
        Update
        """
        pass

    def update_inter_info(
            self,
            vehicle_id,
            inter_waypoint,
            inter_CBV_route_elements,
            inter_CBV_route_ids,
            inter_CBV_dis,
            interaction_meta=None):
        self.inter_info[vehicle_id] = {
            'inter_waypoint': inter_waypoint,
            'inter_CBV_route_elements': inter_CBV_route_elements,
            'inter_CBV_route_ids': inter_CBV_route_ids,
            'inter_CBV_dis': inter_CBV_dis,
            'interaction_meta': interaction_meta or {}
        }
    
    def update_CBV(self, ego_rest_route_waypoints: List[carla.Waypoint]=None):
        pass

    def vis_route(self):
        pass
