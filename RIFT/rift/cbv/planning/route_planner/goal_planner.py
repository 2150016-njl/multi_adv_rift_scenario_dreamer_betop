#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : goal_planner.py
@Date    : 2024/10/12
'''
import carla
from typing import List
from rift.cbv.planning.route_planner.base_planner import CBVBasePlanner
from rift.gym_carla.visualization.visualize import draw_waypoint
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class CBVGoalPlanner(CBVBasePlanner):
    def __init__(self, env_params, goal_waypoints_radius=4.0):
        super(CBVGoalPlanner, self).__init__(env_params)
        self.goal_waypoints_radius = goal_waypoints_radius
        self.frame_rate = env_params['frame_rate']
        self._world = CarlaDataProvider.get_world()

    def reset(self, ego_vehicle: carla.Vehicle, ego_global_route_waypoints: List[carla.Waypoint]):
        """
        Reset the CBV goals
        """
        self.ego_vehicle = ego_vehicle
        self.ego_global_route_waypoints = ego_global_route_waypoints
        self._goal_point = None
        self._previous_goal_point = None
        self._goal_CBV_dis = None
        self._previous_goal_CBV_dis = None
    
    def run_step(self, ego_local_route_waypoints: List[carla.Waypoint], ego_rest_route_waypoints: List[carla.Waypoint]):
        """
        Update the CBV goals
        """
        self._previous_goal_point = self._goal_point
        self._previous_goal_CBV_dis = self._goal_CBV_dis

        # the goal waypoint is related to the ego local route waypoints
        self._goal_point = ego_local_route_waypoints[len(ego_local_route_waypoints) // 3]

        # check whether the CBV reach the goal
        self._check_CBVs_reach_goal()

    def _check_CBVs_reach_goal(self):
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).items():
            CBV_loc = CarlaDataProvider.get_location(CBV)
            dis = CBV_loc.distance(self._goal_point.transform.location)
            if dis < self.goal_waypoints_radius:
                CarlaDataProvider.CBV_reach_goal(self.ego_vehicle, CBV)

    def update_CBV(self, ego_rest_route_waypoints: List[carla.Waypoint]=None):
        """
        After changing the CBV, update the CBV goals
        """
        self._goal_CBV_dis = {}
        # update the goal CBV distance dict
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).items():
            CBV_loc = CarlaDataProvider.get_location(CBV)
            self._goal_CBV_dis[CBV_id] = CBV_loc.distance(self._goal_point.transform.location)
    
    def get_goal_waypoint(self):
        """
        Get the goal waypoint
        """
        return self._goal_point
    
    def get_previous_goal_waypoint(self):
        """
        Get the previous goal waypoint
        """
        return self._previous_goal_point
    
    def get_previous_goal_CBV_dis(self, CBV_id):
        """
        Get the previous goal CBV distance
        """
        return self._previous_goal_CBV_dis[CBV_id]
    
    def vis_route(self):
        # draw the goal waypoint
        draw_waypoint(self._world, self._goal_point, frame_rate=self.frame_rate)
    
