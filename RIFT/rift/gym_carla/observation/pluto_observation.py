#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : pluto_observation.py
@Date    : 2024/09/25
"""
import copy
import numpy as np
import yaml
from rift.cbv.planning.pluto.feature_builder.pluto_feature_builder import PlutoFeatureBuilder
from rift.gym_carla.observation.base_observation import CBVBaseObservation, EgoBaseObservation

from rift.gym_carla.reward.cbv_reward import CBVFineTuneReward
from rift.gym_carla.reward.ego_reward import EgoRouteReward
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.cbv.planning.route_planner.route_planner import CBVRoutePlanner
from rift.util.torch_util import get_device_name
from nuplan_plugin.actor_state.state_representation import StateSE2


class CBVPlutoObservation(CBVBaseObservation):
    type = 'cbv_pluto_obs'

    def __init__(self, env_params, CBVAction, config_path='rift/cbv/planning/config/pluto.yaml'):
        super().__init__(env_params, CBVAction)
        # load the config for pluto model
        self.config = self.load_config(config_path)
        # init the route planner for CBV
        self.CBV_route_planner = CBVRoutePlanner(env_params)
        # init the reward function for CBV
        self.CBV_reward = CBVFineTuneReward(self.CBV_route_planner, self.config)
        self.mode = env_params['mode']
        self._render = env_params['need_video_render']
        self.device = get_device_name()

        self.feature_builder = PlutoFeatureBuilder(self.config, self.CBV_route_planner)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_obs(self, ego_vehicle):
        """
            cbv full obs:
        """
        CBVs_obs = {}
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(ego_vehicle.id).items():
            self._get_obs(CBVs_obs, CBV_id, CBV, ego_vehicle)

        return CBVs_obs

    def _get_obs(self, CBVs_obs, CBV_id, CBV, ego_vehicle):
        # feature data for pluto model
        CBV_nearby_agents = CarlaDataProvider.get_CBV_nearby_agents(ego_vehicle.id, CBV_id)
        pluto_feature, route_ids, reference_lines, route_elements, inter_point = self.feature_builder.build_feature(CBV, CBV_nearby_agents, self.mode)
        
        CBVs_obs[CBV_id] = {
            'raw_pluto_feature': pluto_feature.to_feature_tensor()
        }

        # update the render data
        if self._render:
            # from left-hand to right-hand
            route_waypoints = []
            for wp, road_option in route_elements:
                wp_tran = wp.transform
                route_waypoints.append(StateSE2(x=wp_tran.location.x, y=-wp_tran.location.y, heading=-np.deg2rad(wp_tran.rotation.yaw)))
            inter_wp = StateSE2(x=inter_point.transform.location.x, y=-inter_point.transform.location.y, heading=-np.deg2rad(inter_point.transform.rotation.yaw))

            CBVs_obs[CBV_id].update({
                'route_ids': route_ids,
                'reference_lines': reference_lines,
                'route_waypoints': route_waypoints,
                'interaction_wp': inter_wp                 
            })

    def get_tran_obs(self, ego_vehicle, next_obs):
        tran_obs = {}
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(ego_vehicle.id).items():
            if CBV_id in next_obs:
                tran_obs[CBV_id] = next_obs[CBV_id]
            else:
                self._get_obs(tran_obs, CBV_id, CBV, ego_vehicle)
        return tran_obs
    

class EgoPlutoObservation(EgoBaseObservation):
    type = 'ego_pluto_obs'

    def __init__(self, env_params, EgoAction, config_path='rift/cbv/planning/config/pluto.yaml'):
        super().__init__(env_params, EgoAction)
        self.ego_reward = EgoRouteReward(self.ego_route_planner)
        self.mode = env_params['mode']
        self.device = get_device_name()

        self.config = self.load_config(config_path)
        self.feature_builder = PlutoFeatureBuilder(self.config, self.ego_route_planner)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_obs(self, ego_vehicle):
        """
            ego pluto obs:
        """
        pluto_feature, route_ids, reference_lines, route_elements, inter_point  = self.feature_builder.build_feature(ego_vehicle, CarlaDataProvider.get_ego_nearby_agents(ego_vehicle.id), self.mode)
        pluto_feature_torch = pluto_feature.collate(
            [pluto_feature.to_feature_tensor()]
        ).to_device(self.device)

        return {
            'ego_pluto_feature': pluto_feature_torch
        }

    def get_tran_obs(self, ego_vehicle, next_obs):
        # ego vehicle share same next observation and transition observation
        return copy.deepcopy(next_obs)