#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : dummy_policy.py
@Date    : 2023/10/4
"""
from typing import Dict

import numpy as np

from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class DummyPolicy(CBVBasePolicy):
    name = 'dummy'
    type = 'unlearnable'

    """ This agent is used for scenarios that do not have controllable agents. """
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.logger = logger
        self.logger.log('>> This scenario does not require policy model, using a dummy one', color='yellow')
        self.num_scenario = config['num_scenario']
        # render
        self._render = config['need_video_render']
        if self._render:
            self.reset_render_data()

    def reset_render_data(self):
        self._render_data = {env_id: {
            'ego_states': {},
            "nearby_agents_states": {},
        } for env_id in range(self.num_scenario)}

    def get_render_data(self, env_id):
        return self._render_data[env_id]

    def get_action(self, CBVs_obs_list, infos, deterministic=False) -> Dict[str, list]:
        data = {'CBVs_actions': {env_id:None for env_id in range(self.num_scenario)}}

        if self._render:
            for info in infos:
                env_id = info['env_id']
                # ego states
                ego = CarlaDataProvider.get_ego_vehicle_by_env_id(env_id)
                ego_state = CarlaDataProvider.get_current_state(ego)
                self._render_data[env_id]['ego_states'][ego.id] = ego_state
                # nearby agent states
                for agent in CarlaDataProvider.get_ego_nearby_agents(ego.id):
                    agent_state = CarlaDataProvider.get_current_state(agent)
                    self._render_data[env_id]['nearby_agents_states'][agent.id] = agent_state     
        return data
