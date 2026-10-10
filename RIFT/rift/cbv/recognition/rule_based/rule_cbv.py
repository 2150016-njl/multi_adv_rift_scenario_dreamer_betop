#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : rule_cbv.py
@Date    : 2024/7/7
"""
from rift.cbv.recognition.base_cbv import BaseCBVRecog
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class RuleCBVRecog(BaseCBVRecog):
    name = 'rule'

    """ This is the template for implementing the CBV candidate selection for a scenario. """

    def __init__(self, config, logger):
        super().__init__(config, logger)

    def get_CBVs(self, ego_vehicle, CBVs_id, local_route_waypoints, rest_route_waypoints, red_light_state=None):
        CBV_candidates = self.get_CBV_candidates(ego_vehicle, CBVs_id, rest_route_waypoints)
        CBVs = self.find_closest_vehicle(CBV_candidates, CBVs_id)
        return CBVs

    def find_closest_vehicle(self, CBV_candidates, CBVs_id):
        '''
            rule-based method to find the CBV:
            find the closest N vehicle among all the CBV candidates
        '''
        sample_length = self.max_agent_num - len(CBVs_id)

        return CBV_candidates[:sample_length]


