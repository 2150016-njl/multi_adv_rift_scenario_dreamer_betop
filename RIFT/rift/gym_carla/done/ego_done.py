#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : ego_done.py
@Date    : 2024/10/13
'''


class EgoDone():
    def __init__(self, scenario_manager, logger):
        self.scenario_manager = scenario_manager
        self.logger = logger
        self.ego_vehicle = None

    def set_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def terminated(self):
        """
        get the ego done
        """
        return not self.scenario_manager.running
        