#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : cbv_done.py
@Date    : 2024/10/13
'''


from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class CBVDone():
    def __init__(self, scenario_manager, cbv_recog_policy, logger):
        self.scenario_manager = scenario_manager
        self.cbv_recog_policy = cbv_recog_policy
        self.logger = logger
        self.ego_vehicle = None
        self.ego_length = None
        self.CBVs_terminated = {}
        self.CBVs_truncated = {}
        self.CBVs_done = {}
    
    def set_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        self.ego_length = self.ego_vehicle.bounding_box.extent.x * 2

    def terminated(self, CBVs_collision):
        self.CBVs_terminated = {}
        for CBV_id in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).keys():
            # if CBV collide with the other vehicles, then CBV terminated
            if CBVs_collision[CBV_id] is not None:
                self.CBVs_terminated[CBV_id] = True
                self.logger.log(f">> CBV:{CBV_id} collide")
            # if CBV reach the goal, then CBV terminated
            elif CBV_id in CarlaDataProvider.get_CBVs_reach_goal_by_ego(self.ego_vehicle.id).keys():
                self.CBVs_terminated[CBV_id] = True
                self.logger.log(f">> CBV:{CBV_id} reach the goal")
            else:
                self.CBVs_terminated[CBV_id] = False
        return self.CBVs_terminated

    def truncated(self):
        self.CBVs_truncated = {}
        # check whether the CBV has truncated
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).items():
            self.CBVs_truncated[CBV_id] = False if self.scenario_manager.running else True                

        return self.CBVs_truncated

    def done(self):
        self.CBVs_done = {}
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).items():
            self.CBVs_done[CBV_id] = True if self.CBVs_terminated[CBV_id] or self.CBVs_truncated[CBV_id] else False
        return self.CBVs_done