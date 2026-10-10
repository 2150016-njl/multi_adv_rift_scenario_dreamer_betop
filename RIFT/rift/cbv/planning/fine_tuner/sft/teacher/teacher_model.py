#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : teacher_model.py
@Date    : 2025/01/14
'''
from typing import Dict, List

import numpy as np
import torch
from rift.cbv.planning.fine_tuner.sft.teacher.autopilot import AutoPilot
from rift.ego.pdm_lite.nav_planner import _location_to_gps
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class TeacherModel():

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.target_speed = config['teacher_model']['target_speed']
        self.frame_rate = config['frame_rate']

        self.teacher_models: Dict[str: AutoPilot] = {}

    def get_keys(self):
        return set(self.teacher_models.keys())

    def add_CBV(self, CBV, route_elements, CBV_id):
        gps_route, route = self._process_route(route_elements)
        # initialize the teacher model for one CBV
        teacher_model = AutoPilot(self.target_speed, self.frame_rate)
        teacher_model.set_planner(CBV, gps_route, route)
        self.teacher_models[CBV_id] = teacher_model

    def clean_CBV(self, CBV_id):
        self.teacher_models.pop(CBV_id)

    def get_info(self, CBV, CBV_id):
        control = self.teacher_models[CBV_id].run_step()
        # use the target speed and angle for teacher distillation instead of raw actions
        target_speed = self.teacher_models[CBV_id].target_speed

        CBV_state = CarlaDataProvider.get_current_state(CBV)
        speed = CBV_state.dynamic_car_state.center_velocity_2d.magnitude()  # center forward speed
        x = CBV_state.rear_axle.x
        y = CBV_state.rear_axle.y
        heading = CBV_state.rear_axle.heading
        
        return torch.tensor([target_speed, x, y, heading, speed])

    def _process_route(self, route_elements: List):
        lat_ref, lon_ref = CarlaDataProvider.get_gps_info()
        gps_route = []
        route = []
        for wp, connection in route_elements:
            route.append((wp.transform, connection))
            gps_coord = _location_to_gps(lat_ref, lon_ref, wp.transform.location)
            gps_route.append((gps_coord, connection))
        return gps_route, route
