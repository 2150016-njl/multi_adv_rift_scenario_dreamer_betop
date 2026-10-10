#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : cbv_action.py
@Date    : 2024/09/24
"""
import numpy as np

from rift.gym_carla.action.base_action import BaseAction
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class CBVAction(BaseAction):
    """
        CBV_action
    """
    def __init__(self, acc_range, steer_range, policy_type):
        super().__init__(acc_range, steer_range, policy_type)

    def convert_action(self, action):
        if self.policy_type == 'rl':
            return self.convert_rl_action(action)
        elif self.policy_type == 'il':
            return self.convert_il_action(action)
        else:
            raise NotImplementedError(f"Unsupported CBV policy type: {self.policy_type}.")

    def convert_rl_action(self, action, allow_reverse=True):
        acc, steer = action  # continuous action: acc, steering

        # normalize and clip the action
        acc = acc * self.acc_max
        steer = steer * self.steer_max
        acc = max(min(self.acc_max, acc), self.acc_min)
        steer = max(min(self.steer_max, steer), self.steer_min)

        # Convert acceleration to throttle and brake
        if allow_reverse:
            # allow reverse, convert acc to throttle
            if acc > 0:
                throttle = np.clip(acc / 3, 0, 1)
                brake = 0
                reverse = False
            else:
                reverse = True
                throttle = -np.clip(acc / 3, -1, 0)
                brake = 0
        else:
            reverse = False
            # disable reverse, convert acc to positive throttle and brake
            if acc > 0:
                throttle = np.clip(acc / 3, 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = np.clip(-acc / 8, 0, 1)

        return [reverse, throttle, steer, brake]
    
    def convert_il_action(self, action):
        # Imitation learning based method uses PID controller to generate throttle, steer and brake
        throttle, steer, brake = action
        # Imitation learning based method does not support reverse
        return [False, throttle, steer, brake]

    def inverse_rl_action(self, reverse, throttle, steer, brake, allow_reverse=True, learnable=None):
        if allow_reverse:
            acc = throttle * -3.0 if reverse else throttle * 3.0
        else:
            acc = brake * -8.0 if brake != 0 else throttle * 3.0

        return [acc, steer]
    
    def inverse_il_action(self, throttle, steer, brake):
        acc = brake * -8.0 if brake != 0 else throttle * 3.0

        return [acc, steer]
