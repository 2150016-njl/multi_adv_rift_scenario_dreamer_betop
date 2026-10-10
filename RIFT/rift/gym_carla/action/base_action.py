#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : base_action.py
@Date    : 2024/9/3
"""


class BaseAction(object):
    """
        Base Action
    """
    def __init__(self, acc_range, steer_range, policy_type):
        self.acc_min, self.acc_max = acc_range
        self.steer_min, self.steer_max = steer_range
        self.policy_type = policy_type

    def convert_action(self, action, allow_reverse=False):
        raise NotImplementedError()

    def inverse_action(self, reverse, throttle, steer, brake, allow_reverse=False, learnable=False):
        raise NotImplementedError()


