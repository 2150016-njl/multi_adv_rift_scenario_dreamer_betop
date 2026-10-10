#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2023/10/4
"""

from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='rift.gym_carla.envs:CarlaEnv',
)
