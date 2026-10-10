#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py.py
@Date    : 2024/09/21
"""
from rift.gym_carla.buffer.ego_rollout_buffer import EgoRolloutBuffer
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.gym_carla.buffer.collect_buffer import CollectBuffer


BUFFER_LIST = {
    'ego_rollout': EgoRolloutBuffer,
    'cbv_rollout': CBVRolloutBuffer,
    'collect': CollectBuffer,
}
