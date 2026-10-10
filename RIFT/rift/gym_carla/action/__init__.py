#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py.py
@Date    : 2024/09/24
"""
from rift.gym_carla.action.cbv_action import CBVAction
from rift.gym_carla.action.ego_action import EgoAction

ACTION_TYPE = {
    'ego_action': EgoAction,
    'cbv_action': CBVAction,
}