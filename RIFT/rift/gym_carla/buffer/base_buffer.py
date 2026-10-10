#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : base_buffer.py
@Date    : 2024/09/21
"""


class BaseBuffer:
    name = 'base'

    def __init__(self, num_scenario, mode, logger):
        self.num_scenario = num_scenario
        self.mode = mode
        self.logger = logger

        self.buffer_capacity = 2000
        self.buffer_pos = 0
        self.buffer_full = False
        self.buffer_data = None
        self.temp_buffer = None
    
    def __len__(self):
        return self.buffer_pos
    
    def reset_buffer(self):
        raise NotImplementedError()

    def store(self, data_dict):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

    def save_data(self):
        raise NotImplementedError()