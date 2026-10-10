#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : ego_rollout_buffer.py
@Date    : 2024/09/21
"""
from collections import defaultdict, deque
import copy
import numpy as np

from rift.gym_carla.buffer.base_buffer import BaseBuffer


class EgoRolloutBuffer(BaseBuffer):
    name = 'EgoRolloutBuffer'

    """
        This buffer designed for rl-based on-policy methods
    """
    def __init__(
            self,
            num_scenario,
            mode,
            ego_config,
            logger=None):
        super().__init__(num_scenario, mode, logger)
        assert self.mode == 'train_ego', f'Only initialize {self.name} when training the rl-based onpolicy ego agent'

        self.buffer_capacity = ego_config['buffer_capacity']
        self.data_keys = ego_config['data_keys']
        self.ego_id_set = set()

        self.reset_buffer()

    def reset_buffer(self):
        self.buffer_pos = 0
        self.buffer_full = False

        # Use defaultdict to create deque for each data type
        self.buffer_data = {
            key: deque(maxlen=self.buffer_capacity) for key in self.data_keys
        }
        # Temporary buffer to store intermediate data
        self.temp_buffer = {key: defaultdict(list) for key in self.buffer_data}

    def process_data_dict(self, data_dict):
        """
            the processed_data_dict contains continuous ego trajectory data
        """
        # initialize
        processed_data_dict = {key: [] for key in self.buffer_data.keys()}

        scenario_lengths = set(len(data) for key, data in data_dict.items() if key in self.buffer_data.keys())
        assert len(scenario_lengths) == 1, 'all the data in the data dict should have same length'
        scenario_length = scenario_lengths.pop()

        for i in range(scenario_length):
            ego_id = data_dict['ego_id'][i]
            # transfer necessary data from the data_dict to the temp buffer
            for key, value in self.temp_buffer.items():
                value[ego_id].append(data_dict[key][i])

            if data_dict['ego_terminal'][i]:
                # the trajectory of ego is completed, pop the whole trajectory
                for key, value in processed_data_dict.items():
                    value.extend(self.temp_buffer[key].pop(ego_id))  # pop out specific whole trajectory data

        data_lengths = set(len(data) for data in processed_data_dict.values())
        assert len(data_lengths) == 1, 'the data in the processed data dict should have same length'
        data_length = data_lengths.pop()

        return processed_data_dict, data_length

    def store(self, data_dict):
        processed_data_dict, data_length = self.process_data_dict(data_dict)
        if data_length > 5:  # ignore too short ego trajectory
            if self.buffer_pos + data_length >= self.buffer_capacity:
                # the buffer can just hold part of the trajectory data
                for i in range(data_length):
                    if self.buffer_pos < self.buffer_capacity:
                        for key, data in self.buffer_data.items():
                            # key code: store the data_dict to the self.buffer_data
                            data.append(processed_data_dict[key][i])
                        self.buffer_pos += 1
                    else:
                        break
                self.buffer_full = True
            else:
                # the buffer still can hold the whole trajectory data
                for key, data in self.buffer_data.items():
                    data.extend(processed_data_dict[key])
                self.buffer_pos += data_length

    def get_all_np_data(self):
        '''
        the np array data for traditional rl-based on-policy methods
        '''
        assert self.buffer_pos == self.buffer_capacity, 'only get the data when the buffer is full'
        batch_dict = {}
        for key, deque_data in self.buffer_data.items():
            batch_dict[key] = np.stack([np.array(item) for item in deque_data]).reshape(self.buffer_capacity, -1)

        return batch_dict
