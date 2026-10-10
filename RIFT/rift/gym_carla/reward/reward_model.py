#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : reward_model.py
@Date    : 2025/02/25
'''

import numpy as np


class DenseRewardModel:
    def __init__(self):
        """
        Dense reward model considering driving naturalism.
        """
        self.params = self._sample_params()
    
    def _sample_params(self):
        """
        Sample hyperparameters.
        """
        return {
            'alpha_collision': 20.0,
            'alpha_boundary': 5.0,
            'alpha_comfort': 0.8,
            'alpha_l_align': 0.5,
            'alpha_vel_align': 0.05,
            'alpha_l_center': 0.6,
            'alpha_center_bias': 0.0,
            'alpha_velocity': 0.1,
            'alpha_timestep': 0.1,
        }

    def get_reward(self, delta_dis, delta_angle, speed, acc, angular_speed, angular_acc, collision, offroad):
        """
            Compute the Dense Reward.
        """        

        R_collision = - (self.params['alpha_collision'] + abs(speed)) * collision
        R_offroad = - self.params['alpha_boundary'] * offroad
        R_comfort = - self.params['alpha_comfort'] * (int(abs(acc) > 4) + int(abs(angular_acc) > 4))
        R_l_align = self.params['alpha_l_align'] * (min(np.cos(delta_angle), 0) + self.params['alpha_vel_align'] * min(np.cos(delta_angle) * speed, 0) + 0.25 * (1 - abs(delta_angle) / (np.pi / 2)))
        R_l_center = - self.params['alpha_l_center'] * int(np.cos(delta_angle) > 0.5) * (abs(delta_dis - self.params['alpha_center_bias']) - 0.05 / np.exp(abs(delta_dis - self.params['alpha_center_bias']) - 0.5))
        R_velocity = self.params['alpha_velocity'] * max(np.cos(delta_angle), 0) * int(3 < abs(speed) < 20) * abs(speed)
        R_timestep = - self.params['alpha_timestep'] * int(abs(speed) > 0 or abs(acc) > 0)
        
        total_reward = (R_collision + R_offroad + R_comfort + R_l_align + 
                        R_l_center + R_velocity + R_timestep)
                        
        return total_reward
    
    def get_params(self):
        """
        Return the current hyperparameters of the reward model.
        """
        return self.params
    


class SparseRewardModel:
    def __init__(self):
        """
        Sparse infraction-based reward model.
        """
        self.params = self._sample_params()
    
    def _sample_params(self):
        """
        Sample hyperparameters.
        """
        return {
            'alpha_collision': 15.0,
            'alpha_boundary': 15.0,
        }
    
    def get_reward(self, collision, offroad):
        """
            Compute the Sparse Reward.
        """
        R_collision = - self.params['alpha_collision'] * collision
        R_offroad = - self.params['alpha_boundary'] * offroad

        total_reward = (R_collision + R_offroad)

        return total_reward
    
    def get_params(self):
        """
        Return the current hyperparameters of the reward model.
        """
        return self.params