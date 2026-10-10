#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : pid_controller.py
@Date    : 2024/11/19
'''

from collections import deque
from typing import List
import numpy as np
import torch

class PID(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative



class PIDController(object):
    
    def __init__(self, sample_interval=10, max_throttle=1.0, brake_speed=0.4,brake_ratio=1.1, clip_delta=1.0):
        
        self.sample_interval = int(sample_interval)  # default to be 10
        self.turn_controller = PID(K_P=1.25, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PID(K_P=5.0, K_I=0.5, K_D=1.0, n=20)
        self.alpha = 0.5
        self.beta = 2.5
        self.min_aim_dis = 5.0
        self.max_aim_dis = 8.0
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.desired_speed = None
        self.delta_angle = None

    def control_pid(self, local_pos: np.ndarray, speed: float):
        ''' Predicts vehicle control with a PID controller.
        Args:
            local_pos: the descrete waypoints of the planned trajectory
            speed: current speed (m/s)
        '''

        # resample the trajectory
        local_pos = local_pos[
            self.sample_interval-1 : : self.sample_interval
        ] if local_pos.shape[0] >= self.sample_interval else local_pos[-1:]

        segment_vectors = np.diff(local_pos, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        desired_speed = segment_lengths.mean()  # desired speed
        
        aim_dist = np.clip(self.alpha * speed + self.beta, self.min_aim_dis, self.max_aim_dis)  # aim_dis range from 4m to 7m

        norms = np.linalg.norm(local_pos[:-1], axis=1)
        closest_index = np.abs(norms - aim_dist).argmin()
        aim = local_pos[closest_index]  # aim location

        # brake
        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio
        # throttle
        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0
        # steer
        angle = np.degrees(-np.arctan2(aim[1], aim[0])) / 90
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        self.desired_speed = desired_speed
        self.delta_angle = angle

        return throttle, steer, brake
    
    def batch_control_pid(self, local_traj: torch.Tensor, speed: torch.Tensor):
        ''' Predicts vehicle control with a PID controller.
        Args:
            local_traj: the descrete waypoints of the planned local trajectory (bs, R, M, T // step interval, 2)
            speed: current speed (m/s) (bs,)
        '''
        bs, R, M, T, c = local_traj.shape

        # -----------------------------
        # (1) Compute target_speed
        # -----------------------------
        if T == 1:
            # If there is only one point, compute distance from the origin (0,0)
            # local_traj[..., 0, :] => shape: (bs, R, M, 2)
            # .norm(dim=-1) => shape: (bs, R, M)
            target_speed = local_traj[..., 0, :].norm(dim=-1, p=2)
        else:
            # If T > 1, compute the mean distance between consecutive points
            # diff => (bs, R, M, T-1, 2)
            diff = local_traj[..., 1:, :] - local_traj[..., :-1, :]
            # dist => (bs, R, M, T-1)
            dist = diff.norm(dim=-1, p=2)
            # Mean over (T-1) => (bs, R, M)
            target_speed = dist.mean(dim=-1)

        # -----------------------------
        # (2) Compute angle
        # -----------------------------
        # 2.1 Compute aim_dist => shape: (bs, R, M)
        expanded_speed = speed.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1)
        aim_dist = (self.alpha * expanded_speed + self.beta).clamp(min=self.min_aim_dis, max=self.max_aim_dis)

        # 2.2 Compute the L2 norm for each point => shape: (bs, R, M, T)
        norm_vals = local_traj.norm(dim=-1, p=2)

        # Compare with aim_dist => shape: (bs, R, M, T)
        # aim_dist.unsqueeze(-1) => (bs, R, M, 1), then broadcast to match norm_vals
        abs_diff = (norm_vals - aim_dist.unsqueeze(-1)).abs()

        # Choose the index in T-dimension that minimizes the distance to aim_dist
        # shape: (bs, R, M)
        aim_idx = abs_diff.argmin(dim=-1)

        # 2.3 Gather aim points => shape: (bs, R, M, 2)
        aim_idx_expanded = aim_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 2)
        aim = torch.gather(local_traj, dim=-2, index=aim_idx_expanded).squeeze(-2)

        # 2.4 Compute angle => shape: (bs, R, M)
        # angle = deg(-atan2(y, x)) / 90
        angle_rad = -torch.atan2(aim[..., 1], aim[..., 0])
        angle_deg = torch.rad2deg(angle_rad)
        angle = angle_deg / 90.0

        return target_speed, angle
