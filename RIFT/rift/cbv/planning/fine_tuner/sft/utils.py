#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : utils.py
@Date    : 2024/12/26
'''
import torch


def global_to_local(candidate_trajectories, origin, heading, step_interval:int =10):
    """
    Transform global coordinates to local coordinates
    :param candidate_trajectories: (bs, R, M, T, 6)
    :param origin: (bs, 2)
    :param heading: (bs, 1)
    """
    bs, R, M, T, _ = candidate_trajectories.shape

    if T < step_interval:
        local_traj = candidate_trajectories[:, :, :, -1:, :2]  # (bs, R, M, 1, 2)
    else:
        local_traj = candidate_trajectories[:, :, :, step_interval-1::step_interval, :2]  # (bs, R, M, T//step_interval, 2)

    origin = origin.view(bs, 1, 1, 1, 2)  # (bs, 1, 1, 1, 2)
    rot_mat = torch.stack([
        torch.stack([torch.cos(heading), -torch.sin(heading)], dim=1),
        torch.stack([torch.sin(heading),  torch.cos(heading)], dim=1)
    ], dim=1).view(bs, 1, 1, 1, 2, 2)  # (bs, 1, 1, 1, 2, 2)
    local_pos = torch.einsum('brmtc,brmtcd->brmtd', local_traj - origin, rot_mat)  # (bs, R, M, T//step_interval, 2)

    return local_pos  #  (bs, R, M, T//step_interval, 2)

    