#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : coordinate_utils.py
@Date    : 2023/10/22
"""
import numba
import numpy as np


@numba.njit
def inverse_conversion_2d(point, translation, yaw):
    """
    Performs a forward coordinate conversion on a 2D point
    :param point: Point to be converted
    :param translation: 2D translation vector of the new coordinate system
    :param yaw: yaw in radian of the new coordinate system
    :return: Converted point
    """
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    converted_point = rotation_matrix.T @ (point - translation)
    return converted_point