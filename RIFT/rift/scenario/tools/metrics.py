#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : metrics.py
@Date    : 2025/02/12
'''

import time
from typing import List
import numpy as np

from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState

# Global constants (in uppercase)
K_TTC = 1.0      # Used for TTC determination condition
PI = 3.14159
GAMMA = 0.01396
D_SAFE = 0
KW = 1.2         # Used for EI parallel determination condition
ROUND_DIGITS = 2


# --------------------- Helper Functions ---------------------

def get_vehicle_bbox(x, y, l, w, h):
    """
    Compute the 4 corners of a vehicle's bounding box based on its center (x, y),
    length l, width w, and heading h. This uses the same rotation method as in the
    original compute_RTTC.
    """
    offsets = np.array([
        [l / 2, w / 2],
        [l / 2, -w / 2],
        [-l / 2, w / 2],
        [-l / 2, -w / 2]
    ])
    # Note: The rotation matrix here matches the original code in compute_RTTC.
    rotate_matrix = np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)]
    ])
    return np.array([x, y]) + np.dot(offsets, rotate_matrix)


def get_vehicle_corners_tdm(l, w, h):
    """
    Compute the 4 corners of a vehicle relative to its center using the formulas
    from calculate_TDM_MFD. The corners are computed in a vectorized manner.
    Returns an array of shape (4, 2) where each row is a corner.
    """
    # The formulas below are derived from the original AA1-AA4 and BB1-BB4 definitions.
    corners = np.array([
        [ l/2 * np.cos(h) + w/2 * np.sin(h),  l/2 * np.sin(h) - w/2 * np.cos(h)],
        [ l/2 * np.cos(h) - w/2 * np.sin(h),  l/2 * np.sin(h) + w/2 * np.cos(h)],
        [-l/2 * np.cos(h) + w/2 * np.sin(h), -l/2 * np.sin(h) - w/2 * np.cos(h)],
        [-l/2 * np.cos(h) - w/2 * np.sin(h), -l/2 * np.sin(h) + w/2 * np.cos(h)]
    ])
    return corners


# --------------------- Main Calculation Functions ---------------------

def compute_TTC_lon_1(x_A, y_A, v_A, h_A, l_A, w_A,
                    x_B, y_B, v_B, h_B, l_B, w_B):
    """
    Compute the longitudinal Time-To-Collision (TTC) using method 1.
    Projects the relative position and velocity onto the heading of vehicle A.
    Returns the TTC if the following vehicle is moving faster and lateral offset is safe,
    otherwise returns np.nan.
    """
    theta_A = np.array([np.cos(h_A), np.sin(h_A)])
    theta_A_perp = np.array([-np.sin(h_A), np.cos(h_A)])
    relative_pos = np.array([x_B - x_A, y_B - y_A])

    S0lon = np.dot(relative_pos, theta_A)
    S0lat = np.dot(relative_pos, theta_A_perp)
    v_B_vector = np.array([v_B * np.cos(h_B), v_B * np.sin(h_B)])
    v_B_lon = np.dot(v_B_vector, theta_A)
    v_B_lat = np.dot(v_B_vector, theta_A_perp)

    if S0lon * (v_A - v_B_lon) > 0:  # Ensure that the following vehicle is moving faster than the leading vehicle
        ttc_lon_1 = (abs(S0lon) - (l_A + l_B) / 2) / abs(v_A - v_B_lon)
        # Ensure that at time ttc_lon_1, the lateral offset is within the safe threshold
        if abs(S0lat + v_B_lat * ttc_lon_1) <= K_TTC * ((w_A + w_B) / 2) and ttc_lon_1 >= 0:
            return ttc_lon_1
    return np.nan


def compute_TTC_lon_2(x_A, y_A, v_A, h_A, l_A, w_A,
                    x_B, y_B, v_B, h_B, l_B, w_B):
    """
    Compute the longitudinal Time-To-Collision (TTC) using method 2.
    Projects the relative position and velocity onto the heading of vehicle B.
    Returns the TTC if the following vehicle (in B's frame) is moving faster than A and lateral offset is safe,
    otherwise returns np.nan.
    """
    theta_B = np.array([np.cos(h_B), np.sin(h_B)])
    theta_B_perp = np.array([-np.sin(h_B), np.cos(h_B)])
    relative_pos = np.array([x_A - x_B, y_A - y_B])
    S0lon = np.dot(relative_pos, theta_B)
    S0lat = np.dot(relative_pos, theta_B_perp)

    v_A_vector = np.array([v_A * np.cos(h_A), v_A * np.sin(h_A)])
    v_A_lon = np.dot(v_A_vector, theta_B)
    v_A_lat = np.dot(v_A_vector, theta_B_perp)
    if S0lon * (v_B - v_A_lon) > 0:  # Ensure that the following vehicle (in B's frame) is moving faster than the other
        ttc_lon_2 = (abs(S0lon) - (l_A + l_B) / 2) / abs(v_B - v_A_lon)
        # Ensure that at time ttc_lon_2, the lateral offset is within the safe threshold
        if abs(S0lat + v_A_lat * ttc_lon_2) <= K_TTC * ((w_A + w_B) / 2) and ttc_lon_2 >= 0:
            return ttc_lon_2
    return np.nan


def is_ray_intersect_segment(ray_origin_x, ray_origin_y, ray_direction_x, ray_direction_y,
                             segment_start_x, segment_start_y, segment_end_x, segment_end_y):
    """
    Determine whether a ray (origin + direction) intersects with a line segment.
    Returns the distance from the ray origin to the intersection point if one exists,
    otherwise returns None.
    """
    ray_origin = np.array([ray_origin_x, ray_origin_y])
    ray_direction = np.array([ray_direction_x, ray_direction_y])
    segment_start = np.array([segment_start_x, segment_start_y])
    segment_end = np.array([segment_end_x, segment_end_y])

    v1 = ray_origin - segment_start
    v2 = segment_end - segment_start
    v3 = np.array([-ray_direction[1], ray_direction[0]])
    norm_v3 = np.linalg.norm(v3)
    if norm_v3 < 1e-10:
        return None
    v3 = v3 / norm_v3

    dot = np.dot(v2, v3)
    if abs(dot) < 1e-10:
        if abs(np.cross(v1, v2)) < 1e-10:
            t0 = np.dot(segment_start - ray_origin, ray_direction)
            t1 = np.dot(segment_end - ray_origin, ray_direction)
            if t0 >= 0 and t1 >= 0:
                return min(t0, t1)
            if t0 < 0 and t1 < 0:
                return None
            return 0
        return None

    t1 = np.cross(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot

    if 0 <= t2 <= 1:
        return t1
    return None


def compute_RTTC(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B):
    """
    Compute the Relative Time-To-Collision (RTTC) based on the bounding boxes of two vehicles
    and their relative velocity.
    Returns the RTTC value if a valid collision time is computed, otherwise returns np.nan.
    
    [Optimization]
    - The vehicle bounding boxes are now computed using the helper function get_vehicle_bbox.
    """
    # Compute bounding boxes for vehicle A and B using the new helper function.
    bbox_A = get_vehicle_bbox(x_A, y_A, l_A, w_A, h_A)
    bbox_B = get_vehicle_bbox(x_B, y_B, l_B, w_B, h_B)

    v_A_array = np.array([v_A * np.cos(h_A), v_A * np.sin(h_A)])
    v_B_array = np.array([v_B * np.cos(h_B), v_B * np.sin(h_B)])
    v_rel = v_A_array - v_B_array

    DTC = np.nan

    # Check intersections from vehicle A's corners along the relative velocity vector
    for i in range(4):
        dist_has_negative = False
        for j in range(4):
            dist = is_ray_intersect_segment(
                bbox_A[i][0], bbox_A[i][1],
                v_rel[0], v_rel[1],
                bbox_B[j][0], bbox_B[j][1],
                bbox_B[(j + 1) % 4][0], bbox_B[(j + 1) % 4][1]
            )
            if dist is not None:
                if np.isnan(DTC):
                    DTC = dist
                if dist > 0:
                    DTC = min(DTC, dist)
                if dist < 0:
                    dist_has_negative = True
                if dist_has_negative and dist > 0:
                    return 0

    # Check intersections from vehicle B's corners along the opposite relative velocity vector
    for i in range(4):
        dist_has_negative = False
        for j in range(4):
            dist = is_ray_intersect_segment(
                bbox_B[i][0], bbox_B[i][1],
                -v_rel[0], -v_rel[1],
                bbox_A[j][0], bbox_A[j][1],
                bbox_A[(j + 1) % 4][0], bbox_A[(j + 1) % 4][1]
            )
            if dist is not None:
                if np.isnan(DTC):
                    DTC = dist
                if dist > 0:
                    DTC = min(DTC, dist)
                if dist < 0:
                    dist_has_negative = True
                if dist_has_negative and dist > 0:
                    return 0

    if not np.isnan(DTC) and np.linalg.norm(v_rel) > 1e-12:
        RTTC = DTC / np.linalg.norm(v_rel)
        return RTTC
    return np.nan


def calculate_v_Br(x_A, y_A, v_A, h_A, x_B, y_B, v_B, h_B):
    """
    Calculate the relative speed (v_Br) along the line connecting the centers of the two vehicles.
    This is done by projecting the velocity difference onto the unit vector from A to B.
    A negative v_Br indicates that the vehicles are approaching each other.
    """
    delta_x = x_B - x_A
    delta_y = y_B - y_A
    norm_delta = np.sqrt(delta_x ** 2 + delta_y ** 2)
    if norm_delta != 0:
        unit_vector = np.array([delta_x / norm_delta, delta_y / norm_delta])
        velocity_diff = np.array([
            v_B * np.cos(h_B) - v_A * np.cos(h_A),
            v_B * np.sin(h_B) - v_A * np.sin(h_A)
        ])
        v_Br = -np.dot(unit_vector, velocity_diff)
    else:
        v_Br = 0
    return v_Br


def calculate_TDM_MFD(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B):
    """
    Calculate the Time-to-Distance Margin (TDM), the In-Depth margin (MFD), and the shortest distance
    among 16 computed vectors (used in the computation of ACT and EI).
    
    [Optimization]
    - The four vehicle corners are computed using the helper function get_vehicle_corners_tdm.
    - The 16 vector differences for ACT are computed in a vectorized manner using NumPy broadcasting.
    """
    # Compute the velocity difference and check for division by zero.
    v_diff = np.array([
        v_B * np.cos(h_B) - v_A * np.cos(h_A),
        v_B * np.sin(h_B) - v_A * np.sin(h_A)
    ])
    norm_v_diff = np.linalg.norm(v_diff)
    if norm_v_diff < 1e-12:
        return None, np.nan, np.nan  # Avoid division by zero
    theta_B_prime = v_diff / norm_v_diff

    # Compute delta (distance between centers)
    delta = np.array([x_B - x_A, y_B - y_A])
    # D_t1: the perpendicular distance from delta to theta_B_prime
    D_t1 = np.linalg.norm(delta - np.dot(delta, theta_B_prime) * theta_B_prime)
    AB = delta  # Same as delta

    # Compute vehicle corners for A and B (relative to the center) using the helper function.
    AA = get_vehicle_corners_tdm(l_A, w_A, h_A)  # shape (4, 2)
    BB = get_vehicle_corners_tdm(l_B, w_B, h_B)  # shape (4, 2)

    # Compute the maximum projected distance for vehicle A in the theta_B_prime direction.
    d_A = np.linalg.norm(AA - np.outer(np.dot(AA, theta_B_prime), theta_B_prime), axis=1)
    d_A_max = np.max(d_A)
    # Compute the maximum projected distance for vehicle B.
    d_B = np.linalg.norm(BB - np.outer(np.dot(BB, theta_B_prime), theta_B_prime), axis=1)
    d_B_max = np.max(d_B)

    MFD = D_t1 - (d_A_max + d_B_max)
    D_B_prime = -np.dot(delta, theta_B_prime)
    TDM = D_B_prime / norm_v_diff
    InDepth = D_SAFE - MFD

    # ------------------ Vectorized ACT Computation ------------------
    # Instead of manually computing 16 vectors, we use broadcasting:
    # For each corner in AA (shape: (4,2)) and each corner in BB (shape: (4,2)),
    # compute the vector: (BB[j] + AB - AA[i]) for all i,j.
    diff = BB[np.newaxis, :, :] + AB - AA[:, np.newaxis, :]  # shape: (4, 4, 2)
    # Compute the norms of all 16 vectors.
    norms = np.linalg.norm(diff, axis=2)
    dis_shortest = np.min(norms)
    # -----------------------------------------------------------------

    return TDM, InDepth, dis_shortest


def safe_round(value, digits):
    """
    Safely round the value to the given number of digits.
    If the value is NaN, it returns NaN.
    """
    return round(value, digits) if not np.isnan(value) else np.nan


def compute_ego_critical_metrics(ego_state: CarlaAgentState, ego_nearby_agent_states: List[CarlaAgentState]):
    """
    For each nearby vehicle, compute the vehicle-to-vehicle metrics, then aggregate them.
    
    Metrics:
      - RTTC, ACT: use the minimum value among all nearby vehicles.
      - EI: use the maximum value among all nearby vehicles.
    
    If there are no nearby vehicles, all metrics are returned as NaN.
    """
    if not ego_nearby_agent_states:
        return {'RTTC': np.nan, 'ACT': np.nan, 'EI': np.nan}
    
    # Compute metrics for each nearby agent using list comprehension.
    metrics_list = [
        get_a2a_metrics(
            ego_state.center.x, ego_state.center.y, ego_state.dynamic_car_state.speed, ego_state.center.heading,
            ego_state.car_footprint.length, ego_state.car_footprint.width,
            agent_state.center.x, agent_state.center.y, agent_state.dynamic_car_state.speed, agent_state.center.heading,
            agent_state.car_footprint.length, agent_state.car_footprint.width
        )
        for agent_state in ego_nearby_agent_states
    ]
    
    # Aggregate each metric:
    # - For 'RTTC', and 'ACT', take the minimum value.
    # - For 'EI', take the maximum value.
    aggregated_metrics = {}
    for key in ['RTTC', 'ACT', 'EI']:
        values = np.array([m[key] for m in metrics_list])
        if np.all(np.isnan(values)):
            aggregated_metrics[key] = np.nan
        else:
            if key == 'EI':
                aggregated_metrics[key] = np.nanmax(values)
            else:
                aggregated_metrics[key] = np.nanmin(values)
    
    return aggregated_metrics


def get_a2a_metrics(x_A, y_A, v_A, h_A, l_A, w_A,
                    x_B, y_B, v_B, h_B, l_B, w_B):
    """
    Compute real-time metrics based on the provided vehicle parameters.
    
    Metrics computed:
      - TTC: Time-to-collision using two different longitudinal methods.
      - RTTC: Revised time-to-collision, calculated if vehicles are approaching.
      - ACT: Actual collision time based on the shortest distance and relative speed.
      - EI_v1: An evaluation index computed as InDepth / TDM.
      - EI_v2: An alternative evaluation index computed as InDepth / RTTC.
    """
    # # Compute TTC using two longitudinal methods.
    # ttc_values = [
    #     compute_TTC_lon_1(x_A, y_A, v_A, h_A, l_A, w_A,
    #                       x_B, y_B, v_B, h_B, l_B, w_B),
    #     compute_TTC_lon_2(x_A, y_A, v_A, h_A, l_A, w_A,
    #                       x_B, y_B, v_B, h_B, l_B, w_B)
    # ]
    # # If both TTC values are NaN, then TTC is set to NaN.
    # # Otherwise, take the smallest non-NaN value and round it.
    # TTC = np.nan if np.all(np.isnan(ttc_values)) else safe_round(np.nanmin(ttc_values), 4)

    # Calculate the relative speed along the line connecting the two vehicles.
    v_Br = calculate_v_Br(x_A, y_A, v_A, h_A, x_B, y_B, v_B, h_B)

    # Compute RTTC only if vehicles are approaching (v_Br >= 0); otherwise, set RTTC to NaN.
    RTTC_val = compute_RTTC(x_A, y_A, v_A, h_A, l_A, w_A,
                            x_B, y_B, v_B, h_B, l_B, w_B) if v_Br >= 0 else np.nan
    RTTC = np.nan if (RTTC_val is None or RTTC_val < 0) else safe_round(RTTC_val, ROUND_DIGITS)

    # Only compute ACT, EI, and EI_update if the vehicles are approaching (v_Br > 0).
    if v_Br > 0:
        TDM, InDepth, dis_shortest = calculate_TDM_MFD(
            x_A, y_A, v_A, h_A, l_A, w_A,
            x_B, y_B, v_B, h_B, l_B, w_B
        )
        # If any computed value is None or invalid, set it to NaN.
        TDM = np.nan if TDM is None or TDM < 0 else TDM
        InDepth = np.nan if InDepth is None else InDepth
        dis_shortest = np.nan if dis_shortest is None else dis_shortest

        if not np.isnan(RTTC):
            # Compute ACT: shortest distance divided by relative speed (ensuring non-negative).
            ACT = safe_round(dis_shortest / v_Br, ROUND_DIGITS) if dis_shortest / v_Br >= 0 else np.nan
            # Compute EI_v1: InDepth divided by TDM (if TDM is zero, return NaN).
            EI_v1 = safe_round(InDepth / TDM, ROUND_DIGITS) if TDM != 0 else np.nan
            # Compute EI_v2: InDepth divided by RTTC (if RTTC is zero, return NaN).
            EI_v2 = safe_round(InDepth / RTTC, ROUND_DIGITS) if RTTC != 0 else np.nan
        else:
            ACT = EI_v1 = EI_v2 = np.nan
    else:
        ACT = EI_v1 = EI_v2 = np.nan

    return {
        'RTTC': RTTC,
        'ACT': ACT,
        'EI': EI_v2,
    }


if __name__ == '__main__':
    # Example input parameters for vehicle A (ego vehicle)
    x_A = 0          # Absolute coordinate in meters
    y_A = 0
    v_A = 5          # Speed in m/s
    h_A = 0          # Heading angle in radians, range [-PI, PI] (e.g., 1.57 is 90°)
    l_A = 4.8
    w_A = 1.8

    # Example input parameters for vehicle B (surrounding vehicle)
    x_B = 20
    y_B = 0
    v_B = 5
    h_B = -3.14
    l_B = 4.8
    w_B = 1.8
    
    start_time = time.time()
    metrics = get_a2a_metrics(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B)
    end_time = time.time()
    print(f'Execution time: {end_time - start_time:.6f} seconds')
    print(metrics)

