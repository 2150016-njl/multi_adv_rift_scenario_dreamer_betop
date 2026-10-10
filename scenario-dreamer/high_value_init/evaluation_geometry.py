"""Deterministic, raw-state geometry diagnostics for HV offline evaluation.

These helpers only measure sampled states.  They never clamp, snap, discard,
or otherwise alter a generated state, so they are safe to use for frozen V2/
V3/V4 comparisons.
"""

import hashlib
import itertools
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def seed_everything(base_seed: int, record_index: int = 0, sample_index: int = 0) -> int:
    """Set a stable, per-sample RNG state and return the derived seed."""
    seed = (int(base_seed) + 1_000_003 * int(record_index) + 9_176 * int(sample_index)) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_array(state) -> Optional[np.ndarray]:
    try:
        values = np.asarray(state[:6], dtype=np.float64)
    except (TypeError, ValueError, IndexError):
        return None
    return values if values.shape == (6,) and np.isfinite(values).all() else None


def _angle_abs(delta: float) -> float:
    return abs(math.atan2(math.sin(delta), math.cos(delta)))


def _segment_distance_heading(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> Tuple[float, float]:
    delta = end - start
    norm_sq = float(np.dot(delta, delta))
    if norm_sq <= 1e-9:
        return float(np.linalg.norm(point - start)), float("nan")
    ratio = float(np.clip(np.dot(point - start, delta) / norm_sq, 0.0, 1.0))
    nearest = start + ratio * delta
    return float(np.linalg.norm(point - nearest)), math.atan2(float(delta[1]), float(delta[0]))


def nearest_lane(point: np.ndarray, lanes: Iterable[Dict]) -> Optional[Tuple[float, float]]:
    best = None
    for lane in lanes or []:
        points = np.asarray((lane or {}).get("centerline") or [], dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        for index in range(points.shape[0] - 1):
            distance, heading = _segment_distance_heading(point, points[index, :2], points[index + 1, :2])
            if best is None or distance < best[0]:
                best = (distance, heading)
    return best


def match_to_target_order(states: Sequence, targets: Sequence) -> List:
    """Return raw states ordered by minimum position error to target order."""
    count = min(len(states), len(targets))
    if count <= 0:
        return []
    predicted = list(states[:count])
    target_xy = [_state_array(item) for item in targets[:count]]
    pred_xy = [_state_array(item) for item in predicted]
    if any(item is None for item in target_xy + pred_xy):
        return predicted
    if count <= 8:
        best_perm, best_cost = None, None
        for permutation in itertools.permutations(range(count)):
            cost = sum(float(np.linalg.norm(pred_xy[index][:2] - target_xy[target_index][:2]))
                       for target_index, index in enumerate(permutation))
            if best_cost is None or cost < best_cost:
                best_perm, best_cost = permutation, cost
        return [predicted[index] for index in best_perm]
    remaining, result = list(range(count)), []
    for target in target_xy:
        local = min(remaining, key=lambda index: float(np.linalg.norm(pred_xy[index][:2] - target[:2])))
        result.append(predicted[local])
        remaining.remove(local)
    return result


def position_l1(states: Sequence, targets: Sequence) -> Optional[float]:
    ordered = match_to_target_order(states, targets)
    values = []
    for pred, target in zip(ordered, targets):
        pred_state, target_state = _state_array(pred), _state_array(target)
        if pred_state is not None and target_state is not None:
            values.append(float(np.abs(pred_state[:2] - target_state[:2]).mean()))
    return float(np.mean(values)) if values else None


def _obb_axes(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    yaw, half_length, half_width = float(state[2]), float(state[4]) / 2.0, float(state[5]) / 2.0
    return (np.asarray([math.cos(yaw), math.sin(yaw)]),
            np.asarray([-math.sin(yaw), math.cos(yaw)]), half_length, half_width)


def obb_signed_axis_clearance(left, right) -> Optional[float]:
    """SAT signed separation for two oriented vehicle boxes.

    Positive means separated on at least one SAT axis; non-positive means the
    boxes overlap.  This is an axis-projection clearance, deliberately not a
    claim of exact Euclidean rectangle distance.
    """
    a, b = _state_array(left), _state_array(right)
    if a is None or b is None or min(a[4], a[5], b[4], b[5]) <= 0.0:
        return None
    au, av, al, aw = _obb_axes(a)
    bu, bv, bl, bw = _obb_axes(b)
    delta = b[:2] - a[:2]
    gaps = []
    for axis in (au, av, bu, bv):
        a_radius = al * abs(float(np.dot(axis, au))) + aw * abs(float(np.dot(axis, av)))
        b_radius = bl * abs(float(np.dot(axis, bu))) + bw * abs(float(np.dot(axis, bv)))
        gaps.append(abs(float(np.dot(delta, axis))) - a_radius - b_radius)
    return float(max(gaps))


def _numeric(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def geometry_metrics(
        states: Sequence,
        record: Dict,
        lane_half_width: float = 1.75,
        min_center_clearance: float = 2.5,
        fov: Optional[float] = None) -> Dict:
    """Measure nearest/associated-lane and centre/OBB clearance on raw states."""
    condition = record.get("condition") or {}
    map_payload = condition.get("map") or {}
    lanes = map_payload.get("lanes_v3") or map_payload.get("lanes_v4") or []
    associations = (record.get("target") or {}).get("map_associations_v3") or \
                   (record.get("target") or {}).get("map_associations_v4") or []
    targets = (record.get("target") or {}).get("generated_agent_states") or []
    observed = condition.get("observed_agent_states") or []
    ego = condition.get("ego_state") or [0.0, 0.0, 0.0, 0.0, 5.176, 2.297, 1.0]
    half_fov = float(fov if fov is not None else condition.get("fov", 64.0)) / 2.0

    nearest_distances, nearest_heading_errors = [], []
    associated_distances, associated_heading_errors = [], []
    corridor_failures = ego_failures = observed_failures = pair_failures = off_fov_failures = 0
    obb_values, obb_overlap = [], 0
    ego_obb_overlap = observed_obb_overlap = pair_obb_overlap = 0
    invalid_obb = 0

    ordered = match_to_target_order(states, targets)
    for index, state in enumerate(states):
        value = _state_array(state)
        if value is None:
            invalid_obb += 1
            off_fov_failures += 1
            continue
        nearest = nearest_lane(value[:2], lanes)
        if nearest is not None:
            nearest_distances.append(nearest[0])
            nearest_heading_errors.append(_angle_abs(float(value[2]) - nearest[1]))
            corridor_failures += int(nearest[0] > lane_half_width)
        ego_distance = float(np.linalg.norm(value[:2]))
        ego_failures += int(ego_distance < min_center_clearance)
        off_fov_failures += int(abs(value[0]) > half_fov or abs(value[1]) > half_fov)
        if observed:
            valid_observed = [_state_array(item) for item in observed]
            valid_observed = [item for item in valid_observed if item is not None]
            if valid_observed:
                observed_failures += int(min(float(np.linalg.norm(value[:2] - item[:2])) for item in valid_observed) < min_center_clearance)

        ego_gap = obb_signed_axis_clearance(value, ego)
        if ego_gap is None:
            invalid_obb += 1
        else:
            obb_values.append(ego_gap)
            ego_obb_overlap += int(ego_gap <= 0.0)
        for item in observed:
            gap = obb_signed_axis_clearance(value, item)
            if gap is None:
                continue
            obb_values.append(gap)
            observed_obb_overlap += int(gap <= 0.0)

    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            first, second = _state_array(states[left]), _state_array(states[right])
            if first is not None and second is not None:
                pair_failures += int(float(np.linalg.norm(first[:2] - second[:2])) < min_center_clearance)
            gap = obb_signed_axis_clearance(states[left], states[right])
            if gap is not None:
                obb_values.append(gap)
                pair_obb_overlap += int(gap <= 0.0)

    for state, association in zip(ordered, associations):
        value = _state_array(state)
        lane_index = (association or {}).get("lane_index") if isinstance(association, dict) else None
        if value is None or lane_index is None:
            continue
        try:
            lane = lanes[int(lane_index)]
        except (IndexError, TypeError, ValueError):
            continue
        lane_measurement = nearest_lane(value[:2], [lane])
        if lane_measurement is not None:
            associated_distances.append(lane_measurement[0])
            associated_heading_errors.append(_angle_abs(float(value[2]) - lane_measurement[1]))

    obb_overlap = ego_obb_overlap + observed_obb_overlap + pair_obb_overlap
    return {
        "nearest_lane_distance_mean": _numeric(nearest_distances),
        "nearest_lane_distance_max": float(max(nearest_distances)) if nearest_distances else None,
        "lane_heading_abs": _numeric(nearest_heading_errors),
        "associated_lane_distance_mean": _numeric(associated_distances),
        "associated_lane_distance_max": float(max(associated_distances)) if associated_distances else None,
        "associated_lane_heading_abs": _numeric(associated_heading_errors),
        "associated_lane_unavailable_count": max(0, len(states) - len(associated_distances)),
        "lane_corridor_failure_count": corridor_failures,
        "ego_clearance_failure_count": ego_failures,
        "observed_clearance_failure_count": observed_failures,
        "pair_clearance_failure_count": pair_failures,
        "off_fov_failure_count": off_fov_failures,
        "obb_axis_clearance_mean": _numeric(obb_values),
        "obb_axis_clearance_min": float(min(obb_values)) if obb_values else None,
        "obb_overlap_count": obb_overlap,
        "ego_obb_overlap_count": ego_obb_overlap,
        "observed_obb_overlap_count": observed_obb_overlap,
        "pair_obb_overlap_count": pair_obb_overlap,
        "obb_invalid_state_count": invalid_obb,
    }
