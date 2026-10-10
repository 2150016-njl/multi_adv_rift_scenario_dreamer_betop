from typing import Deque, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from nuplan_plugin.actor_state.ego_state import EgoState
from nuplan_plugin.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan_plugin.actor_state.vehicle_parameters import VehicleParameters


def _se2_vel_acc_to_ego_state(
    state: StateSE2,
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    timestamp: float,
    vehicle: VehicleParameters,
) -> EgoState:
    """
    Convert StateSE2, velocity and acceleration to EgoState given a timestamp.

    :param state: input SE2 state
    :param velocity: [m/s] longitudinal velocity, lateral velocity
    :param acceleration: [m/s^2] longitudinal acceleration, lateral acceleration
    :param timestamp: [s] timestamp of state
    :return: output agent state
    """
    return EgoState.build_from_rear_axle(
        rear_axle_pose=state,
        rear_axle_velocity_2d=StateVector2D(*velocity),
        rear_axle_acceleration_2d=StateVector2D(*acceleration),
        tire_steering_angle=0.0,
        time_point=TimePoint(int(timestamp * 1e6)),
        vehicle_parameters=vehicle,
        is_in_auto_mode=True,
    )


def _get_fixed_timesteps(state: EgoState, future_horizon: float, step_interval: float) -> List[float]:
    """
    Get a fixed array of timesteps starting from a state's time.

    :param state: input state
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :return: constructed timestep list
    """
    timesteps = np.arange(0.0, future_horizon, step_interval) + step_interval
    timesteps += state.time_point.time_s

    return list(timesteps.tolist())


def _project_from_global_to_ego_centric_ds(
    ego_poses: npt.NDArray[np.float32], values: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Project value from the global xy frame to the ego centric ds frame.

    :param ego_poses: [x, y, heading] with size [planned steps, 3].
    :param values: values in global frame with size [planned steps, 2]
    :return: values projected onto the new frame with size [planned steps, 2]
    """
    headings = ego_poses[:, -1:]

    values_lon = values[:, :1] * np.cos(headings) + values[:, 1:2] * np.sin(headings)
    values_lat = values[:, :1] * np.sin(headings) - values[:, 1:2] * np.cos(headings)
    values = np.concatenate((values_lon, values_lat), axis=1)
    return values


def _get_velocity_and_acceleration(
    ego_poses: List[StateSE2], ego_history: Deque[EgoState], timesteps: List[float]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Given the past, current and planned ego poses, estimate the velocity and acceleration by taking the derivatives.

    :param ego_poses: a list of the planned ego poses
    :param ego_history: the ego history that includes the current
    :param timesteps: [s] timesteps of the planned ego poses
    :return: the approximated velocity and acceleration in ego centric frame
    """
    ego_history_len = len(ego_history)
    current_ego_state = ego_history[-1]

    # Past and current
    timesteps_past_current = [state.time_point.time_s for state in ego_history]
    ego_poses_past_current: npt.NDArray[np.float32] = np.stack(
        [np.array(state.rear_axle.serialize()) for state in ego_history]
    )

    # Planned
    dt = current_ego_state.time_point.time_s - ego_history[-2].time_point.time_s
    timesteps_current_planned: npt.NDArray[np.float32] = np.array([current_ego_state.time_point.time_s] + timesteps)
    ego_poses_current_planned: npt.NDArray[np.float32] = np.stack(
        [current_ego_state.rear_axle.serialize()] + [pose.serialize() for pose in ego_poses]
    )

    # Interpolation to have equal space for derivatives
    ego_poses_interpolate = interp1d(
        timesteps_current_planned, ego_poses_current_planned, axis=0, fill_value='extrapolate'
    )
    timesteps_current_planned_interp = np.arange(
        start=current_ego_state.time_point.time_s, stop=timesteps[-1] + 1e-6, step=dt
    )
    ego_poses_current_planned_interp = ego_poses_interpolate(timesteps_current_planned_interp)

    # Combine past current and planned
    timesteps_past_current_planned = [*timesteps_past_current, *timesteps_current_planned_interp[1:]]
    ego_poses_past_current_planned: npt.NDArray[np.float32] = np.concatenate(
        [ego_poses_past_current, ego_poses_current_planned_interp[1:]], axis=0
    )

    # Take derivatives
    ego_velocity_past_current_planned = approximate_derivatives(
        ego_poses_past_current_planned[:, :2], timesteps_past_current_planned, axis=0
    )
    ego_acceleration_past_current_planned = approximate_derivatives(
        ego_poses_past_current_planned[:, :2], timesteps_past_current_planned, axis=0, deriv_order=2
    )

    # Only take the planned for output
    ego_velocity_planned_xy = ego_velocity_past_current_planned[ego_history_len:]
    ego_acceleration_planned_xy = ego_acceleration_past_current_planned[ego_history_len:]

    # Projection
    ego_velocity_planned_ds = _project_from_global_to_ego_centric_ds(
        ego_poses_current_planned_interp[1:], ego_velocity_planned_xy
    )
    ego_acceleration_planned_ds = _project_from_global_to_ego_centric_ds(
        ego_poses_current_planned_interp[1:], ego_acceleration_planned_xy
    )

    # Interpolate back
    ego_velocity_interp_back = interp1d(
        timesteps_past_current_planned[ego_history_len:], ego_velocity_planned_ds, axis=0, fill_value='extrapolate'
    )
    ego_acceleration_interp_back = interp1d(
        timesteps_past_current_planned[ego_history_len:],
        ego_acceleration_planned_ds,
        axis=0,
        fill_value='extrapolate',
    )

    ego_velocity_planned_ds = ego_velocity_interp_back(timesteps)
    ego_acceleration_planned_ds = ego_acceleration_interp_back(timesteps)

    return ego_velocity_planned_ds, ego_acceleration_planned_ds


def approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f'{poly_order} < {window_length} does not hold!')

    dx = np.diff(x)
    if not (dx > 0).all():
        raise RuntimeError('dx is not monotonically increasing!')

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y, polyorder=poly_order, window_length=window_length, deriv=deriv_order, delta=dx, axis=axis
    )
    return derivative