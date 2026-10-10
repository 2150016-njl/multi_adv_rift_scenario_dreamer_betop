from functools import lru_cache
from typing import Callable, List, Optional, Set, Tuple, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString

from nuplan_plugin.actor_state.state_representation import ProgressStateSE2, StateSE2

from nuplan_plugin.path.interpolated_path import InterpolatedPath
from nuplan_plugin.path.utils import calculate_progress


def create_path_from_se2(states: List[StateSE2]) -> InterpolatedPath:
    """
    Constructs an InterpolatedPath from a list of StateSE2.
    :param states: Waypoints to construct an InterpolatedPath.
    :return: InterpolatedPath.
    """
    progress_list = calculate_progress(states)

    # Find indices where the progress states are repeated and to be filtered out.
    progress_diff = np.diff(progress_list)
    repeated_states_mask = np.isclose(progress_diff, 0.0)

    progress_states = [
        ProgressStateSE2(progress=progress, x=point.x, y=point.y, heading=point.heading)
        for point, progress, is_repeated in zip(states, progress_list, repeated_states_mask)
        if not is_repeated
    ]
    return InterpolatedPath(progress_states)


def path_to_linestring(path: List[StateSE2]) -> LineString:
    """
    Converts a List of StateSE2 into a LineString
    :param path: path to be converted
    :return: LineString.
    """
    return LineString(np.array([(point.x, point.y) for point in path]))