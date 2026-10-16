import math

from hv_v8.common import yaw_wrap
from hv_v8.p2_support import d_interval_for_box


def test_yaw_wrap():
    assert abs(abs(yaw_wrap(math.pi)) - math.pi) < 1e-12
    assert abs(abs(yaw_wrap(3 * math.pi)) - math.pi) < 1e-12


def test_slot0_interval_centerline_ahead():
    # Ego frame == world frame; lane heading +x, d is +y.
    got = d_interval_for_box(10.0, 0.0, 0.0, 3.6, slot=0)
    assert got is not None
    assert got[0] < 0 < got[1]


def test_slot0_rejects_centerline_behind_if_lane_width_cannot_reach_front():
    got = d_interval_for_box(-40.0, 0.0, 0.0, 3.6, slot=0)
    assert got is None


def test_slot1_allows_behind():
    got = d_interval_for_box(-20.0, 0.0, 0.0, 3.6, slot=1)
    assert got is not None
