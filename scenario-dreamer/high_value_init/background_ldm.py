"""Ordinary-BV LDM: unchanged Scenario Dreamer network, directional task/losses.

The class intentionally reuses the V3 LDM modules and V4-A loss-only helper.
It adds no topology encoder, node feature, output head, or RIFT-label input.
"""

from high_value_init.high_value_ldm_v4a import RouteConditionedHighValueLDMV4A


class RouteConditionedBackgroundLDM(RouteConditionedHighValueLDMV4A):
    """Variable-current-front-count background scene-completion model."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("forward_min_x", 0.0)
        kwargs.setdefault("target_fov_margin", 2.0)
        super().__init__(*args, **kwargs)

