import unittest
import numpy as np
from audit_hv_forward_map_proxy import Accumulator, distances, lane_segments


class TestForwardMapProxy(unittest.TestCase):
    def test_x_positive_is_strict_and_fov64_means_half32(self):
        c = Accumulator()
        c.add([[0.,0.,0.,0.,4.6,1.9,1.], [33.,0.,0.,0.,4.6,1.9,1.]], None)
        self.assertEqual(c.counts['forward_fov_actors'], 0)

    def test_missing_map_is_unknown_not_success(self):
        c = Accumulator()
        c.add([[10.,0.,0.,0.,4.6,1.9,1.], [20.,0.,0.,0.,4.6,1.9,1.]], None)
        self.assertEqual(c.counts['forward_map_unknown_actors'], 2)
        self.assertIsNone(c.result()['forward_centerline_gt_1p75_rate'])

    def test_union_of_lanes_accepts_alternative_to_gt_lane(self):
        segments = lane_segments({'map': {'lane_polylines': [
            [[0.,0.], [30.,0.]], [[0.,4.], [30.,4.]]]}})
        self.assertAlmostEqual(float(distances(np.array([[10.,4.]]), segments)[0]), 0.)

    def test_threshold_exceedance_is_only_proxy(self):
        c = Accumulator()
        segments = lane_segments({'map': {'lane_polylines': [[[0.,0.], [30.,0.]]]}})
        c.add([[10.,2.,0.,0.,4.6,1.9,1.], [20.,0.,0.,0.,4.6,1.9,1.]], segments)
        self.assertEqual(c.counts['forward_centerline_gt_1.75_actors'], 1)
        self.assertEqual(c.counts['both_forward_centerline_le_3.0_records'], 1)


if __name__ == '__main__':
    unittest.main()
