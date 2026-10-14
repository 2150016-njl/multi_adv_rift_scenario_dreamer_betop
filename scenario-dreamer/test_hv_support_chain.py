"""Support counterexamples: no simulator, checkpoint or temporary files."""
import copy
import unittest

import torch
from audit_hv_support_chain import analyze_record
from high_value_init.v5_model import context_features
from test_high_value_v5 import assigned_record


class TestSupportChain(unittest.TestCase):
    def test_known_chart_targets_are_representable(self):
        r = analyze_record(assigned_record())
        self.assertTrue(all(a['chart_roundtrip'] for a in r['actors']))
        self.assertFalse(r['native_anchor_available'])

    def test_at_least_one_is_not_both(self):
        r = assigned_record()
        r['targets'][1] = [30., 25., 0., 2.]
        flags = analyze_record(r)['actors']
        self.assertEqual(sum(a['chart_roundtrip'] for a in flags), 1)

    def test_map_valid_is_not_candidate_support(self):
        r = assigned_record()
        # Parallel lane 14m away: legal directed map geometry, no local
        # route-contact anchor <=5m. No invented connector or RIFT label.
        r['condition']['map']['lane_polylines'].append([[-30., 14.], [30., 14.]])
        r['targets'][0] = [12., 14., 0., 3.]
        a = analyze_record(r)['actors'][0]
        self.assertTrue(a['raw_map'])
        self.assertFalse(a['candidate_geometry'])

    def test_chart_rejects_speed_even_if_geometry_matches(self):
        r = assigned_record()
        r['targets'][0][3] = 13.
        a = analyze_record(r)['actors'][0]
        self.assertTrue(a['candidate_anchor'])
        self.assertFalse(a['chart_roundtrip'])

    def test_current_context_encoder_loses_obstacle_arrangement(self):
        a = assigned_record()['condition']
        b = copy.deepcopy(a)
        a['observed_agent_states'] = [[15.,0.,0.,3.,4.6,1.9], [-15.,0.,0.,3.,4.6,1.9]]
        b['observed_agent_states'] = [[0.,15.,0.,3.,4.6,1.9], [0.,-15.,0.,3.,4.6,1.9]]
        self.assertTrue(torch.equal(context_features(a), context_features(b)))
        # This test documents an information-loss counterexample, not a fix.


if __name__ == '__main__':
    unittest.main()
