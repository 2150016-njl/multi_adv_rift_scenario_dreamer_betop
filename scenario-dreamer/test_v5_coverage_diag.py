"""Read-only candidate-coverage diagnostic tests, no CARLA, GPU or labels online."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from audit_v5_candidate_coverage import process_one,positive_metrics,raw_lane_match
from high_value_init.v5_candidates import make_candidates


def example(i, weak=False, offroad=False):
    route=[[float(x),0.] for x in range(0,42,2)]
    lanes=[[[float(x),0.] for x in range(-24,39,2)],
           [[20.,float(y)] for y in range(-30,31,2)]]
    s=[[10.,.1,0.,3.],[20.1,-11.,math.pi/2.,2.]]
    if offroad:s[0][1]=9.
    labs=[{'interaction_strength':.1 if weak else .8,
           'route_distance_difference_m':1.,
           'route_distance_difference_threshold_m':6.,
           'conflict_point_local_xy':[20.,0.]} for _ in range(2)]
    return {'town':'Town12', 'scene_id':f'scene{i}',
            'condition':{'ego_state':[0.,0.,0.,3.,5.176,2.297,1.],
                         'ego_route':{'polyline':route},
                         'map':{'lane_polylines':lanes},
                         'observed_agent_states':[]},
            'target':{'generated_agent_states':s,
                      'route_interaction_supervision':labs}}

class TestCoverage(unittest.TestCase):
    def test_score_filter_is_not_map_geometry(self):
        r=example(0,weak=True)
        self.assertIsNotNone(positive_metrics(r['target']['generated_agent_states'],r['target']['route_interaction_supervision']))
        self.assertGreaterEqual(len(make_candidates(r['condition'])),2)

    def test_nearest_raw_map_support_distinct_from_route_candidate(self):
        r=example(0,offroad=True)
        d=raw_lane_match(r['condition']['map']['lane_polylines'],r['target']['generated_agent_states'][0],1.25,.38)
        self.assertGreater(d[0],1.25)

    def test_stage_audit_computes_absolute_counts(self):
        rows=[example(0),example(1,weak=True),example(2,offroad=True)]
        with tempfile.TemporaryDirectory() as work:
            p=Path(work)/'test.jsonl'
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            class Args:
                max_records_per_split=0
                min_rift_score=.4
                max_gt_lane_distance=1.25
                max_anchor_distance=12.
            report=process_one('test',p,Args)
        c=report['counts']
        self.assertEqual(c['total'],3)
        self.assertEqual(c['passed_score_gate'],2)
        self.assertEqual(c['failed_score_gate'],1)
        self.assertEqual(c['both_heading_match_raw_hdmap'],1)
        self.assertEqual(c['assign_baseline_32'],1)

if __name__=='__main__': unittest.main()
