"""CPU-only geometry, representation, training-gradient and S1 contract tests."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
import sys

import torch

from high_value_init.v5_candidates import make_candidates, assign_pair, closest, signature
from high_value_init.v5_model import CARLAJointRawGenerator, candidate_tensors
from eval_high_value_scene_generator_v5 import oracle,raw_metrics


def scene():
    # Direct CARLA ego-local coordinates: two directed approach lanes
    route=[[float(x),0.] for x in range(0,41,2)]
    lane0=[[float(x),0.] for x in range(-24,39,2)]
    lane1=[[20.,float(y)] for y in range(-30,31,2)]
    return {'ego_state':[0.,0.,0.,3.,5.176,2.297,1.],
            'ego_route':{'polyline':route},
            'map':{'lane_polylines':[lane0,lane1]},
            'observed_agent_states':[],'num_generate_agents':2}


def assigned_record():
    c=scene(); candidates=make_candidates(c)
    targets=[[10.,0.1,0.,3.], [20.1,-11.,math.pi/2.,2.]]
    anchors=[{'conflict_point_local_xy':[20.,0.]},
             {'conflict_point_local_xy':[20.,0.]}]
    a=assign_pair(candidates,targets,anchors,max_pos_m=1.25,max_heading_rad=.38,max_anchor_m=12.)
    if a is None:
        raise AssertionError('Fixture targets cannot be represented by candidate chart')
    return {'condition':c,'candidates':candidates,'targets':targets,
            'assigned_indices':[x[0] for x in a],
            'scene_id':'fixture','episode_key':'fixture_one','town':'Town12'}


class TestV5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_candidate_both_approaches(self):
        c=make_candidates(scene())
        self.assertGreaterEqual(len(c),2)
        self.assertEqual({0,1}, {x['lane_id'] for x in c})
        self.assertTrue(all(x['anchor_s']>3. for x in c))

    def test_candidates_never_inspect_labels_or_targets(self):
        cond=scene();a=make_candidates(cond)
        cond['target']={'generated_agent_states':[[100,100]]}
        cond['rift_labels']=[{'is_interactive':True,'score':10}]
        self.assertEqual(a,make_candidates(cond))

    def test_representation_oracle_raw(self):
        r=assigned_record(); model=CARLAJointRawGenerator(hidden=24)
        o=oracle(model,r)
        self.assertLess(o['position_max_m'],.03)
        self.assertLess(o['heading_max_rad'],.01)
        self.assertLess(o['speed_max_mps'],.02)

    def test_joint_training_gradient(self):
        r=assigned_record();model=CARLAJointRawGenerator(hidden=24)
        l=model.training_losses(r)
        self.assertTrue(bool(torch.isfinite(l['loss'])))
        l['loss'].backward()
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in model.parameters() if x.grad is not None),0.)

    def test_one_joint_sample_two_states_raw_finite(self):
        r=assigned_record();model=CARLAJointRawGenerator(hidden=24)
        torch.manual_seed(40)
        with torch.no_grad():o=model.sample_once(r['condition'],r['candidates'])
        self.assertEqual(len(o['generated_agent_states']),2)
        self.assertTrue(all(len(s)==7 and 0<s[3]<12 for s in o['generated_agent_states']))
        self.assertEqual(o['raw_state_postprocessing'],'none')
        self.assertEqual(o['sampler'],'one_joint_categorical_one_8d_gaussian')
        self.assertEqual(len(o['candidate_indices']),2)
        self.assertIn('native_rift_conflict_checked',raw_metrics(r,o))

    def test_reject_nonrepresentable_training_labels(self):
        r=assigned_record()
        targets=[[10.,20.,0.,3.],[20.,-11.,math.pi/2.,2.]]
        a=assign_pair(r['candidates'],targets,max_pos_m=1.25)
        self.assertIsNone(a)

    def test_evaluating_does_not_modify_condition(self):
        r=assigned_record();original=copy.deepcopy(r)
        m=CARLAJointRawGenerator(hidden=24)
        with torch.no_grad():sample=m.sample_once(r['condition'],r['candidates'])
        self.assertEqual(original,r)
        self.assertEqual(len(sample['generated_agent_states']),2)

    def test_actual_online_wrapper_contract_without_carla(self):
        import importlib
        import sys
        import types
        fake=types.ModuleType('high_value_init.scene_generator')
        fake._snapshot_to_record=lambda snapshot, **k: {'condition':scene()}
        prior=sys.modules.get('high_value_init.scene_generator')
        sys.modules['high_value_init.scene_generator']=fake
        try:
            sys.modules.pop('high_value_init.scene_generator_v5',None)
            v5=importlib.import_module('high_value_init.scene_generator_v5')
            with tempfile.TemporaryDirectory() as d:
                ck=Path(d)/'best.ckpt'
                model=CARLAJointRawGenerator(hidden=24)
                torch.save({'schema_version':'hv_v5_raw_checkpoint_v1',
                            'config':{'hidden':24,'map_mode':'right_handed'},
                            'model':model.state_dict()},ck)
                config={'hv_num_generate_agents':2,'hv_sd_fov':64,
                        'hv_generation_mode':'full_fov_inpainting',
                        'hv_snapshot_map_coordinate_mode':'right_handed'}
                w=v5.ScenarioDreamerHighValueSceneGeneratorV5(str(ck),device='cpu',config=config)
                snap={'scene_id':'dummy','diagnostics':{'map_coordinate_mode':'right_handed'},
                      'rift_labels':[{'actor_id':99,'is_interactive':True,'score':999}]}
                torch.manual_seed(101)
                out=w.generate(snap)
                self.assertEqual(out['coordinate_frame'],'ego_local_rh_x_forward')
                self.assertFalse(out['rift_labels_used_as_input'])
                self.assertEqual(len(out['generated_agent_states']),2)
                self.assertEqual(out['sampler'],'one_joint_categorical_one_8d_gaussian')
                snap['rift_labels']=[{'actor_id':30000,'is_interactive':False}]
                torch.manual_seed(101)
                out_again=w.generate(snap)
                self.assertEqual(out['generated_agent_states'],out_again['generated_agent_states'])
                with self.assertRaises(ValueError):
                    v5.ScenarioDreamerHighValueSceneGeneratorV5(str(ck),device='cpu',
                     config=dict(config,hv_snapshot_map_coordinate_mode='legacy'))
        finally:
            sys.modules.pop('high_value_init.scene_generator_v5',None)
            if prior is None:sys.modules.pop('high_value_init.scene_generator',None)
            else:sys.modules['high_value_init.scene_generator']=prior

    def test_hash_is_not_filename(self):
        a={'town':'Town12','condition':scene(),'target':{'generated_agent_states':[[10,0,0,1],[20,0,0,1]]},'scene_id':'one'}
        b=copy.deepcopy(a);b['scene_id']='two'
        self.assertEqual(signature(a),signature(b))

if __name__=='__main__':unittest.main()
