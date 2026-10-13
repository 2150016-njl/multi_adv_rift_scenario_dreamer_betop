"""Tests for no false spawn-fidelity alarms or misleading green rate on crash."""
import importlib.util
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

PATH=Path(__file__).resolve().parent/'audit_hv_one_shot_results.py'
spec=importlib.util.spec_from_file_location('audit_one_shot_under_test',PATH)
audit_mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_mod)

class TestRuntimeAudit(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)
        self.states=[[2.0,1.0,0.25,4.0,4.5,1.8,1.0],
                     [3.0,-1.0,-0.3,7.0,4.5,1.8,1.0]]
        self.record={
          'scene':{'ego':{'x':50.0,'y':40.0,'yaw_deg':90.0}},
          'proposal':{'generated_agent_states':self.states},
          'result':{
             'scene_id':'sceneA','success':True,'generator_called':True,
             'rift_audit_status':'deferred_to_normal_cbvs_recog',
             'generation_max_attempts':1,
             'generation_attempts':[{'generation_attempt_index':0,'success':True}],
             'spawned_actor_ids':[101,102],
             'spawn_attempts':[]
          }
        }
        for i,state in enumerate(self.states):
            yaw=math.radians(90.0)
            x=50+math.cos(yaw)*state[0]-math.sin(yaw)*state[1]
            y=40+math.sin(yaw)*state[0]+math.cos(yaw)*state[1]
            self.record['result']['spawn_attempts'].append({
              'index':i,'generated_local_state':state,
              'raw_world_transform':{'x':x,'y':y,'yaw_deg':90+math.degrees(state[2])},
              'spawn_transform':{'x':x,'y':y,'yaw_deg':90+math.degrees(state[2])},
              'spawn_acceptance_observation_phase':'pre_tick_unconfirmed',
              'spawn_acceptance_xy_error_m':48,
              'spawn_acceptance_yaw_error_deg':90,
            })
        self.flush()
    def tearDown(self):self.tmp.cleanup()
    def flush(self):
        (self.path/'sceneA_frame0_online_generation.json').write_text(json.dumps(self.record))
    def test_aborted_run_has_unknown_green_rate_and_warnings_not_errors(self):
        summary=audit_mod.audit(self.path, simulator_exit_code=1)
        self.assertEqual(summary['contract_errors'],[])
        self.assertEqual(len(summary['diagnostic_warnings']),2)
        self.assertIsNone(summary['native_green_rate_per_eligible_request'])
        self.assertFalse(summary['simulator_completed'])
        self.assertEqual(summary['num_both_spawned'],1)
    def test_completed_run_has_measured_zero_green_rate(self):
        summary=audit_mod.audit(self.path,simulator_exit_code=0)
        self.assertEqual(summary['native_green_rate_per_eligible_request'],0.0)
        self.assertTrue(summary['simulator_completed'])
    def test_empty_dump_cannot_pass(self):
        (self.path/'sceneA_frame0_online_generation.json').unlink()
        summary=audit_mod.audit(self.path,simulator_exit_code=0)
        self.assertTrue(any('No online generation' in e for e in summary['contract_errors']))
    def test_raw_request_changed_is_still_hard_error(self):
        self.record['result']['spawn_attempts'][0]['raw_world_transform']['x']+=0.3
        self.flush()
        summary=audit_mod.audit(self.path,simulator_exit_code=0)
        self.assertTrue(any('rewritten' in x for x in summary['contract_errors']))
    def test_spawn_command_rewrite_is_hard_error(self):
        self.record['result']['spawn_attempts'][0]['spawn_transform']['x']+=0.3
        self.flush()
        summary=audit_mod.audit(self.path,simulator_exit_code=0)
        self.assertTrue(any('spawn command' in x for x in summary['contract_errors']))
    def test_float32_carla_rounding_is_not_state_rewriting(self):
        # Simulate CARLA's actual Python binding: constructor rounds to float32.
        self.record['scene']['ego']['x']=2140.12345
        self.record['scene']['ego']['y']=-1420.98765
        self.record['scene']['ego']['yaw_deg']=73.1297
        yaw=math.radians(self.record['scene']['ego']['yaw_deg'])
        for i,state in enumerate(self.states):
            world_x=self.record['scene']['ego']['x']+math.cos(yaw)*state[0]-math.sin(yaw)*state[1]
            world_y=self.record['scene']['ego']['y']+math.sin(yaw)*state[0]+math.cos(yaw)*state[1]
            world_yaw=math.degrees(yaw+state[2])
            row=self.record['result']['spawn_attempts'][i]
            row['raw_world_transform']={'x':world_x,'y':world_y,'yaw_deg':world_yaw}
            row['spawn_transform']={
                'x':struct.unpack('<f',struct.pack('<f',world_x))[0],
                'y':struct.unpack('<f',struct.pack('<f',world_y))[0],
                'yaw_deg':struct.unpack('<f',struct.pack('<f',world_yaw))[0]}
        self.flush()
        summary=audit_mod.audit(self.path,simulator_exit_code=1)
        self.assertEqual(summary['contract_errors'], [])
        self.assertIsNone(summary['native_green_rate_per_eligible_request'])

    def test_float32_rounding_does_not_hide_real_spawn_command_rewrite(self):
        self.test_float32_carla_rounding_is_not_state_rewriting()
        self.record['result']['spawn_attempts'][0]['spawn_transform']['x']+=0.002
        self.flush()
        summary=audit_mod.audit(self.path,simulator_exit_code=1)
        self.assertTrue(any('spawn command' in e for e in summary['contract_errors']))

    def test_native_activation_must_match_scene_id(self):
        (self.path/'lifecycle.jsonl').write_text(json.dumps(
            {'event':'rift_cbv_activated','actor_id':102,'scene_id':'otherScene'})+'\n')
        summary=audit_mod.audit(self.path,simulator_exit_code=0)
        self.assertEqual(summary['num_at_least_one_native_green'],0)
    def test_native_activation_counted_for_spawned_actor_only(self):
        (self.path/'lifecycle.jsonl').write_text(json.dumps({'event':'rift_cbv_activated','actor_id':102,'scene_id':'sceneA'})+'\n')
        summary=audit_mod.audit(self.path,simulator_exit_code=0)
        self.assertEqual(summary['num_at_least_one_native_green'],1)
        self.assertEqual(summary['native_green_rate_per_eligible_request'],1.0)

if __name__=='__main__':unittest.main()
