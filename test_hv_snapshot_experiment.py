"""CPU/memory adapter tests; no CARLA server or temporary directories."""
import ast
from collections import deque
import copy
import io
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import audit_hv_snapshot_experiment as audit
from high_value_init.native_state_audit import NativeAudit, native_actor, spawn_frozen_cohort
from build_hv_snapshot_audit_manifest import pair_job
from hv_audit_inputs import manifest_jobs, snapshot_paths


def pose(i,x,y=0.,yaw=0.):
    return dict(actor_id=i,type_id='vehicle.tesla.model3',x=x,y=y,z=0.,yaw_deg=yaw,
                length=4.624,width=1.909,speed_mps=3.)


def scene():
    return dict(coordinate_frame='carla_world',scene_id='fixture',town='Town12',
                ego=pose(100,0.),agents=[pose(101,12.),pose(102,20.)],
                ego_route=[dict(x=float(x),y=0.,z=0.) for x in range(0,42,2)],
                map_lanes=[dict(centerline=[[-30.,0.],[40.,0.]])],
                rift_labels=[dict(actor_id=101,is_interactive=True,score=.8,
                                 interaction_waypoint=dict(x=25.,y=0.,z=0.)),
                             dict(actor_id=102,is_interactive=False,score=0.)])


class Vec:
    def __init__(self,x=0.,y=0.,z=0.):self.x,self.y,self.z=x,y,z
    def distance(self,b):return math.sqrt((self.x-b.x)**2+(self.y-b.y)**2+(self.z-b.z)**2)
    def make_unit_vector(self):
        n=self.distance(Vec())
        return Vec(self.x/n,self.y/n,self.z/n)
    def get_vector_angle(self,b):
        return math.acos(max(-1.,min(1.,(self.x*b.x+self.y*b.y+self.z*b.z)/(self.distance(Vec())*b.distance(Vec())))))


class Rotation:
    def __init__(self,yaw=0.,pitch=0.,roll=0.):self.yaw,self.pitch,self.roll=yaw,pitch,roll
    def get_forward_vector(self):
        a=math.radians(self.yaw)
        return Vec(math.cos(a),math.sin(a),0.)


class Transform:
    def __init__(self,location=None,rotation=None):self.location,self.rotation=location or Vec(),rotation or Rotation()
    def get_forward_vector(self):return self.rotation.get_forward_vector()


CARLA=SimpleNamespace(Location=Vec,Vector3D=Vec,Rotation=Rotation,Transform=Transform,
                      LaneType=SimpleNamespace(Driving=1))


class Waypoint:
    def __init__(self,x,y=0.,yaw=0.):
        self.transform=Transform(Vec(x,y,0.),Rotation(yaw))
        self.road_id=1;self.lane_id=-1;self.section_id=0;self.s=float(x)
        self.is_junction=False;self.lane_width=3.5


class Cache:
    def __init__(self,config):self.inter_info={}
    def update_inter_info(self,i,wp,route,ids,dist,interaction_meta=None):
        self.inter_info[i]=dict(inter_waypoint=wp,inter_CBV_route_elements=route,
                                interaction_meta=interaction_meta or {})


def native_class():
    # Execute the real native class body with CPU stand-ins for CARLA/map/GRP.
    # Dependency stand-ins test adapter/control flow, not real map semantics.
    path=Path('RIFT/rift/cbv/recognition/base_cbv.py')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseCBVRecog')
    class ForbiddenProvider:
        def __getattr__(self,name):raise AssertionError('Live provider must not be used')
    scope=dict(math=math,deque=deque,carla=CARLA,CBVBasePlanner=Cache,
               CarlaDataProvider=ForbiddenProvider(),is_within_distance_ahead=lambda *a:True)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['BaseCBVRecog'],scope['CarlaDataProvider']


CONFIG={'eval':dict(spawn_radius=50,activate_radius=150,coord_interval=60,traffic_intensity=.6,
                    max_route_len=400,max_ego_route_len=50,search_radius=60,max_interaction_fov=150,
                    max_interaction_dis=90,min_interaction_dis_diff=20,down_sample_step=1,max_agent_num=2)}


class TestSnapshot(unittest.TestCase):
    def job(self,s,selected=None,generated=None,extra=None):
        args=SimpleNamespace(blueprint='vehicle.tesla.model3',fov=64.,mode='snapshot')
        values=({},s,selected or [],generated,'vehicle.tesla.model3','hash')
        with patch.object(audit,'read_job',return_value=values):
            return audit.audit_job(dict(snapshot_path='memory.json',**(extra or {})),args)

    def test_forward_positive_pool_and_candidate_coverage(self):
        r=self.job(scene());s=audit.summarize([r],'snapshot')
        c=s['groups']['raw_pool']['counts']
        self.assertEqual(c['saved_forward_positive_actors'],1)
        self.assertEqual(c['saved_positive_candidate_supported_actors'],1)
        self.assertFalse(s['initial_candidate_evidence'])
        self.assertIsNone(r['raw_pool'][0]['native'])


    def test_native_nonfinite_observation_does_not_stop_later_records(self):
        first=TestSnapshot().job(scene())
        first['raw_pool'][0]['road_geometry']={'preview_heading_error_deg':float('nan')}
        first['raw_pool'][0]['native']={'evaluated':True,'native_candidate':True,
                                       'candidate_debug':{'rejection_counts':{}}}
        second=TestSnapshot().job(scene())
        output=io.StringIO()
        clean=audit.write_report(output,first)
        audit.write_report(output,second)
        def reject_constant(token):
            raise AssertionError('Nonstandard JSON number: '+token)
        rows=[json.loads(line,parse_constant=reject_constant) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows),2)
        self.assertIsNone(rows[0]['raw_pool'][0]['road_geometry']['preview_heading_error_deg'])
        self.assertEqual(clean['numeric_diagnostics']['nonfinite_values'],[
            {'json_pointer':'/raw_pool/0/road_geometry/preview_heading_error_deg','original_value':'NaN'}])
        self.assertTrue(clean['raw_pool'][0]['native']['native_candidate'])
        self.assertTrue(math.isnan(first['raw_pool'][0]['road_geometry']['preview_heading_error_deg']))
        summary=audit.summarize(rows,'native')
        self.assertEqual(summary['planned_jobs'],2)
        self.assertEqual(summary['numeric_diagnostics']['jobs_with_nonfinite_numbers'],1)
        self.assertFalse(summary['full_evaluation_eligible'])

    def test_infinities_nested_numpy_scalars_and_json_pointer_escaping(self):
        clean,values=audit.json_compatible({'x/y~':[np.float64('inf'),float('-inf'),np.array([float('nan'),0.])]})
        self.assertEqual(clean,{'x/y~':[None,None,[None,0.]]})
        self.assertEqual([v['original_value'] for v in values],['Infinity','-Infinity','NaN'])
        self.assertEqual(values[0]['json_pointer'],'/x~1y~0/0')
        json.dumps(clean,allow_nan=False)

    def test_finite_data_and_boolean_decisions_keep_their_values(self):
        report={'value':np.float32(1.5),'count':np.int64(2),'accepted':np.bool_(True),'missing':None,'zero':0.}
        output=io.StringIO();clean=audit.write_report(output,report)
        self.assertEqual(clean['value'],1.5)
        self.assertEqual(clean['count'],2)
        self.assertIs(clean['accepted'],True)
        self.assertEqual(clean['numeric_diagnostics']['nonfinite_count'],0)

    def test_encoder_failure_never_writes_partial_record(self):
        output=io.StringIO()
        with self.assertRaises(TypeError):audit.write_report(output,{'unsupported':object()})
        self.assertEqual(output.getvalue(),'')

    def test_missing_label_is_unknown(self):
        s=scene();s['rift_labels']=[]
        r=self.job(s)
        self.assertIsNone(r['raw_pool'][0]['saved_positive'])
        self.assertEqual(audit.summarize([r],'snapshot')['groups']['raw_pool']['counts']['saved_forward_label_unknown_actors'],2)

    def test_missing_map_does_not_certify_candidate_failure(self):
        s=scene();s['map_lanes']=[]
        r=self.job(s)
        self.assertIsNone(r['raw_pool'][0]['candidate_geometry_supported'])
        summary=audit.summarize([r],'snapshot')
        self.assertIsNone(summary['groups']['raw_pool']['saved_positive_candidate_coverage'])
        self.assertEqual(summary['candidate_inputs_missing_jobs'],1)

    def test_replacement_excludes_selected_gt_from_generated_occupancy(self):
        r=self.job(scene(),[101,102],[[12.,0.,0.,3.,4.6,1.9,1.],[20.,0.,0.,3.,4.6,1.9,1.]])
        self.assertEqual(r['comparison_context'],'matched_target_replacement')
        self.assertTrue(all(a['occupancy_2d']['clear'] for a in r['generated']))
        self.assertTrue(all(a['candidate_anchor_supported'] is None for a in r['generated']))

    def test_append_keeps_existing_actors_as_obstacles(self):
        r=self.job(scene(),[],[[12.,0.,0.,3.,4.6,1.9,1.],[25.,4.,0.,3.,4.6,1.9,1.]])
        self.assertEqual(r['generated'][0]['occupancy_2d']['context_overlap_ids'],[101])

    def test_gt_and_generated_use_same_selected_sparse_context(self):
        s=scene();s['agents'].append(pose(103,12.))
        r=self.job(s,[101,102],[[12.,0.,0.,3.,4.6,1.9,1.],[20.,0.,0.,3.,4.6,1.9,1.]],
                   extra={'context_actor_ids':[]})
        self.assertFalse(r['raw_pool'][0]['occupancy_2d']['clear'])
        self.assertTrue(r['selected_gt'][0]['occupancy_2d']['clear'])
        self.assertTrue(r['generated'][0]['occupancy_2d']['clear'])

    def test_full_physical_context_keeps_hidden_actor_without_leaking_it_into_model_input(self):
        s=scene();s['agents'].append(pose(103,28.))
        generated=[[28.,0.,0.,3.,4.6,1.9,1.],[25.,5.,0.,3.,4.6,1.9,1.]]
        values=({},s,[101,102],generated,'vehicle.tesla.model3','hash')
        args=SimpleNamespace(blueprint='vehicle.tesla.model3',fov=64.,mode='snapshot',physical_context='full_snapshot')
        with patch.object(audit,'read_job',return_value=values):
            r=audit.audit_job({'snapshot_path':'memory.json','context_actor_ids':[]},args)
        self.assertEqual(r['conditioning_context_actor_ids'],[])
        self.assertEqual(r['context_actor_ids'],[103])
        self.assertEqual(r['generated'][0]['occupancy_2d']['context_overlap_ids'],[103])
        self.assertTrue(all(a['occupancy_2d']['clear'] for a in r['selected_gt']))
        self.assertEqual(r['generated'][0]['raw_generated_local_state'],generated[0])

    def test_rotation_world_to_ego(self):
        e=pose(1,10.,20.,90.);p=pose(2,10.,30.,90.)
        self.assertAlmostEqual(audit.local_state(e,p)[0],10.)
        self.assertAlmostEqual(audit.local_state(e,p)[1],0.)

    def test_read_job_accepts_raw_world_and_rejects_local_dataset(self):
        with patch.object(Path,'read_bytes',return_value=json.dumps(scene()).encode()):
            self.assertEqual(audit.read_job({'snapshot_path':'memory'},'vehicle.tesla.model3')[1]['town'],'Town12')
        with patch.object(Path,'read_bytes',return_value=b'{"coordinate_frame":"ego_local"}'):
            with self.assertRaises(ValueError):audit.read_job({'snapshot_path':'memory'},'vehicle.tesla.model3')


class TestIsolatedNative(unittest.TestCase):
    def run_actor(self,actor,existing=()):
        cls,provider=native_class()
        world_map=SimpleNamespace(get_waypoint=lambda loc,**k:Waypoint(loc.x,loc.y,90.))
        planner=SimpleNamespace(trace_route=lambda origin,destination:([(Waypoint(destination.x,destination.y),None)],{},10.))
        report=native_actor(cls,Cache,CARLA,world_map,planner,CONFIG,'eval',pose(100,0.),actor,
                            [Waypoint(0.),Waypoint(10.),Waypoint(20.)],existing)
        self.assertIs(cls.get_CBV_candidates.__globals__['CarlaDataProvider'],provider)
        return report

    def test_original_native_algorithm_uses_isolated_provider(self):
        result=self.run_actor(pose(101,10.,-10.,90.))
        self.assertTrue(result['native_candidate'])
        self.assertEqual(result['interaction_meta']['distance_diff'],0.)
        self.assertTrue(result['target_route_to_interaction'])

    def test_distance_rejection_comes_from_native_code(self):
        result=self.run_actor(pose(101,3.))
        self.assertFalse(result['native_candidate'])
        self.assertEqual(result['candidate_debug']['rejection_counts']['distance'],1)

    def test_capacity_rejection_comes_from_native_code(self):
        result=self.run_actor(pose(101,15.),existing=[8,9])
        self.assertFalse(result['native_candidate'])
        self.assertEqual(result['candidate_debug']['rejection_counts']['max_agent_num_reached'],1)


class TestStaticSpawn(unittest.TestCase):
    def test_single_try_cleanup_and_original_gt_velocity(self):
        created=[];calls=[]
        class Actor:
            def __init__(self,i):
                self.id=i;self.type_id='vehicle.tesla.model3';self.destroyed=False
                self.bounding_box=SimpleNamespace(extent=Vec(2.3,.95,.7),location=Vec())
            def set_simulate_physics(self,value):self.physics=value
            def set_target_velocity(self,v):self.v=v
            def destroy(self):self.destroyed=True;return True
        class World:
            def get_blueprint_library(self):return SimpleNamespace(find=lambda name:SimpleNamespace(id=name))
            def try_spawn_actor(self,bp,t):
                calls.append(t)
                if len(calls)==3:return None
                a=Actor(len(calls));created.append(a);return a
            def tick(self):raise AssertionError('No ticks allowed')
        e=pose(100,0.);e.update(vx=1.,vy=2.,vz=3.)
        r=spawn_frozen_cohort(World(),CARLA,e,[],[pose(101,12.),pose(102,20.)])
        self.assertEqual(len(calls),3)
        self.assertFalse(r['targets_all_accepted'])
        self.assertTrue(r['context_complete'])
        self.assertTrue(all(a.destroyed for a in created))
        self.assertEqual(r['attempts'][0]['requested_velocity'],dict(x=1.,y=2.,z=3.))
        self.assertEqual(r['world_ticks_requested'],0)
        self.assertFalse(r['speed_execution_verified'])
        self.assertTrue(all(a.get('raw_request_fidelity') for a in r['attempts']))


class TestManifestAndPrecision(unittest.TestCase):
    def fixture(self):
        s=scene();s['agents'].append(pose(103,28.))
        condition=audit.condition_from_snapshot(s)
        condition.update(ego_state=audit.local_state(s['ego'],s['ego']),
                         observed_agent_states=[audit.local_state(s['ego'],s['agents'][2])])
        record={'scene_id':s['scene_id'],'town':s['town'],'condition':condition,
                'target':{'generated_agent_states':[audit.local_state(s['ego'],p) for p in s['agents'][:2]]}}
        sample={'scene_id':s['scene_id'],'generated':{'generated_agent_states':[[13.,0.,0.,3.,4.6,1.9,1.],[22.,0.,0.,3.,4.6,1.9,1.]]}}
        return s,record,sample

    def test_manifest_matches_states_and_context(self):
        s,r,g=self.fixture()
        j=pair_job('memory.json',json.dumps(s).encode(),r,g)
        self.assertEqual(j['target_actor_ids'],[101,102])
        self.assertEqual(j['context_actor_ids'],[103])
        self.assertEqual(j['generated_agent_states'],g['generated']['generated_agent_states'])

    def test_identical_scene_id_is_insufficient(self):
        s,r,g=self.fixture();s['agents'][0]['x']+=1.
        with self.assertRaises(ValueError):pair_job('memory.json',json.dumps(s).encode(),r,g)

    def test_ambiguous_state_match_is_rejected(self):
        s,r,g=self.fixture();s['agents'].append(pose(104,12.))
        with self.assertRaises(ValueError):pair_job('memory.json',json.dumps(s).encode(),r,g)

    def test_saved_spawn_rounding_is_accepted_but_rewriting_is_not(self):
        e=pose(100,2140.12345,-1420.98765,73.1297)
        a=math.radians(e['yaw_deg']);state=[12.,4.,.1,3.,4.6,1.9,1.]
        f32=lambda v:struct.unpack('f',struct.pack('f',v))[0]
        saved={'x':f32(e['x']+math.cos(a)*12.-math.sin(a)*4.),
               'y':f32(e['y']+math.sin(a)*12.+math.cos(a)*4.),
               'yaw_deg':f32(e['yaw_deg']+math.degrees(.1)),'z':.3}
        engine=object.__new__(NativeAudit)
        p=engine.generated_pose(e,state,-1,'vehicle.tesla.model3',saved)
        self.assertEqual(p['z_source'],'saved_spawn_request')
        saved['x']+=.1
        with self.assertRaises(ValueError):engine.generated_pose(e,state,-1,'vehicle.tesla.model3',saved)


class InputEvidenceTests(unittest.TestCase):
    def test_missing_collection_fails_before_empty_jobs(self):
        with patch.object(Path,'is_dir',return_value=False):
            with self.assertRaisesRegex(FileNotFoundError,'summary.json'):
                snapshot_paths('missing_collection')

    def test_symlink_worker_walk_and_cycle_are_handled(self):
        rows=[('collection',['worker'],[]),('collection/worker',['loop'],['s.json']),
              ('collection/worker/loop',[],['duplicate.json'])]
        def resolved(path):
            return Path('real_worker' if 'worker' in str(path) else 'real_root')
        with patch.object(Path,'is_dir',return_value=True), \
                patch.object(Path,'resolve',resolved), patch('hv_audit_inputs.os.walk',return_value=rows) as walk:
            self.assertEqual(snapshot_paths('collection'),[Path('collection/worker/s.json')])
            self.assertTrue(walk.call_args.kwargs['followlinks'])

    def test_failed_pairing_rejects_even_nonempty_manifest(self):
        with patch.object(Path,'read_bytes',return_value=b'{"snapshot_path":"x"}\n'), \
                patch.object(Path,'exists',return_value=True), \
                patch.object(Path,'read_text',return_value='{"passed":false}'):
            with self.assertRaisesRegex(ValueError,'Pairing failed'):
                manifest_jobs('pairs.jsonl')

    def test_changed_manifest_rejected_against_evidence(self):
        with patch.object(Path,'read_bytes',return_value=b'{"snapshot_path":"x"}\n'), \
                patch.object(Path,'exists',return_value=True), \
                patch.object(Path,'read_text',return_value='{"passed":true,"manifest_sha256":"wrong"}'):
            with self.assertRaisesRegex(ValueError,'hash differs'):
                manifest_jobs('pairs.jsonl')

    def test_partial_manifest_cannot_run(self):
        with self.assertRaisesRegex(ValueError,'Partial pairing'):
            manifest_jobs('pairs.jsonl.partial.jsonl')


if __name__=='__main__':unittest.main()
