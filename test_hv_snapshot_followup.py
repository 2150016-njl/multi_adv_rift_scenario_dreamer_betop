"""Regression checks for sparse pairing, 3D geometry and observation-only probes."""
import ast
import copy
from contextlib import nullcontext
import json
import math
import os
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
import unittest
from xml.etree import ElementTree
from unittest.mock import patch
import numpy as np

import audit_hv_snapshot_experiment as audit
from build_hv_snapshot_audit_manifest import base_scene_id,pair_job,replay_fingerprint,select_records
from high_value_init.audit_physics_geometry import boxes_intersect,whole_box_road
from high_value_init.native_state_audit import probe_cohort
from high_value_init.v6_conditions import make_condition
from test_hv_snapshot_experiment import CARLA,Vec,Transform,pose,scene
import test_hv_snapshot_experiment as fixtures


def method(path,classname,name,scope):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==classname)
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),scope)
    return scope[name]


def box(center,extent=(2.,1.,.7),yaw=0.):
    a=math.radians(yaw);axes=np.array([[math.cos(a),-math.sin(a),0.],
                                     [math.sin(a),math.cos(a),0.],[0.,0.,1.]])
    return dict(center=list(center),axes=axes.tolist(),extent=list(extent),
                vertices=[(np.array(center)+axes@(np.array(extent)*np.array([x,y,z]))).tolist()
                          for x in (-1,1) for y in (-1,1) for z in (-1,1)])


class PairRegression(unittest.TestCase):
    def test_real_sparse_id_contract_keeps_raw_scene_and_sparse_context(self):
        s,r,g=fixtures.TestManifestAndPrecision().fixture()
        r['scene_id']+=':sparse3';g['scene_id']=r['scene_id']
        r['context_sampling']={'sparse_context_copy_index':3}
        r['condition']['observed_agent_states']=[]
        job=pair_job('memory.json',json.dumps(s).encode(),r,g)
        self.assertEqual(base_scene_id(r),s['scene_id'])
        self.assertEqual(job['context_actor_ids'],[])
        self.assertEqual(job['dataset_scene_id'],r['scene_id'])

    def test_sparse_suffix_without_metadata_is_not_silently_stripped(self):
        self.assertEqual(base_scene_id({'scene_id':'fixture:sparse1'}),'fixture:sparse1')

    def test_equivalence_ignores_ids_but_not_world_state_or_labels(self):
        a=scene();b=copy.deepcopy(a);b['agents'][0]['actor_id']=901
        b['rift_labels'][0]['actor_id']=901
        self.assertEqual(replay_fingerprint(a),replay_fingerprint(b))
        b['agents'][0]['z']+=.1
        self.assertNotEqual(replay_fingerprint(a),replay_fingerprint(b))
        b=copy.deepcopy(a);b['rift_labels'][0]['is_interactive']=False
        self.assertNotEqual(replay_fingerprint(a),replay_fingerprint(b))

    def test_builder_sparse_order_metadata_is_checked(self):
        s,r,g=fixtures.TestManifestAndPrecision().fixture()
        r['rift_audit']={'removed_positive_local_ids':[0,1],
                        'observed_agent_count_before_context_sampling':1}
        r['context_sampling']={'kept_observed_indices':[0]}
        pair_job('memory',json.dumps(s).encode(),r,g)
        r['context_sampling']['kept_observed_indices']=[1]
        with self.assertRaisesRegex(ValueError,'sparse context indices'):
            pair_job('memory',json.dumps(s).encode(),r,g)

    def test_recorded_source_sha_prevents_equivalent_wrong_provenance(self):
        s,r,g=fixtures.TestManifestAndPrecision().fixture()
        r['source_snapshot']={'sha256':'wrong'}
        with self.assertRaisesRegex(ValueError,'source snapshot SHA'):
            pair_job('memory',json.dumps(s).encode(),r,g)

    def test_stratified_selection_deduplicates_sparse_variants_and_preserves_indices(self):
        records=[]
        for town in ('Town12','Town15'):
            for f in range(4):
                base=dict(scene_id='s_frame%d'%f,town=town,episode_key=town,
                          condition={'observed_agent_states':[]})
                records.append(base)
                records.append(dict(base,scene_id=base['scene_id']+':sparse0',
                                    context_sampling={'sparse_context_copy_index':0}))
        with patch('build_hv_snapshot_audit_manifest.jsonl',return_value=iter(records)):
            selected=select_records('memory',6,'stratified',0)
        self.assertEqual(len(selected),6)
        self.assertEqual({r['town'] for _,r in selected[:2]},{'Town12','Town15'})
        self.assertEqual(len({(r['town'],r['scene_id']) for _,r in selected}),6)
        self.assertTrue(all(i%2==0 for i,_ in selected))


class GeometryRegression(unittest.TestCase):
    def test_missing_job_prevents_false_complete_native_scene_rate(self):
        r=fixtures.TestSnapshot().job(scene())
        for a in r['raw_pool']:a['native']={'evaluated':True,'native_candidate':True}
        s=audit.summarize([r,{'errors':['failed job']}],'native')
        self.assertIsNone(s['groups']['raw_pool']['at_least_one_front_native_scene_rate'])
        self.assertEqual(s['groups']['raw_pool']['missing_cohort_scene_count'],1)
    def test_overpass_boxes_do_not_overlap_in_3d(self):
        self.assertFalse(boxes_intersect(box((0,0,0)),box((0,0,5))))
        self.assertTrue(boxes_intersect(box((0,0,0)),box((2,0,0),yaw=45)))

    def test_rotated_boxes_can_separate(self):
        self.assertFalse(boxes_intersect(box((0,0,0),yaw=30),box((10,10,0),yaw=-30)))

    def test_whole_footprint_detects_overhang_and_wrong_elevation(self):
        points=[dict(x=x,y=0.,z=0.,yaw_deg=0.,lane_width_m=3.5) for x in (-10.,10.)]
        segments=[{'points':points}]
        self.assertTrue(whole_box_road(box((0,0,1)),segments)['whole_box_in_sampled_driving_polygon'])
        self.assertFalse(whole_box_road(box((0,1.4,1)),segments)['whole_box_in_sampled_driving_polygon'])
        self.assertFalse(whole_box_road(box((0,0,8)),segments)['whole_box_in_sampled_driving_polygon'])

    def test_polygon_hole_is_not_certified_by_corner_checks(self):
        def segment(y,w):return {'points':[dict(x=x,y=y,z=0.,yaw_deg=0.,lane_width_m=w) for x in (-10.,10.)]}
        # All four corners lie on driving strips, but a median runs through box.
        r=whole_box_road(box((0,0,1)),[segment(-1.,.5),segment(1.,.5)])
        self.assertFalse(r['whole_box_in_sampled_driving_polygon'])
        self.assertGreater(r['outside_area_m2'],0.)


class ConditionRegression(unittest.TestCase):
    def lanes(self):
        return [dict(lane_uid='a',predecessors=[],successors=['b'],centerline=[[10,4,0],[15,0,0]],width_m=[3.,4.]),
                dict(lane_uid='b',predecessors=['a'],successors=[],centerline=[[15,0,0],[30,0,0]],width_m=[4.,4.])]

    def test_directed_approach_and_actor_tokens_are_preserved(self):
        s=scene()
        # Use real road/section/lane UID convention for the route-root lane.
        lanes=self.lanes();lanes[1]['lane_uid']='1:0:-1';lanes[0]['successors']=['1:0:-1']
        for p in s['ego_route']:p.update(road_id=1,section_id=0,lane_id=-1)
        c=make_condition(s,lanes,s['agents'])
        self.assertEqual(c['approaches'][0]['directed_lane_path_to_ego_route'],['1:0:-1'])
        approach=next(a for a in c['approaches'] if a['lane_uid']=='a')
        self.assertEqual(approach['directed_lane_path_to_ego_route'],['a','1:0:-1'])
        self.assertEqual(len(c['actor_tokens']),2)
        self.assertEqual(c['map']['lanes_v6'][0]['width_m'],[3.,4.])
        self.assertEqual(c['output_domain']['x_min_exclusive'],0.)
        self.assertTrue(all(p[0]>0 for a in c['approaches'] for p in a['forward_visible_points']))

    def test_no_route_identity_remains_unknown_not_infeasible(self):
        c=make_condition(scene(),self.lanes(),[])
        self.assertFalse(c['topology_route_roots_known'])
        self.assertIn('NOT_proven_infeasible',c['feasibility_status'])

    def test_training_edges_follow_nuplan_receiver_convention(self):
        edges=lambda n:np.array([[i,j] for i in range(n) for j in range(n)]).T
        scope={'np':np,'get_edge_index_complete_graph':lambda n:SimpleNamespace(t=lambda:SimpleNamespace(tolist=lambda:edges(n).T.tolist())),
               'get_lane_connection_type_onehot_nuplan':lambda name:np.eye(4)[{'none':0,'pred':1,'succ':2,'self':3}[name]]}
        fn=method('scenario-dreamer/high_value_init/rift_dataset.py','HighValueSceneGraphBuilder','_lane_connections',scope)
        relation=np.argmax(fn(None,2,self.lanes()),axis=1).tolist()
        self.assertEqual(relation,[3,1,2,3])

    def test_lane_pruning_retains_ids_and_remaps_edges(self):
        scope={'np':np,'Dict':dict,'resample_polyline':lambda a,n:a,
               'get_lane_type_onehot_nuplan':lambda name:np.array([1.,0.,0.]),
               'get_edge_index_complete_graph':lambda n:SimpleNamespace(t=lambda:SimpleNamespace(
                   tolist=lambda:[[i,j] for i in range(n) for j in range(n)])),
               'get_lane_connection_type_onehot_nuplan':lambda name:np.eye(4)[{'none':0,'pred':1,'succ':2,'self':3}[name]]}
        path='scenario-dreamer/high_value_init/rift_dataset.py';cls='HighValueSceneGraphBuilder'
        connections=method(path,cls,'_lane_connections',scope)
        arrays=method(path,cls,'_lane_arrays',scope)
        rows=[dict(lane_uid='far',successors=['middle'],predecessors=[],centerline=[[20,0],[21,0]]),
              dict(lane_uid='middle',successors=['near'],predecessors=['far'],centerline=[[10,0],[11,0]]),
              dict(lane_uid='near',successors=[],predecessors=['middle'],centerline=[[1,0],[2,0]])]
        obj=SimpleNamespace(num_points_per_lane=2,max_num_lanes=2,_lane_inside_fov=lambda lane:True,
                            _lane_connections=lambda n,metadata:connections(None,n,metadata))
        lanes,_,edges=arrays(obj,{'map':{'lane_polylines':[r['centerline'] for r in rows],'lanes_v6':rows}})
        self.assertEqual(lanes[:,0,0].tolist(),[1.,10.])
        self.assertEqual(np.argmax(edges,axis=1).tolist(),[3,2,1,3])


class ProbeRegression(unittest.TestCase):
    def test_ticks_measure_velocity_and_collision_without_state_correction(self):
        actor=SimpleNamespace(id=11)
        sensors=[]
        class Sensor:
            def listen(self,callback):self.callback=callback
            def stop(self):self.stopped=True
            def destroy(self):self.destroyed=True;return True
        class World:
            frame=10
            def get_snapshot(self):
                t=Transform(Vec((self.frame-10)*.15,0,0))
                state=SimpleNamespace(get_transform=lambda:t,get_velocity=lambda:Vec(3.,0.,0.))
                return SimpleNamespace(frame=self.frame,find=lambda aid:state)
            def get_blueprint_library(self):return SimpleNamespace(find=lambda name:name)
            def spawn_actor(self,*args,**kw):
                sensor=Sensor();sensors.append(sensor);return sensor
            def tick(self):
                self.frame+=1
                if self.frame==11:sensors[0].callback(SimpleNamespace(frame=11,normal_impulse=Vec(1,0,0),
                                              other_actor=SimpleNamespace(id=99,type_id='static.wall')))
                return self.frame
        s=scene();p=s['agents'][0]
        attempts=[dict(source_actor_id=p['actor_id'],requested_pose=p,requested_velocity={'x':3.,'y':0.,'z':0.})]
        with patch('time.sleep'):
            r=probe_cohort(World(),CARLA,{p['actor_id']:(actor,Transform())},attempts,3,None,s)
        self.assertEqual(r['ticks_observed'],3)
        self.assertEqual(r['measurements'][0]['first_tick_velocity_error_mps'],0.)
        self.assertEqual(r['measurements'][0]['collision_event_count'],1)
        self.assertTrue(all(sensor.destroyed for sensor in sensors))
        self.assertFalse(r['post_spawn_corrections'])

    def test_sensor_failure_returns_unknown_and_cleans_owned_sensor(self):
        class BadWorld:
            def get_snapshot(self):return SimpleNamespace(frame=10)
            def get_blueprint_library(self):raise RuntimeError('sensor unavailable')
        with patch('time.sleep'):
            r=probe_cohort(BadWorld(),CARLA,{},[],3,None,scene())
        self.assertTrue(r['errors']);self.assertEqual(r['ticks_observed'],0)


class TelemetryRegression(unittest.TestCase):
    def test_normal_gate_does_not_call_recognizer_and_selection_calls_once(self):
        scope={}
        fn=method('RIFT/rift/gym_carla/envs/carla_env.py','CarlaEnv','CBVs_recog',scope)
        calls=[];events=[];activated=[]
        selected=[SimpleNamespace(id=11)]
        policy=SimpleNamespace(max_agent_num=2,get_CBVs=lambda *a:(calls.append(a) or selected))
        obj=SimpleNamespace(high_value_initializer=None,high_value_generation_initializer=None,
            cbv_policy_type='attention',CBVs={},cbv_recog_policy=policy,time_step=24,
            ego_vehicle=object(),local_route_waypoints=[],rest_route_waypoints=[],red_light_state=False,
            _record_hv_recognition_step=lambda reason,actors:events.append((reason,actors)),
            _activate_CBV=lambda a:activated.append(a))
        fn(obj);self.assertFalse(calls);self.assertEqual(events[-1][0],'warmup_le_25')
        obj.time_step=26;fn(obj)
        self.assertEqual(len(calls),1);self.assertEqual(activated,selected)
        self.assertIsNone(events[-1][0])

    def test_attention_observation_reuses_single_existing_inference(self):
        calls=[];a=SimpleNamespace(id=11);b=SimpleNamespace(id=12)
        scope={'torch':SimpleNamespace(no_grad=lambda:nullcontext()),
            'get_bev_boxes':lambda *a:[],
            'get_input_batch':lambda *a:(([],[],None,None,False),[11,12],[a,b]),
            'get_attn_norm_vehicles':lambda *a:np.array([.2,.8])}
        fn=method('RIFT/rift/cbv/recognition/attention_based/attn_model.py','AttnModel','get_CBVs',scope)
        obj=SimpleNamespace(net=lambda *a,**kw:(calls.append(1) or (None,[])),config={'attention_score':'AllLayer'},
                            viz_attn_map=False,get_most_relevant_vehicles=lambda *a,**kw:[b])
        self.assertEqual(fn(obj,object(),[a,b],[Vec(),Vec()],False,1),[b])
        self.assertEqual(calls,[1]);self.assertEqual(obj.last_selection_debug['selected_actor_ids'],[12])


class LauncherContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path='RIFT/scripts/run.py';tree=ast.parse(Path(path).read_text(encoding='utf-8'))
        main=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                  and any(isinstance(c,ast.Constant) and c.value=='__main__' for c in n.test.comparators))
        statements=[]
        for node in main.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
            statements.append(node)
        scope={'osp':os.path,'__file__':str(Path(path).resolve())}
        exec(compile(ast.Module(body=statements,type_ignores=[]),path,'exec'),scope)
        cls.parser=scope['parser']

    def parse_launcher(self,path):
        text=Path(path).read_text(encoding='utf-8')
        cmd=text.split('scripts/run.py ',1)[1].split('2>&1',1)[0].replace('\\\n',' ')
        for old,new in {'$BASE_PORT':'33000','$BASE_TM_PORT':'53000','$SEED':'0',
                        '$map_mode':'legacy','$intensity':'0.2'}.items():cmd=cmd.replace(old,new)
        return self.parser.parse_args(shlex.split(cmd)),text

    def test_strict_launcher_uses_attention_one_shot_and_valid_xml_ids(self):
        a,text=self.parse_launcher('RIFT/scripts/run_hv_strict_diagnostics.sh')
        self.assertEqual(a.cbv_recog,'attention');self.assertEqual(a.hv_generation_max_attempts,1)
        self.assertFalse(a.hv_generation_rolling);self.assertFalse(a.hv_generation_partial_context)
        self.assertFalse(a.hv_generation_gate_on_existing_rift_positive)
        self.assertFalse(a.hv_generation_register_cbv_from_audit)
        ids=set(re.search(r'ROUTES="\$\{ROUTES:-([^}]+)\}',text).group(1).split(','))
        known={r.get('id') for r in ElementTree.parse('RIFT/rift/scenario/route/bench2drive220.xml').getroot().findall('route')}
        self.assertTrue(ids<=known)

    def test_collection_launcher_keeps_negative_pool_and_exports_topology(self):
        a,_=self.parse_launcher('RIFT/scripts/run_hv_collection_diagnostics.sh')
        self.assertEqual(a.mode,'collect_hv_generation')
        self.assertEqual(a.hv_generation_collect_min_forward_rift_positive,0)
        self.assertTrue(a.hv_snapshot_native_map_export)
        self.assertEqual(a.hv_collection_traffic_intensity,.2)
        self.assertEqual(a.hv_generation_max_frames_per_episode,40)


if __name__=='__main__':unittest.main()
