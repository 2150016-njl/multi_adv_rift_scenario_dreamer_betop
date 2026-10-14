import copy
import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'scenario-dreamer'))
from v7.coordinates import world_to_local,local_to_world,velocity_to_local,velocity_to_world,infer_frame_from_map
from v7.condition_builder import build_condition
from v7.schema import validate_condition,validate_target,validate_triplets
from v7.directed_map import DirectedMapBuilder,directed_path
from v7.approach_builder import ConflictAnchorBuilder,ApproachBuilder,intervals
from v7.diagnostics import inspect_actor
from v7.pipeline import target_record,unique_index,observation_for


def lane(uid='1:0:1',points=None,successors=None):
    return {'lane_uid':uid,'road_id':int(uid.split(':')[0]),'section_id':0,'lane_id':1,
            'is_junction':False,'centerline':points or [[0.,0.,0.],[50.,0.,0.]],
            'width_m':[3.5,3.5],'successors':successors or [],'predecessors':[]}


def condition():
    return build_condition('sample',{'ego':dict(x=0.,y=0.,z=0.,yaw_deg=0.,speed_mps=4.),'actors':[],
        'ego_route':[dict(x=x,y=0.,z=0.,road_id=1,section_id=0,lane_id=1,s=x) for x in [10.,20.,40.]],
        'observation_view':'normal_online_snapshot'},[lane()],{'opendrive_sha256':'map'})


def actor(x=12.,y=0.,yaw=0.,speed=20.):
    pose={'actor_id':7,'type_id':'historical','x':x,'y':y,'z':0.,'yaw_deg':yaw,'speed_mps':speed}
    wp=lambda x:dict(x=x,y=0.,z=0.,yaw_deg=0.,road_id=1,section_id=0,lane_id=1,s=x)
    return {'actor_id':7,'type_id':'historical','pose':pose,'local_state':[x,y,math.radians(yaw),speed,4.,2.,1.],
        'forward_fov':0<x<=32 and abs(y)<=32,'road_geometry':{'preview_waypoint':wp(x+2)},
        'native':{'native_candidate':True,'interaction_waypoint':wp(40.),'target_route_to_interaction':[wp(x+2),wp(40.)]}}


class V7Tests(unittest.TestCase):
    def test_four_directions_and_nonzero_yaw_position_velocity_roundtrip(self):
        points=np.array([[10.,0.,3.],[-10.,0.,-1.],[0.,10.,5.],[0.,-10.,0.]])
        for yaw in [0.,30.,90.,-135.,179.]:
            ego=dict(x=123.,y=-9.,z=2.,yaw_deg=yaw)
            np.testing.assert_allclose(local_to_world(ego,world_to_local(ego,points)),points,atol=1.e-12)
            np.testing.assert_allclose(velocity_to_world(ego,velocity_to_local(ego,points)),points,atol=1.e-12)
        ego=dict(x=0.,y=0.,z=0.,yaw_deg=0.)
        self.assertEqual(world_to_local(ego,[0.,10.,0.]),[0.,10.,0.])

    def test_map_only_frame_fit(self):
        ego=dict(x=41.,y=-20.,z=4.,yaw_deg=67.)
        world=[lane(points=local_to_world(ego,[[0.,0.,0.],[20.,0.,0.]]))]
        local=[lane(points=[[0.,0.,0.],[20.,0.,0.]])]
        recovered,error=infer_frame_from_map(world,local)
        self.assertLess(error,1.e-12)
        for key in ego:self.assertAlmostEqual(ego[key],recovered[key])

    def test_target_labels_and_generated_output_cannot_enter_condition(self):
        record=condition();poison=copy.deepcopy(record)
        poison['condition']['native']={'native_candidate':True}
        with self.assertRaisesRegex(ValueError,'leaked'):validate_condition(poison)
        poison=copy.deepcopy(record);poison['condition']['coordinate_frame']='ego_local_x_forward_y_left'
        with self.assertRaisesRegex(ValueError,'frame'):validate_condition(poison)

    def test_observation_allowlist_ignores_native_poison_and_online_offline_use_same_builder(self):
        observation={'ego':dict(x=0.,y=0.,z=0.,yaw_deg=0.),'actors':[],
                     'ego_route_local_xy':[[0.,0.],[40.,0.]],'observation_view':'normal_online_snapshot'}
        before=build_condition('id',observation,[lane()],{})
        observation.update(native={'actor_id':7},target_states=[[1.,2.,3.,4.]])
        after=build_condition('id',observation,[lane()],{})
        self.assertEqual(before,after)

    def test_packed_slots_front_first_second_can_be_rear_and_no_arbitrary_speed_cap(self):
        target=target_record('sample',[actor(-12.),actor(12.,speed=22.)])
        validate_target(target)
        self.assertEqual(target['target_states'][0][0],12.)
        self.assertEqual(target['target_states'][1][0],-12.)
        self.assertEqual(target['target_states'][0][3],22.)
        target['valid_mask']=[False,True]
        with self.assertRaises(ValueError):validate_target(target)

    def test_missing_front_target_is_unknown_not_zero(self):
        target=target_record('sample',[actor(-12.)]);validate_target(target)
        self.assertIsNone(target['num_targets'])
        self.assertIsNone(target['valid_mask'])
        target.update(num_targets=0,valid_mask=[False,False],target_states=[None,None])
        with self.assertRaisesRegex(ValueError,'zero'):validate_target(target)

    def test_ids_must_be_unique_and_three_files_aligned(self):
        with self.assertRaises(ValueError):unique_index([{'record_index':2}]*2,'fixture')
        c=condition();t=target_record('sample',[actor()]);o={'schema_version':'hv_v7_oracle_v1','record_id':'sample'}
        validate_triplets([c],[t],[o])
        o['record_id']='other'
        with self.assertRaises(ValueError):validate_triplets([c],[t],[o])

    def test_edges_remain_directed_unknown_boundaries_and_cycles_are_not_xy_repaired(self):
        graph=DirectedMapBuilder().build([lane(successors=['2:0:1','9:0:1']),lane('2:0:1',successors=['1:0:1'])],
                                      dict(x=0.,y=0.,z=0.,yaw_deg=0.),'hash')
        self.assertEqual(directed_path(graph,'1:0:1','2:0:1'),['1:0:1','2:0:1'])
        self.assertIsNone(directed_path(graph,'1:0:1','9:0:1'))
        self.assertEqual(len(graph['boundary_edges']),1)
        self.assertFalse(graph['topology_complete'])

    def test_anchor_outside_output_domain_is_retained_and_chart_limits_are_not_applied(self):
        c=condition();anchors=ConflictAnchorBuilder().build(c);approaches=ApproachBuilder().build(c,anchors)
        self.assertTrue(any(s['local_xyz'][0]==40 for a in anchors for s in a['route_samples']))
        # Large heading/lateral residuals are measured, not prematurely filtered.
        result=inspect_actor(c,actor(y=1.5,yaw=60.,speed=22.),anchors,approaches)
        self.assertTrue(result['preliminary_directed_support'])
        self.assertEqual(result['lane_chart_representability'],'not_tested')
        self.assertGreater(result['center_projection_to_preview_lane']['heading_error_rad'],.4)

    def test_output_rectangle_clips_segment_even_when_no_sample_point_is_inside(self):
        self.assertEqual(intervals([[-50.,0.,0.],[50.,0.,0.]],True),[[50.,82.]])
        self.assertEqual(intervals([[0.,-50.,0.],[0.,50.,0.]],True),[])

    def test_default_requires_real_snapshot_and_never_silently_reconstructs(self):
        from unittest.mock import patch
        with patch('v7.pipeline.resolve_source',return_value=None):
            with self.assertRaises(FileNotFoundError):observation_for('.',{'snapshot_path':'missing','target_actor_ids':[]},{})

    def test_reversed_export_is_a_map_failure_not_a_chart_yaw_limit(self):
        c=condition();node=c['condition']['directed_map']['lanes'][0]
        node['centerline_world_xyz'].reverse();node['centerline_local_xyz'].reverse()
        anchors=ConflictAnchorBuilder().build(c);approaches=ApproachBuilder().build(c,anchors)
        row=inspect_actor(c,actor(),anchors,approaches)
        self.assertTrue(row['preview_lane_direction_opposed'])
        self.assertFalse(row['preliminary_directed_support'])
        self.assertIn('exported_centerline_direction_opposes_native_waypoint',row['failure_reasons'])

    def test_native_calibration_comparison_detects_route_and_distance_changes(self):
        from v7.calibration import compare_native
        reference=actor()['native'];actual=copy.deepcopy(reference)
        self.assertEqual(compare_native(reference,actual),[])
        actual['target_route_to_interaction'][0]['lane_id']=2
        self.assertIn('native_route_changed',compare_native(reference,actual))

    def test_calibration_refuses_reserved_or_not_dedicated_server_before_import(self):
        from v7.calibration import calibrate
        for port,dedicated in [(38000,True),(31120,False)]:
            with self.assertRaises(ValueError):calibrate(None,None,'127.0.0.1',port,30,dedicated,{})

    def test_live_map_direction_uses_same_lane_next_not_cross_road_station_reset(self):
        from types import SimpleNamespace
        from v7.carla_map import export_directed_lanes
        class WP:
            lane_id=1;section_id=0;lane_type='Driving';is_junction=False;lane_width=3.5
            def __init__(self,road,s,x):
                self.road_id=road;self.s=s;self.fwd=[];self.back=[]
                self.transform=SimpleNamespace(location=SimpleNamespace(x=x,y=0.,z=0.),rotation=SimpleNamespace(yaw=0.))
            def next(self,step):return self.fwd
            def previous(self,step):return self.back
        a,b,c=WP(1,10.,0.),WP(1,9.,1.),WP(2,100.,2.)
        a.fwd=[b];b.back=[a];b.fwd=[c];c.back=[b]
        world_map=SimpleNamespace(generate_waypoints=lambda step:[])
        lanes,evidence=export_directed_lanes(world_map,dict(x=0.,y=0.),[b])
        self.assertEqual(lanes[0]['s_samples_m'],[10.,9.])
        self.assertEqual(lanes[0]['station_travel_direction'],-1)
        self.assertEqual(lanes[0]['centerline'],[[0.,0.,0.],[1.,0.,0.]])
        self.assertIn('2:0:1',lanes[0]['successors'])
        self.assertFalse(evidence['topology_complete'])

    def test_real_snapshot_path_checks_sha_and_withholds_only_bound_actors(self):
        import json
        from unittest.mock import patch
        target=actor()['pose'];visible=dict(actor()['pose'],actor_id=8,x=20.)
        scene={'coordinate_frame':'carla_world','idx':999,'ego':dict(x=0.,y=0.,z=0.,yaw_deg=0.),
               'ego_route':[],'agents':[target,visible]}
        pair={'snapshot_path':'snapshot.json','snapshot_sha256':'expected','target_actor_ids':[7]}
        ledger={'raw_pool':[{'actor_id':a['actor_id'],'pose':a} for a in scene['agents']]}
        with patch('v7.pipeline.resolve_source',return_value=Path('snapshot.json')), \
                patch('v7.pipeline.sha',return_value='expected'), \
                patch.object(Path,'read_text',return_value=json.dumps(scene)):
            observation,evidence=observation_for('.',pair,ledger)
            self.assertEqual([a['actor_id'] for a in observation['actors']],[8])
            self.assertEqual(evidence['snapshot_status'],'sha_verified')
            self.assertEqual(evidence['snapshot_internal_idx'],999)
        with patch('v7.pipeline.resolve_source',return_value=Path('snapshot.json')), \
                patch('v7.pipeline.sha',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):observation_for('.',pair,ledger)


if __name__=='__main__':unittest.main()
