"""Offline regression for V7 round3; requires only numpy, no CARLA process."""
import copy
import importlib
import json
import math
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from v7.station_audit import station_to_progress,typed_station_path,actor_station_audit
from v7_round3_cli import normalize_map,jobs_index


def lane(uid, ss=(0.,10.,20.), direction=1, y=0.):
    ss=list(ss)
    progress=[0.,10.,20.]
    assert len(ss)==len(progress)
    road,sect,lid=map(int,uid.split(':'))
    return {'lane_uid':uid,'road_id':road,'section_id':sect,'lane_id':lid,
            'station_travel_direction':direction,'opendrive_s_samples':ss,
            'directed_progress_m':progress,'width_m':[3.5]*3,
            'centerline_local_xyz':[[x,y,0.] for x in progress],
            'centerline_world_xyz':[[x,y,0.] for x in progress], 'is_junction':False}


def edge(a,b,source_s,destination_s,kind='lane_change_left'):
    return {'source':a,'destination':b,'source_s_m':source_s,
            'destination_s_m':destination_s,'kind':kind,'destination_exported':True}


class StationGraphTests(unittest.TestCase):
    def test_same_lane_forward_and_reverse(self):
        G={'lanes':[lane('1:0:1')], 'edges':[]}
        self.assertEqual(typed_station_path(G,'1:0:1',1.,'1:0:1',19.)['status'],'supported')
        self.assertEqual(typed_station_path(G,'1:0:1',18.,'1:0:1',4.)['status'],'unknown')

    def test_typed_station_change_only_before_passage(self):
        G={'lanes':[lane('1:0:1'),lane('1:0:2',y=3.5)],
           'edges':[edge('1:0:1','1:0:2',10.,10.)]}
        p=typed_station_path(G,'1:0:1',1.,'1:0:2',18.)
        self.assertEqual(p['status'],'supported')
        self.assertEqual(p['typed_transition_count'],1)
        self.assertEqual(typed_station_path(G,'1:0:1',13.,'1:0:2',18.)['status'],'unknown')
        self.assertEqual(typed_station_path(G,'1:0:1',1.,'1:0:2',8.)['status'],'unknown')

    def test_opendrive_decreasing_travel(self):
        r=lane('1:0:-1',ss=(20.,10.,0.),direction=-1)
        self.assertAlmostEqual(station_to_progress(r,15.),5.)
        self.assertAlmostEqual(station_to_progress(r,5.),15.)
        self.assertIsNone(station_to_progress(r,33.))

    def test_legacy_untyped_edge_cannot_certify(self):
        G={'lanes':[lane('1:0:1'),lane('1:0:2')],
           'edges':[{'source':'1:0:1','destination':'1:0:2','kind':'longitudinal','source_s_m':None,'destination_s_m':None}]}
        self.assertEqual(typed_station_path(G,'1:0:1',1.,'1:0:2',18.)['status'],'unknown')

    def test_bounded_search_does_not_prematurely_abort(self):
        A,B,C='1:0:1','1:0:2','1:0:3'
        G={'lanes':[lane(A),lane(B),lane(C)], 'edges':[
            edge(A,C,10.,10.),edge(A,B,10.,10.),edge(B,C,10.,10.)]}
        self.assertEqual(typed_station_path(G,A,1.,C,19.,max_transitions=1)['status'],'supported')

    def test_real_pose_membership_flag(self):
        uid='1:0:1'; graph={'lanes':[lane(uid)],'edges':[]}
        c={'condition':{'directed_map':graph,'ego_world_pose':{'x':0.,'y':0.,'z':0.,'yaw_deg':0.}}}
        actor = {
            'pose': {
                'x': 1.,
                'y': 2.2,
                'z': 0.,
                'yaw_deg': 0.
            },
            'road_geometry': {
                'preview_waypoint': {
                    'road_id': 1,
                    'section_id': 0,
                    'lane_id': 1,
                    'x': 0.,
                    'y': 0.,
                    'z': 0.,
                    'yaw_deg': 0.,
                    's': 0.
                }
            },
            'native': {
                'interaction_waypoint': {
                    'road_id': 1,
                    'section_id': 0,
                    'lane_id': 1,
                    'x': 15.,
                    'y': 0.,
                    'z': 0.,
                    'yaw_deg': 0.,
                    's': 15.
                }
            }
        }
        result=actor_station_audit(c,actor,anchor_match=['route-lane:'+uid])
        self.assertEqual(result['status'],'unknown')
        self.assertFalse(result['lateral_membership'])
        self.assertEqual(
            result['reason'],
            'initial_lateral_membership_failed'
        )

    def test_map_normalization_does_not_modify_source(self):
        def p(s,x):return [x,0.,0.]
        a={'lanes':[{'lane_uid':'1:0:1','centerline':[p(1,0),p(2,1e-6),p(3,1)],
                    's_samples_m':[0.,0.000001,1.], 'width_m':[3.,3.,3.],
                    'yaw_samples_deg':[0.,0.,0.], 'station_travel_direction':1}],
           'evidence':{}}
        copy_a=copy.deepcopy(a)
        out,n=normalize_map(copy_a)
        self.assertEqual(n,1)
        self.assertEqual(len(out['lanes'][0]['centerline']),2)
        self.assertEqual(len(a['lanes'][0]['centerline']),3)

    def test_missing_preview_station_fails_closed(self):
        uid = '1:0:1'
        graph = {'lanes': [lane(uid)], 'edges': []}

        c = {
            'condition': {
                'directed_map': graph,
                'ego_world_pose': {
                    'x': 0.,
                    'y': 0.,
                    'z': 0.,
                    'yaw_deg': 0.
                }
            }
        }

        actor = {
            'pose': {
                'x': 1.,
                'y': 0.,
                'z': 0.,
                'yaw_deg': 0.
            },
            'road_geometry': {
                'preview_waypoint': {
                    'road_id': 1,
                    'section_id': 0,
                    'lane_id': 1
                }
            },
            'native': {
                'interaction_waypoint': {
                    'road_id': 1,
                    'section_id': 0,
                    'lane_id': 1,
                    'x': 15.,
                    'y': 0.,
                    'z': 0.,
                    'yaw_deg': 0.,
                    's': 15.
                }
            }
        }

        result = actor_station_audit(
            c,
            actor,
            anchor_match=['route-lane:' + uid]
        )

        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(
            result['reason'],
            'native_preview_station_unrepresentable'
        )
        self.assertNotIn('lateral_membership', result)

    def test_preview_to_native_trace_edge_is_checked(self):
        from v7.diagnostics import inspect_actor
        A,B,C='1:0:1','2:0:1','3:0:1'
        g={'lanes':[lane(A),lane(B),lane(C)],
           'edges':[{'source':B,'destination':C,'destination_exported':True,
                     'kind':'longitudinal','source_s_m':10.,'destination_s_m':0.}]}
        c={'condition':{'directed_map':g,'ego_world_pose':{
            'x':0.,'y':0.,'z':0.,'yaw_deg':0.}}}
        def wp(uid,x=3):
            road,sect,lid=map(int,uid.split(':'))
            return {'road_id':road,'section_id':sect,'lane_id':lid,
                    'x':x,'y':0.,'z':0.,'s':x,'yaw_deg':0.}
        a={'actor_id':11,'type_id':'vehicle.tesla.model3',
           'pose':{'x':3.,'y':0.,'z':0.,'yaw_deg':0.},
           'local_state':[3.,0.,0.,0.], 'forward_fov':True,
           'road_geometry':{'preview_waypoint':wp(A)},
           'native':{'native_candidate':True,
                     'interaction_waypoint':wp(C,15),
                     'target_route_to_interaction':[wp(B,5),wp(C,10)]}}
        r=inspect_actor(c,a,[],[])
        self.assertIn((A,B),[(e['source'],e['destination']) for e in r['native_route_missing_edges']])

    def test_duplicate_plan_rejected(self):
        with self.assertRaises(ValueError):jobs_index({'jobs':[{'record_index':2},{'record_index':2}]})


if __name__=='__main__':unittest.main()
