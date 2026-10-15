"""V7 round-2 regression tests. No live CARLA required.

Run from repository root after applying patch:
  python -m unittest -v test_hv_v7_round2 test_hv_v7
"""
import math
import sys
import types
import unittest
from enum import IntFlag
from unittest.mock import patch

from v7.carla_map import export_directed_lanes
from v7.directed_map import DirectedMapBuilder, directed_path
from v7.approach_builder import ConflictAnchorBuilder, ApproachBuilder


class LaneChange(IntFlag):
    None_ = 0
    Right = 1
    Left = 2
    Both = 3


class Waypoint:
    lane_type = 'Driving'
    is_junction = False
    lane_width = 3.5

    def __init__(self, road, lane, s, x, y=0., yaw=0.):
        self.road_id,self.section_id,self.lane_id=road,0,lane
        self.s=float(s)
        self.transform=types.SimpleNamespace(
            location=types.SimpleNamespace(x=float(x),y=float(y),z=0.),
            rotation=types.SimpleNamespace(yaw=float(yaw)))
        self._next=[];self._previous=[];self.left=None;self.right=None
        self.left_lane_marking=types.SimpleNamespace(lane_change=LaneChange.None_)
        self.right_lane_marking=types.SimpleNamespace(lane_change=LaneChange.None_)

    def next(self,step):return self._next
    def previous(self,step):return self._previous
    def get_left_lane(self):return self.left
    def get_right_lane(self):return self.right


class Map:
    def __init__(self,seeds,topology=()):self.seeds=seeds;self.topology=topology
    def generate_waypoints(self,step):return self.seeds
    def get_topology(self):return self.topology


class MapRepairTests(unittest.TestCase):
    def setUp(self):
        self.carla=types.SimpleNamespace(LaneChange=LaneChange)
        self.ego={'x':0.,'y':0.,'z':0.,'yaw_deg':0.}

    def export(self,world,route=(),**kwargs):
        with patch.dict(sys.modules,{'carla':self.carla}):
            return export_directed_lanes(world,self.ego,route,**kwargs)

    def test_many_near_duplicate_global_seeds_do_not_expand_recursively(self):
        points=[Waypoint(1,1,10+i/1000.,10+i/1000.) for i in range(1500)]
        points.append(Waypoint(1,1,12.,12.))
        lanes,evidence=self.export(Map(points),step_m=1.,max_samples=2000)
        self.assertEqual(len(lanes),1)
        self.assertLess(evidence['visited_samples'],6)
        self.assertLess(evidence['waypoint_probe_count'],1550)
        self.assertIn('s_samples_m',lanes[0])

    def test_negative_open_drive_station_is_still_forward_lane_travel(self):
        first=Waypoint(17,1,10.,0.,yaw=0.)
        second=Waypoint(17,1,9.,1.,yaw=0.)
        first._next=[second];second._previous=[first]
        lanes,evidence=self.export(Map([]),[first],max_samples=20)
        self.assertEqual(lanes[0]['station_travel_direction'],-1)
        self.assertEqual(lanes[0]['s_samples_m'],[10.,9.])
        self.assertEqual(lanes[0]['centerline'][0][:2],[0.,0.])

    def test_lane_change_requires_marking_and_matching_travel_direction(self):
        a=Waypoint(137,-1,10.,10.,0.);a2=Waypoint(137,-1,12.,12.,0.)
        b=Waypoint(137,-2,10.,10.,3.5);b2=Waypoint(137,-2,12.,12.,3.5)
        a.left=b;a.left_lane_marking.lane_change=LaneChange.Left
        lanes,evidence=self.export(Map([a,a2,b,b2]),max_samples=20)
        change=[e for e in evidence['typed_transitions'] if e['kind']=='lane_change_left']
        self.assertEqual(len(change),1)
        graph=DirectedMapBuilder().build(lanes,self.ego,'map-sha',evidence=evidence)
        self.assertEqual(directed_path(graph,'137:0:-1','137:0:-2'),['137:0:-1','137:0:-2'])
        self.assertGreater(graph['lane_change_edge_count'],0)
        a.left_lane_marking.lane_change=LaneChange.None_
        _,blocked=self.export(Map([a,a2,b,b2]),max_samples=20)
        self.assertFalse([e for e in blocked['typed_transitions'] if e['kind'].startswith('lane_change')])
        a.left_lane_marking.lane_change=LaneChange.Left
        b.transform.rotation.yaw=180.
        _,opposing=self.export(Map([a,a2,b,b2]),max_samples=20)
        self.assertFalse([e for e in opposing['typed_transitions'] if e['kind'].startswith('lane_change')])

    def test_map_export_is_bounded_and_explicitly_incomplete(self):
        seeds=[Waypoint(1,1,float(i),float(i)) for i in range(15)]
        with self.assertRaisesRegex(RuntimeError,'budget'):
            self.export(Map(seeds),max_samples=10)
        lanes,evidence=self.export(Map(seeds),max_samples=20)
        self.assertFalse(evidence['topology_complete'])
        self.assertEqual(evidence['export_schema_version'],'hv_v7_native_map_v2')

    def test_preliminary_paths_keep_map_only_typed_lane_change(self):
        a=Waypoint(137,-1,10.,10.,0.);a2=Waypoint(137,-1,12.,12.,0.)
        b=Waypoint(137,-2,10.,10.,3.5);b2=Waypoint(137,-2,12.,12.,3.5)
        a.left=b;a.left_lane_marking.lane_change=LaneChange.Left
        lanes,evidence=self.export(Map([a,a2,b,b2]),max_samples=20)
        graph=DirectedMapBuilder().build(lanes,self.ego,'map-sha',evidence=evidence)
        condition={'condition':{'ego_route':{'waypoints':[
            {'declared_lane_uid':'137:0:-2','ego_route_distance_m':0.,'lane_matches':[
                {'lane_uid':'137:0:-2','progress_m':1.,'projected_xyz':[11.,3.5,0.]}]}]},
            'directed_map':graph}}
        anchors=ConflictAnchorBuilder().build(condition)
        approaches=ApproachBuilder().build(condition,anchors)
        self.assertTrue(any(a['start_lane_uid']=='137:0:-1' for a in approaches))
        self.assertTrue(all(a['representation_status']=='pre_chart_support_only' for a in approaches))


if __name__=='__main__':unittest.main()
