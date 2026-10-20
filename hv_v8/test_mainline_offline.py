#!/usr/bin/env python3
"""Offline source-contract tests. No CARLA; they MUST NOT be reported as P1/P2 PASS."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

def load(name,loc):
    sp=importlib.util.spec_from_file_location(name,loc)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

base=Path(__file__).parent
p1=load('isolated',base/'p1_isolated95.py')
p2=load('direction',base/'p2_directional_support_v0.py')
lat=load('lateral',base/'p2_lateral_legal_segments.py')
assert p1.failure_class('P1_FAIL ego_route[9] exact lane identity mismatch')=='route_exact_lane_identity_mismatch'
assert p1.failure_class('P1_FAIL ego_route[0] exact get_waypoint returned None')=='route_exact_lookup_missing'
assert p1.failure_class('Traceback\nModuleNotFoundError: No module named hv_v8.common')=='unhandled_runtime_failure'
assert p1.FROZEN_SHA=='b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb'
print('P1_FAIL_CLASSIFICATION_FROM_ACTUAL_LOG_MESSAGES=PASS')

# Scientific guard: generator input contains condition + map edges only; adding
# arbitrary fake GT diagnostic values cannot change candidate generation.
graph={'schema':'p2_lane_topology_probe_v1','town':'Town15','node_keys':[[1,0,-1],[2,0,-1],[3,0,-1]],
       'directed_edges':[{'from':[2,0,-1],'to':[1,0,-1]},{'from':[3,0,-1],'to':[2,0,-1]}],
       'records':[{'fake_secret_GT':'a'}]}
conditions=[{'record_index':100,'town':'Town15','ego_route':[{'road_id':1,'section_id':0,'lane_id':-1}], 'manifest_scene_id':'a'}]
x=p2.make(graph,conditions,max_hops=4)
graph['records']=[{'fake_secret_GT':'b','target_actor_ids':[999]}]
y=p2.make(graph,conditions,max_hops=4)
assert x==y and x['GT_used_in_generator'] is False
assert x['records'][0]['candidate_lane_hops'][0]['lane']==[1,0,-1]
assert len(x['records'][0]['candidate_lane_hops'])==3
print('P2_GT_LEAKAGE_GUARD_AND_DIRECTED_BACKTRACE=PASS')

class LaneChange:
    Left=1;Right=2;Both=3
class LaneType:
    Driving=8
class Carla:
    LaneChange=LaneChange;LaneType=LaneType
class Obj:
    pass

def waypoint(lane):
    w=Obj();w.is_junction=False;w.lane_type=LaneType.Driving
    w.lane_change=LaneChange.Left;w.road_id=140;w.section_id=0;w.lane_id=lane
    w.transform=Obj();w.transform.location=Obj();w.transform.rotation=Obj()
    w.transform.location.x=0.;w.transform.location.y=0.;w.transform.location.z=0.
    w.transform.rotation.yaw=0.
    w.get_right_lane=lambda:None
    return w
w=waypoint(-2);v=waypoint(-3)
v.transform.location.y=3.5
w.get_left_lane=lambda:v
assert lat.valid_lateral(w,'left',Carla) is v
w.lane_change=LaneChange.Right
assert lat.valid_lateral(w,'left',Carla) is None
w.lane_change=LaneChange.Left;v.lane_id=3
assert lat.valid_lateral(w,'left',Carla) is None
v.lane_id=-3;v.transform.location.z=1.4
assert lat.valid_lateral(w,'left',Carla) is None
print('P2_LATERAL_PERMISSION_DIRECTION_ELEVATION_GUARDS=PASS')

if len(sys.argv)>1:
    ev=Path(sys.argv[1]);g=json.loads((ev/'p2_topology_town15_20261019_041323_1291.json').read_text())
    cond=[json.loads(s) for s in (ev/'p2_conditions_95.jsonl').read_text().splitlines() if s.strip()]
    a=p2.make(g,cond)
    assert len(a['records'])==12
    assert sum(len(rec['candidate_lane_hops']) for rec in a['records'])>0
    print('P2_REAL_TOWN15_12_CONDITION_ONLY_BASELINE=PASS_NOT_APPROACH_RECALL')
print('ALL_OFFLINE_CHECKS_PASS_NOT_CARLA')
