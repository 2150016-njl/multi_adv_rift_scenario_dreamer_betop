#!/usr/bin/env python3
"""Real source import + positive/negative contracts. No CARLA PASS claimed."""
from __future__ import annotations
import ast,hashlib,json,math,subprocess,sys,tempfile,shutil
from pathlib import Path
from types import SimpleNamespace as O
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from hv_v8 import p1_native_audit as sci
from hv_v8 import p1_science_contract_v2 as geo
from hv_v8 import p2_approach_v1 as app

class Vec:
    def __init__(self,x,y,z):self.x,self.y,self.z=x,y,z
    def __sub__(self,o):return Vec(self.x-o.x,self.y-o.y,self.z-o.z)
    def distance(self,o):return math.sqrt((self.x-o.x)**2+(self.y-o.y)**2+(self.z-o.z)**2)
    def get_vector_angle(self,o):
        a=math.sqrt(self.x*self.x+self.y*self.y+self.z*self.z)
        b=math.sqrt(o.x*o.x+o.y*o.y+o.z*o.z)
        return math.acos(max(-1,min(1,(self.x*o.x+self.y*o.y+self.z*o.z)/(a*b)))) if a*b else float('nan')

class Rot:
    def __init__(self,yaw=0,pitch=0):self.yaw=yaw;self.pitch=pitch
    def get_forward_vector(self):
        y=math.radians(self.yaw);p=math.radians(self.pitch)
        return Vec(math.cos(p)*math.cos(y),math.cos(p)*math.sin(y),math.sin(p))
class Tr:
    def __init__(self,location,rotation):self.location=location;self.rotation=rotation
    def get_forward_vector(self):return self.rotation.get_forward_vector()
class WP:
    def __init__(self,road,laneid,s,x,y,z,junction=False,yaw=0):
        self.road_id=road;self.section_id=0;self.lane_id=laneid;self.s=s
        self.lane_type='Driving';self.is_junction=junction;self.transform=Tr(Vec(x,y,z),Rot(yaw))
    def next(self,step):return []
class Map:
    def __init__(self,source,exact,length=40):self.source=source;self.exact=exact;self.length=length
    def to_opendrive(self):return '<OpenDRIVE><road id="10" length="%.5f" /></OpenDRIVE>'%self.length
    def get_waypoint_xodr(self,road,laneid,s):return self.source.get((road,laneid,round(s,4)))
    def get_waypoint(self,location,project_to_road,lane_type):
        return self.exact.get((location.x,location.y,location.z))
CARLA=O(Location=lambda x,y,z:Vec(x,y,z),LaneType=O(Driving='Driving'))

def row(s,x,j=False):
    return {'x':x,'y':0,'z':0,'yaw_deg':0,'road_id':10,'section_id':0,'lane_id':-1,'s':s,'is_junction':j}

def must_fail(fn):
    try:fn()
    except (ValueError,RuntimeError,geo.require.__globals__['ContractError']):return
    raise AssertionError('expected strict refusal')

def tests():
    w0=WP(10,-1,5,5,0,0)
    w1=WP(10,-1,7,7,0,0,junction=True)
    alien=WP(99,2,99,7,0,0,junction=True)
    src={(10,-1,5.0):w0,(10,-1,7.0):w1}
    exact={(5,0,0):w0,(7,0,0):alien}
    route,checks=geo.route_waypoints_v2(CARLA,Map(src,exact),{'ego_route':[row(5,5),row(7,7,True)]})
    assert len(route)==2 and checks[1]['exact_lookup_status']=='verified_junction_overlap_other_road'
    assert route[1] is w1 and checks[1]['directed_link_to_previous']=='same_directed_lane_station'
    print('TEST_JUNCTION_SOURCE_WINS_BUT_EXACT_CONFLICT_PRESERVED=PASS')
    must_fail(lambda:geo.route_waypoints_v2(CARLA,Map(src,exact),{'ego_route':[row(5,5),row(7,7,False)]}))
    print('TEST_NONJUNCTION_MISMATCH_FAIL_CLOSED=PASS')
    broken_src=dict(src);broken_src[(10,-1,7.0)]=WP(10,-1,7,50,0,0,True)
    must_fail(lambda:geo.route_waypoints_v2(CARLA,Map(broken_src,exact),{'ego_route':[row(5,5),row(7,7,True)]}))
    print('TEST_INVALID_XODR_SOURCE_REJECTED=PASS')
    must_fail(lambda:geo.route_waypoints_v2(CARLA,Map(src,exact),{'ego_route':[row(7,7,True),row(5,5)]}))
    print('TEST_WRONG_WAY_DIRECTED_STATION_REJECTED=PASS')
    # 3D FOV differs from XY: forward +x, target with same xy but steep z.
    a=Tr(Vec(0,0,0),Rot(0))
    b=Tr(Vec(1,0,10),Rot(0))
    assert sci.is_within_distance_ahead(b,a,100,35) is False
    assert sci.is_within_distance_ahead(b,a,100,90) is True
    print('TEST_NATIVE_3D_VECTOR_ANGLE_NOT_XY=PASS')
    assert ' // 2' in __import__('inspect').getsource(sci.exact_interaction)
    print('TEST_RIFT_HALF_FOV_FLOOR=PASS')
    candidate={'candidate_filter_pass':None}
    assert sci.native_truth_status(candidate,{'status':'complete','native_interaction':True})==('unknown',None)
    print('TEST_NAN_SEMANTICS_UNKNOWN_NOT_FALSE=PASS')
    # Test candidate angle conversion radians->degrees with synthetic junction.
    actor={'_ego_location':Vec(0,0,0),'x':20,'y':0,'z':0,'yaw_deg':0,'pitch_deg':0,'roll_deg':0}
    old=sci.carla_transform;oldprev=sci.preview_waypoint
    try:
        sci.carla_transform=lambda carla,a:Tr(Vec(20,0,0),Rot(0))
        sci.preview_waypoint=lambda carla,cmap,tr:O(is_junction=True,transform=Tr(Vec(20,0,0),Rot(40)))
        c=sci.candidate_semantics(CARLA,None,actor)
        assert abs(c['junction_heading_error_deg']-40.0)<1e-8 and c['junction_heading_pass'] is False
        print('TEST_JUNCTION_RAD_TO_DEG_CONVERTED=PASS')
        sci.preview_waypoint=lambda carla,cmap,tr:O(is_junction=True,transform=O(get_forward_vector=lambda:Vec(0,0,0)))
        c=sci.candidate_semantics(CARLA,None,actor)
        assert c['candidate_filter_pass'] is None and c['heading_numeric_status']=='unknown_nonfinite'
        print('TEST_JUNCTION_NAN_NOT_LEGAL_NEGATIVE=PASS')
    finally:sci.carla_transform=old;sci.preview_waypoint=oldprev
    topology={'town':'Town15','map_sha256':'a'*64,'node_keys':[[1,0,-1],[2,0,-1],[3,0,-1]],
              'directed_edges':[{'from':[1,0,-1],'to':[2,0,-1]}]}
    lateral={'town':'Town15','opendrive_sha256':'a'*64,'GT_used':False,'intervals':[
        {'from_lane':[3,0,-1],'to_lane':[2,0,-1],'source_s_min':5,'source_s_max':15,
         'source_marking_permission_checked':True,'geometry_checks_passed_at_samples':True}]}
    geometry={'town':'Town15','map_sha256':'a'*64,'GT_used':False,
         'sampled_lane_station_ranges':[{'lane':[1,0,-1],'s_min':0,'s_max':30},
           {'lane':[2,0,-1],'s_min':0,'s_max':30},{'lane':[3,0,-1],'s_min':0,'s_max':25}],
         'records':[{'record_index':90,'crossing_anchor_candidates':[]}]}
    cond=[{'town':'Town15','record_index':90,'ego_route':[{'road_id':2,'section_id':0,'lane_id':-1}],
           'gt_target_positions_in_condition':False}]
    doc=app.make(topology,cond,lateral,geometry)
    assert {tuple(x['lane']) for x in doc['records'][0]['approaches']}=={(1,0,-1),(2,0,-1),(3,0,-1)}
    assert any(x['lateral_transition_s_bounds'] is not None for x in doc['records'][0]['approaches'])
    ev=app.evaluate(doc,[{'record_index':90,'actor_id':99,'actor_lane':[3,0,-1]}],False)
    assert ev['counts']['hit']==1 and ev['formal_GT_approach_recall_proven'] is False
    print('TEST_P2_LEGAL_LATERAL_S_INTERVAL_AND_GT_SEPARATION=PASS')
    bad=[dict(cond[0],targets=[{'actor_id':99}])]
    must_fail(lambda:app.make(topology,bad,lateral,geometry))
    must_fail(lambda:app.evaluate(doc,[{'record_index':90,'actor_id':99,'actor_lane':[3,0,-1]}],True))
    print('TEST_GT_LEAKAGE_AND_FAKE_110_REJECTED=PASS')
    for name in ('p1_native_audit','p1_physics_independent','p2_approach_v1','p2_geometry_sampler','p2_prepare_town','p2_lateral_legal_segments','p2_topology_probe'):
        cp=subprocess.run([sys.executable,'-m','hv_v8.'+name,'--help'],cwd=str(ROOT),
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        assert cp.returncode==0,(name,cp.stderr)
    print('TEST_REAL_MODULE_ENTRYPOINTS_IMPORT_AND_HELP=PASS')
    # Recreate a real capsule from exactly the pinned parent frozen auditor,
    # then import it using python -m from the capsule, not a mock import.
    import prepare_science_capsule as prep
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); existing=root/'data/hv_init/v7/p1_frozen_capsule_world_transition_depcomplete_v2_20261019/hv_v8'
        existing.mkdir(parents=True)
        shutil.copy2(ROOT/'parent_p1_native_audit.py',existing/'p1_native_audit.py')
        shutil.copy2(ROOT/'hv_v8/common.py',existing/'common.py')
        shutil.copy2(ROOT/'hv_v8/p1_world_guard.py',existing/'p1_world_guard.py')
        prep.COMMON=hashlib.sha256((existing/'common.py').read_bytes()).hexdigest()
        cap=prep.prepare(root,ROOT)
        m=cap/'hv_v8/p1_native_audit.py'
        cmd=[sys.executable,'-m','hv_v8.p1_native_audit','--help']
        env=dict(__import__('os').environ,PYTHONPATH=str(cap))
        cp=subprocess.run(cmd,cwd=str(cap),env=env,text=True,capture_output=True)
        assert cp.returncode==0 and '--manifest' in cp.stdout,(cp.returncode,cp.stderr)
        assert hashlib.sha256(m.read_bytes()).hexdigest()==prep.CHILD
        must_fail(lambda:prep.prepare(root,ROOT)) if False else None
        try:prep.prepare(root,ROOT)
        except FileExistsError:pass
        else:raise AssertionError('capsule overwritten')
    print('TEST_ACTUAL_IMMUTABLE_CAPSULE_AND_PYTHON_MINUS_M_IMPORT=PASS')
    for file in list((ROOT/'hv_v8').glob('*.py'))+[ROOT/'prepare_science_capsule.py']:
        ast.parse(file.read_text(),feature_version=8)
    print('TEST_PY38_SYNTAX_GRAMMAR=PASS')
    print('ALL_SCIENCE_V2_OFFLINE_CONTRACTS_PASS_NOT_CARLA')

if __name__=='__main__':tests()
