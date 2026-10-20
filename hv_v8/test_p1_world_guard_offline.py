#!/usr/bin/env python3
"""Contract tests for correct world EPISODE handling; NOT real CARLA."""
import ast
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

here=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('p1_world_guard',str(here/'p1_world_guard.py'))
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
S='<OpenDRIVE><road id="1" length="3"/></OpenDRIVE>'
GOOD=hashlib.sha256(S.encode('utf8')).hexdigest()
BAD=hashlib.sha256(b'wrong').hexdigest()

class World:
    def __init__(self, ep, town, xodr=S):
        self.id=ep
        self.map=type('M',(), {'name':'Carla/Maps/'+town+'/'+town,
                              'to_opendrive':lambda self:xodr})()
    def get_map(self): return self.map

class FakeClock:
    def __init__(self):self.now=0.0
    def time(self):return self.now
    def sleep(self,n):self.now+=n
class Client:
    def __init__(self, before, after):
        self.before=before;self.after=list(after);self.load_calls=0;self.calls=0
    def get_world(self):
        self.calls+=1
        if self.load_calls==0: return self.before
        if len(self.after)>1:return self.after.pop(0)
        return self.after[0]
    def load_world(self,town): self.load_calls+=1;return self.before

# False-positive original behavior: load_world returns OLD world, then new world appears.
f=FakeClock(); c=Client(World(8,'Town15'),[World(8,'Town15'),World(8,'Town15'),World(9,'Town11'),World(9,'Town11')])
with tempfile.TemporaryDirectory() as d:
    evidence=Path(d)/'transition.json'
    w,cm,sha,ev=g.load_verified_world(c,'Town11',GOOD,155,evidence,10,.25,f.time,f.sleep)
    assert w.id==9 and cm.name.endswith('Town11') and sha==GOOD and c.load_calls==1
    assert len(ev['observations'])==4 and ev['status']=='verified' and evidence.is_file()
    assert json.loads(evidence.read_text())['verified_episode_id']==9
    assert g.assert_same_episode_map(c,9,'Town11',GOOD)
print('TEST_STALE_WORLD_TOWN15_TO_TOWN11_BARRIER=PASS')

# Same-Town reload still requires fresh episode (not just Town/SHA equality).
f=FakeClock(); c=Client(World(4,'Town12'),[World(4,'Town12'),World(5,'Town12'),World(5,'Town12')])
assert g.load_verified_world(c,'Town12',GOOD,0,None,10,.1,f.time,f.sleep)[0].id==5
print('TEST_SAME_TOWN_RELOAD_REQUIRES_NEW_EPISODE=PASS')

# Wrong map never accepted: timeout, after exactly ONE load attempt.
f=FakeClock();c=Client(World(4,'Town15'),[World(4,'Town15')]);
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'failure.json'
    try:g.load_verified_world(c,'Town11',GOOD,155,p,.4,.1,f.time,f.sleep)
    except g.MapTransitionFailure as exc:assert 'timeout' in str(exc)
    else:raise AssertionError('old world unexpectedly accepted')
    assert c.load_calls==1 and json.loads(p.read_text())['status']=='fail_closed'
print('TEST_STALE_WORLD_TIMEOUT_FAIL_CLOSED=PASS')

# Correct Town but OpenDRIVE hash differs: never silently accept.
f=FakeClock();c=Client(World(4,'Town15'),[World(5,'Town11','WRONG')])
try:g.load_verified_world(c,'Town11',GOOD,155,None,1,.1,f.time,f.sleep)
except g.MapTransitionFailure as exc:assert 'SHA mismatch' in str(exc)
else:raise AssertionError('bad SHA accepted')
print('TEST_REAL_MAP_SHA_CONFLICT_FAIL_CLOSED=PASS')

# Another client switches/reloads the world after the barrier.
c=Client(World(4,'Town12'),[World(5,'Town12')]);c.load_calls=1
try:g.assert_same_episode_map(c,4,'Town12',GOOD)
except g.MapTransitionFailure as exc:assert 'world changed' in str(exc)
else:raise AssertionError('episode change not detected')
print('TEST_EXTERNAL_WORLD_CHANGE_FAIL_CLOSED=PASS')

# Parent and child source: only audit_record function was changed.
parent=Path('/mnt/data/p1_latest_audit_inspect/data/hv_init/v7/p1_frozen_capsule_occupancy_joint_20261019_034017_031802/hv_v8/p1_native_audit.py')
child=here/'p1_native_audit_worldfix.py'
a=child.read_text();assert hashlib.sha256(a.encode()).hexdigest()=='b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb'
assert 'assert_same_episode_map(' in a
assert 'load_verified_world(' in a
assert 'world = client.load_world(' not in a
# Portable source correctness does not assume a local absolute parent exists.
if parent.is_file():
    assert hashlib.sha256(parent.read_bytes()).hexdigest()=='ed27a2b78ffc7c3e6159db85d5756ff2ad9bfef754e79043225d343f5840854f'
    original=ast.parse(parent.read_text()); modified=ast.parse(a)
    f=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before,after=f(original),f(modified)
    assert before.keys()==after.keys()
    assert [k for k in before if before[k]!=after[k]]==['audit_record']
print('TEST_FROZEN_SCIENCE_FUNCTIONS_UNCHANGED_EXCEPT_WORLD_LIFECYCLE=PASS')

# Census also references shared guard and writes correct per-Town partial summaries.
census=(here/'p1_route_geometry95_forensic_v2.py').read_text()
assert "from p1_world_guard import load_verified_world, assert_same_episode_map" in census
assert "summary['towns'][town]['status_counts'] = dict(town_counts)" in census
ast.parse(census)
print('TEST_CENSUS_GUARD_AND_PARTIAL_SUMMARY_CONTRACT=PASS')
print('P1_WORLD_GUARD_OFFLINE_CONTRACT_PASS_NOT_REAL_CARLA')

# Production census driver through two distinct Towns with the same transition race.
spec=importlib.util.spec_from_file_location('route_v2',str(here/'p1_route_geometry95_forensic_v2.py'))
censusmod=importlib.util.module_from_spec(spec)
import sys
sys.path.insert(0,str(here))
spec.loader.exec_module(censusmod)
class A:
    def filter(self,_):return []
class CM:
    def __init__(self,town):self.name='Carla/Maps/'+town+'/'+town
    def to_opendrive(self):return S
    def get_waypoint_xodr(self,*_):return self.wp()
    def get_waypoint(self,*_,**__):return self.wp()
    def wp(self):
        tr=type('T',(),{'location':type('L',(),{'x':3.,'y':5.,'z':12.})(),
                        'rotation':type('R',(),{'yaw':45.})()})()
        return type('WP',(),{'road_id':1,'section_id':0,'lane_id':-1,'s':2.,'transform':tr})()
class W:
    def __init__(self, ep, town):self.id=ep;self.map=CM(town)
    def get_map(self):return self.map
    def get_actors(self):return A()
class Client2:
    def __init__(self):
        self.active=W(1,'Town15');self.pending=[];self.seq=1
    def load_world(self,town):
        self.seq+=1
        self.pending=[self.active, W(self.seq,town), W(self.seq,town)]
        return self.active
    def get_world(self):
        if self.pending:
            if len(self.pending)>1:
                world=self.pending.pop(0)
            else:world=self.pending[0]
            if world.id==self.seq:self.active=world
        return self.active
class C:
    class LaneType:
        Driving=1;Any=2
    class Location:
        def __init__(self,x,y,z):self.x=x;self.y=y;self.z=z

r={'road_id':1,'section_id':0,'lane_id':-1,'s':2.,'x':3.,'y':5.,'z':12.,'yaw_deg':45.}
records=[(0,{'record_index':0,'scene_id':'one','snapshot_sha256':'a'}, {'town':'Town12','ego_route':[r]}),
         (1,{'record_index':1,'scene_id':'two','snapshot_sha256':'b'}, {'town':'Town11','ego_route':[r]})]
expected={'Town12':GOOD,'Town11':GOOD}
with tempfile.TemporaryDirectory() as d:
    out=Path(d)/'out'
    censusmod.run(C,Client2(),records,expected,{'mock':True},out,50)
    summ=json.loads((out/'summary.json').read_text())
    assert summ['records_scanned']==2 and summ['unresolved_points']==0
    assert summ['point_status_counts']=={'verified_xodr_and_exact':2}
    assert summ['towns']['Town12']['status_counts']=={'verified_xodr_and_exact':1}
    assert summ['towns']['Town11']['status_counts']=={'verified_xodr_and_exact':1}
    assert (out/'map_transition_Town11.json').exists()
print('TEST_CENSUS_TWO_TOWN_REAL_DRIVER_STALE_WORLD_TRANSITION=PASS')
