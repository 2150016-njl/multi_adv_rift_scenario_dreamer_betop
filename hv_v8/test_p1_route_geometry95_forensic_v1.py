#!/usr/bin/env python3
"""Offline contract for diagnostic classifications, not a CARLA PASS."""
from pathlib import Path
import importlib.util
import json
import tempfile
from types import SimpleNamespace as N

spec=importlib.util.spec_from_file_location('forensic',str(Path(__file__).with_name('p1_route_geometry95_forensic_v1.py')))
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class Vec:
    def __init__(self,x,y,z=0.): self.x=x;self.y=y;self.z=z
class Rotation:
    def __init__(self,yaw): self.yaw=yaw
class WP:
    def __init__(self,road,section,lane,s,x,y,z,yaw):
        self.road_id=road;self.section_id=section;self.lane_id=lane;self.s=s
        self.transform=N(location=Vec(x,y,z),rotation=Rotation(yaw))
class Carla:
    class LaneType:
        Driving='Driving'; Any='Any'
    Location=Vec

base={'x':3.,'y':5.,'z':12.,'yaw_deg':45.,'road_id':269,'section_id':0,'lane_id':1,'s':13.}
w=WP(269,0,1,13.,3.,5.,12.,45.)
class Map:
    def __init__(self,xodr=w,exact=w,any_wp=None,proj=None):
        self.xodr=xodr;self.exact=exact;self.any_wp=any_wp;self.proj=proj
    def get_waypoint_xodr(self,*args):return self.xodr
    def get_waypoint(self,loc,project_to_road=False,lane_type='Driving'):
        return self.proj if project_to_road else (self.any_wp if lane_type=='Any' else self.exact)

def status(m, road_lengths=None):
    return mod.route_probe(Carla,m,base,road_lengths or {269:100.})['status']

assert status(Map())=='verified_xodr_and_exact'
assert status(Map(exact=None,any_wp=w,proj=w))=='xodr_source_verified_exact_none'
assert status(Map(xodr=None,exact=None))=='both_lookups_none'
assert status(Map(exact=WP(269,0,2,13,3,5,12,45)))=='xodr_source_verified_exact_other_lane'
assert status(Map(xodr=WP(269,0,1,13,25,25,12,45)))=='xodr_source_geometry_or_identity_mismatch'
assert status(Map(xodr=None, exact=w),{269:13.})=='verified_xodr_road_end_exception'
assert status(Map(xodr=None, exact=w),{269:100.})=='xodr_missing_not_verified_endpoint'
print('TEST_CLASSIFICATION_7_NEGATIVE_AND_POSITIVE_CONTRACTS_PASS')

# Test run() produces persistent record-wise, town-wise outputs on an in-memory
# mock world; never claim this mocks native-RIFT or CARLA collision semantics.
class Actors:
    def filter(self,p): return []
class World:
    def __init__(self,town,cm):self.town=town;self.cm=cm
    def get_map(self):return self.cm
    def get_actors(self):return Actors()
class MockMap(Map):
    def __init__(self,town):super().__init__();self.name='Carla/Maps/'+town+'/'+town; self.town=town
    def to_opendrive(self):return '<OpenDRIVE><road id="269" length="100" /></OpenDRIVE>'
class Client:
    def __init__(self):self.town='Town12';self.world=World(self.town,MockMap(self.town))
    def load_world(self,town):self.town=town;self.world=World(town,MockMap(town));return self.world
    def get_world(self):return self.world
client=Client()
expected={t:mod.digest(MockMap(t).to_opendrive().encode()) for t in ('Town11','Town12')}
records=[]
for i,town in enumerate(('Town12','Town11','Town12')):
    row={'record_index':i,'scene_id':'same_scene','snapshot_sha256':'sha-'+str(i)}
    sc={'town':town,'scene_id':'same_scene','ego_route':[dict(base)]}
    records.append((i,row,sc))
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)/'report'
    mod.run(Carla,client,records,expected,{'test':True},out,50)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='complete_query_census_not_p1_acceptance'
    assert summary['records_scanned']==3 and summary['points_scanned']==3
    assert summary['towns']['Town11']['records']==1
    assert summary['towns']['Town12']['records']==2
    assert summary['unresolved_points']==0
    assert len((out/'records.jsonl').read_text().splitlines())==3
    assert (out/'anomalies.jsonl').read_text()==''
print('TEST_3_RECORD_TWO_TOWN_END_TO_END_MOCK_REPORT_PASS')
print('OFFLINE_TEST_ONLY_NOT_CARLA_PASS')
