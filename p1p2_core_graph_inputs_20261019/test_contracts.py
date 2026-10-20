#!/usr/bin/env python3
"""Offline regression: 95 data checks, actual Town15 graph and map guard mocks."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

HERE=Path(__file__).resolve().parent

def import_file(name):
    sp=importlib.util.spec_from_file_location(name,str(HERE/(name+'.py')))
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

c=import_file('collect_existing');e=import_file('export_opendrive_once')

# Same input structure as frozen real snapshot manifest. Synthetic scenes are ONLY mock tests.
with TemporaryDirectory() as tmp:
    root=Path(tmp);base=root/'data/hv_init/v7';(base/'manifests').mkdir(parents=True)
    manifest=[]
    for idx in range(95):
        scene={'town':'Town12','ego_route':[
            {'road_id':1,'section_id':0,'lane_id':-1,'s':10.,'x':0.,'y':0.,'z':0.,'yaw_deg':0.,'is_junction':False},
            {'road_id':2,'section_id':0,'lane_id':-1,'s':0.,'x':1.,'y':0.,'z':0.,'yaw_deg':0.,'is_junction':True},
            {'road_id':2,'section_id':0,'lane_id':-1,'s':1.,'x':2.,'y':0.,'z':0.,'yaw_deg':0.,'is_junction':True}]}
        snap=base/('scene_%d.json'%idx)
        raw=json.dumps(scene).encode();snap.write_bytes(raw)
        manifest.append({'record_index':idx,'scene_id':'s%d'%idx,'episode_key':'episode',
                         'snapshot_path':str(snap),'snapshot_sha256':hashlib.sha256(raw).hexdigest()})
    mf=base/'manifests/p1_calibration95.jsonl';mm=base/'manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    mf.write_text(''.join(json.dumps(v)+'\n' for v in manifest));mm.write_text('{}')
    c.MANIFEST_SHA=c.digest(mf);c.MAP_MANIFEST_SHA=c.digest(mm)
    x=c.p1_extract(root,base)
    assert x['records_checked']==95
    assert len(x['all_key_change_occurrences'])==95
    assert len(x['unique_directed_key_changes'])==1
    assert len(x['same_lane_reverse_station_warnings'])==0
    # Corrupt immutable source: must fail closed instead of emitting an incomplete catalog
    (base/'scene_31.json').write_text('{}')
    try:c.p1_extract(root,base)
    except ValueError as exc:assert 'snapshot SHA mismatch' in str(exc)
    else:raise AssertionError('snapshot immutability broken')
print('TEST_P1_EXACT_95_TRANSITION_ENUMERATION_AND_SHA_GUARD=PASS')

# Test against actual user-uploaded pinned Town15 graph, if this local repository has the evidence.
real=Path('/mnt/data/current_review_src/data/hv_init/v7')
if (real/c.GRAPH).is_file() and (real/c.APP_DIR/'approaches.json').is_file():
    p=c.p2_extract(real)
    assert p['candidate_paths']==852
    assert p['unique_longitudinal_edges_required']==124
    assert p['longitudinal_edge_references']==5748
    print('TEST_REAL_TOWN15_GT_FREE_852_PATHS_124_EDGES=PASS')
else:
    print('TEST_REAL_TOWN15_INPUTS_NOT_LOCAL_SKIPPED')

class FakeMap:
    def __init__(self,name,raw):self.name='Carla/Maps/'+name+'/'+name;self.raw=raw
    def to_opendrive(self):return self.raw
class FakeWorld:
    def __init__(self,name,eid,raw,vehicles=0):self.id=eid;self.map=FakeMap(name,raw);self.vehicles=vehicles
    def get_map(self):return self.map
    def get_actors(self):return self
    def filter(self,name):return [1]*self.vehicles if name=='vehicle.*' else []
class FakeClient:
    def __init__(self,worlds):self.worlds=list(worlds);self.idx=0;self.loads=0
    def get_world(self):
        v=self.worlds[min(self.idx,len(self.worlds)-1)];self.idx+=1;return v
    def load_world(self,name):self.loads+=1
old=FakeWorld('Town15',1,'old');new=FakeWorld('Town12',2,'correct');wrong=FakeWorld('Town15',2,'old')
cl=FakeClient([old,wrong,new,new,new,new]);raw,eid,switched=e.verified_world(cl,'Town12',e.sha(b'correct'),True)
assert (raw,eid,switched)==(b'correct',2,True) and cl.loads==1
print('TEST_REAL_WORLDSWITCH_STALE_EPISODE_FENCE=PASS')
try:e.verified_world(FakeClient([FakeWorld('Town15',1,'old',vehicles=1)]),'Town12',e.sha(b'correct'),True)
except e.ContractError as err:assert 'nonempty' in str(err)
else:raise AssertionError('occupied world not blocked')
print('TEST_REFUSE_SWITCH_NONEMPTY_WORLD=PASS')
try:e.fetch_same_episode(FakeClient([FakeWorld('Town12',3,'WRONG')]),FakeWorld('Town12',3,'WRONG'),'Town12',e.sha(b'correct'))
except e.ContractError as err:assert 'SHA mismatch' in str(err)
else:raise AssertionError('map SHA not enforced')
print('TEST_WRONG_MAP_SHA_REJECTED=PASS')
print('ALL_TESTS_OFFLINE_NOT_REAL_CARLA_PASS')
