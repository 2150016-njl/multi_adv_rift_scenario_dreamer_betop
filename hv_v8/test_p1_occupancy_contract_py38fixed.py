#!/usr/bin/env python3
"""Focused mock contract tests. These are NOT CARLA tests."""
import ast
import json
import pathlib
import sys
import types

DIR=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(DIR))
from p1_occupancy_contract_repair import patched_source, PARENT_SHA
import hashlib
import importlib.util


def fake_import(original):
    patched,_=patched_source(original)
    # Preserve the frozen module's future annotations when isolating its function.
    # Omitting this directive makes Python 3.8 EVALUATE dict[str, Any] and
    # list[dict[str, Any]] at function-definition time (TypeError).
    t=ast.parse(patched, feature_version=(3, 8))
    future=[n for n in t.body if isinstance(n, ast.ImportFrom)
            and n.module=='__future__' and any(x.name=='annotations' for x in n.names)]
    nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='scene_occupancy_spawn']
    if len(future) != 1 or len(nodes) != 1:
        raise RuntimeError('missing source future-annotations contract or occupancy function')
    comp=compile(ast.Module(body=future+nodes,type_ignores=[]),'mock_occupancy','exec')
    # Assert the isolated mock has the same annotation semantics as the real
    # Python 3.8 module; a newer local Python must not conceal a 3.8 error.
    import __future__
    if not (comp.co_flags & __future__.annotations.compiler_flag):
        raise RuntimeError('future annotations not active in extracted mock')
    ns={'Any':object, 'carla_transform':lambda c,a: dict(a)}
    exec(comp,ns)
    return ns['scene_occupancy_spawn']

class Reply:
    def __init__(self,error='',actor_id=None):self.error=error;self.actor_id=actor_id
class Spawn:
    def __init__(self,bp,tr):self.bp=bp;self.tr=tr
class Destroy:
    def __init__(self,id):self.id=id
class BP:
    def find(self,name):
        if not name.startswith('vehicle.'):
            raise ValueError('bad BP '+name)
        return name
class Alist(list):
    def filter(self,pat):
        if pat.startswith('vehicle.') or pat.startswith('walker.'):
            return list(self)
        return []
class World:
    def __init__(self):self.live={};self.serial=100000
    def get_blueprint_library(self):return BP()
    def get_actors(self):return Alist([types.SimpleNamespace(id=i) for i in self.live])
class Client:
    def __init__(self,world,ego_reject=False,target2_reject=False,cleanup_fail=False):
        self.w=world;self.ego_reject=ego_reject;self.target2_reject=target2_reject;self.cleanup_fail=cleanup_fail
        self.target_first_live_on_second=None
        self.commands=[]
    def apply_batch_sync(self,cmds,tick):
        assert tick is False
        res=[]
        for c in cmds:
            self.commands.append(c)
            if isinstance(c,Destroy):
                if self.cleanup_fail: res.append(Reply('cleanup_error'))
                else:self.w.live.pop(c.id,None);res.append(Reply())
            else:
                aid=str(c.tr.get('actor_id'))
                targets=[(x,v) for x,v in self.w.live.items() if v[0]=='vehicle.tesla.model3']
                if aid=='11143' and c.bp=='vehicle.tesla.model3':
                    self.target_first_live_on_second=bool(targets)
                if (self.ego_reject and aid=='__ego__') or (self.target2_reject and aid=='11143' and c.bp=='vehicle.tesla.model3'):
                    res.append(Reply('Spawn failed because of collision at spawn position'))
                else:
                    self.w.serial+=1; n=self.w.serial;self.w.live[n]=(c.bp,aid);res.append(Reply(actor_id=n))
        return res

class CARLA:
    class command:
        SpawnActor=Spawn
        DestroyActor=Destroy


def cases(fn,snapshot):
    agents=snapshot['agents']
    labels=snapshot['rift_labels']
    positive={str(l['actor_id']) for l in labels if l['is_interactive']}
    targets=[dict(a) for a in agents if str(a['actor_id']) in positive]
    # Historical original scene: 1 ego + 6 background; ego rejection stops target testing.
    w=World();c=Client(w,ego_reject=True)
    d=fn(c,w,CARLA,snapshot,targets)
    assert d['status']=='unknown' and d['reason']=='background_scene_replay_incomplete',d
    assert d['expected_background_count']==7 and d['expected_target_count']==2,d
    assert len(d['background'])==7 and len(d['targets'])==0,d
    assert len(w.live)==0 and d['cleanup_verified'] is True,d
    assert set(d['expected_background_actor_ids']) == {'__ego__','11133','11134','11138','11139','11140','11142'}
    print('MOCK_RECORD100_EGO_REJECT_BACKGROUND_7_AND_CLEANUP=PASS')
    # Joint: 1st target must remain when 2nd target is attempted.
    w=World();c=Client(w,target2_reject=True)
    d=fn(c,w,CARLA,snapshot,targets)
    assert d['status']=='complete' and d['cleanup_verified'] is True,d
    assert d['joint_model3_spawn_accepted'] is False and c.target_first_live_on_second is True,d
    assert [x['status'] for x in d['targets']]==['accepted_in_scene','rejected_in_scene'],d
    assert not w.live
    print('MOCK_JOINT_FIRST_HELD_UNTIL_SECOND_REJECTS=PASS')
    # Both accepted (this still doesn't prove pose or full-scene physics).
    w=World();c=Client(w)
    d=fn(c,w,CARLA,snapshot,targets)
    assert d['status']=='complete' and d['joint_model3_spawn_accepted'] is True,d
    assert d['full_scene_physical_feasibility']=='unknown' and all(t['post_spawn_pose_verified'] is False for t in d['targets'])
    assert not w.live
    print('MOCK_SPAWN_ACCEPTED_NOT_EQUALS_PHYSICALLY_VERIFIED=PASS')
    # Missing schema; no spawn.
    sc=dict(snapshot);sc.pop('agents')
    w=World();c=Client(w);d=fn(c,w,CARLA,sc,targets)
    assert d['status']=='unknown' and not c.commands and not w.live
    print('MOCK_MISSING_AGENTS_NO_FALLBACK=PASS')
    # Wrong target selection; no spawn.
    w=World();c=Client(w);d=fn(c,w,CARLA,snapshot,targets[:1])
    assert d['status']=='unknown' and d['reason']=='target_selection_not_equal_rift_positive' and not c.commands
    print('MOCK_INCOMPLETE_TARGET_SET_REJECTED=PASS')
    # Bad label/agent IDs; no spawn.
    sc=dict(snapshot);sc['rift_labels']=[dict(x) for x in labels];sc['rift_labels'][0]['actor_id']='BAD'
    w=World();c=Client(w);d=fn(c,w,CARLA,sc,targets)
    assert d['status']=='unknown' and d['reason']=='agents_labels_identity_conflict' and not c.commands
    print('MOCK_AGENTS_LABELS_ID_DRIFT_REJECTED=PASS')
    # Failed cleanup must downgrade status despite all spawns accepted.
    w=World();c=Client(w,cleanup_fail=True)
    d=fn(c,w,CARLA,snapshot,targets)
    assert d['status']=='unknown' and d['cleanup_verified'] is False,d
    print('MOCK_CLEANUP_FAILURE_DEGRADES_TO_UNKNOWN=PASS')
    # Unexpected existing actors; never kill them.
    w=World();w.live[55]=('vehicle.other','other');c=Client(w)
    d=fn(c,w,CARLA,snapshot,targets)
    assert d['status']=='unknown' and d['reason']=='nonempty_world_refuse_spawns' and not c.commands
    assert 55 in w.live
    print('MOCK_NONEMPTY_WORLD_REFUSED_NO_FOREIGN_DELETION=PASS')
    print('P1_OCCUPANCY_OFFLINE_CONTRACT_8_CASES_PASS_NOT_CARLA')


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--parent-source',required=True)
    ap.add_argument('--snapshot',required=True)
    a=ap.parse_args()
    p=pathlib.Path(a.parent_source)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==PARENT_SHA
    scene=json.loads(pathlib.Path(a.snapshot).read_text())
    fn=fake_import(p.read_text())
    cases(fn,scene)

if __name__=='__main__':main()
