"""Offline regression for world readiness; no CARLA server required."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('v7_p3_native',ROOT/'scenario-dreamer/high_value_init/native_state_audit.py')
MOD=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

class FakeMap:
    def __init__(self,name,text):self.name,self.text=name,text
    def to_opendrive(self):return self.text

class FakeWorld:
    def __init__(self,name,text,id):self.map,self.id=FakeMap(name,text),id
    def get_map(self):return self.map

class FakeClient:
    def __init__(self,seq):self.seq=iter(seq);self.last=None;self.loads=[]
    def get_world(self):
        try:self.last=next(self.seq)
        except StopIteration:pass
        return self.last
    def load_world(self,town):self.loads.append(town)

class MapReadinessTests(unittest.TestCase):
    def engine(self,seq):
        engine=MOD.NativeAudit.__new__(MOD.NativeAudit)
        engine.client=FakeClient(seq);engine.load_maps=True
        engine.world=None;engine.grp=None;engine.resolution=1.
        engine.grp_class=lambda world_map,resolution:(world_map.name,resolution)
        return engine

    def test_transition_old_world_then_new_world(self):
        old=FakeWorld('Carla/Maps/Town04','old',1)
        new=FakeWorld('Carla/Maps/Town11/Town11','correct',2)
        e=self.engine([old,old,new])
        expected=hashlib.sha256(b'correct').hexdigest()
        e.prepare('Town11',expected_sha256=expected,readiness_timeout_s=2.)
        self.assertEqual(e.map_sha256,expected)
        self.assertEqual(e.client.loads,['Town11'])
        self.assertEqual(e.grp[0],new.map.name)
        self.assertEqual(e.prepare_trace[-1]['status'],'ready')

    def test_old_world_persists_fail_closed(self):
        old=FakeWorld('Carla/Maps/Town15','old',1)
        e=self.engine([old])
        with self.assertRaisesRegex(RuntimeError,'did not become ready'):
            e.prepare('Town11',expected_sha256=hashlib.sha256(b'expected').hexdigest(),
                      readiness_timeout_s=0.)
        self.assertIsNone(e.world)
        self.assertEqual(e.client.loads,['Town11'])

    def test_wrong_sha_never_certified(self):
        world=FakeWorld('Carla/Maps/Town11/Town11','wrong',1)
        e=self.engine([world])
        with self.assertRaisesRegex(RuntimeError,'did not become ready'):
            e.prepare('Town11',expected_sha256=hashlib.sha256(b'correct').hexdigest(),
                      readiness_timeout_s=0.)
        self.assertEqual(e.client.loads,[])
        self.assertIsNone(e.grp)

    def test_same_town_stale_sha_then_fresh_sha(self):
        stale=FakeWorld('Carla/Maps/Town11/Town11','stale',1)
        fresh=FakeWorld('Carla/Maps/Town11/Town11','fresh',2)
        e=self.engine([stale,stale,fresh])
        expected=hashlib.sha256(b'fresh').hexdigest()
        e.prepare('Town11',expected_sha256=expected,readiness_timeout_s=2.)
        self.assertEqual(e.map_sha256,expected)
        self.assertEqual(e.client.loads,[])
        self.assertTrue(any(x['status']=='sha_mismatch' for x in e.prepare_trace))

    def test_current_map_correct_reuse_validated(self):
        world=FakeWorld('Carla/Maps/Town11/Town11','correct',1)
        e=self.engine([world])
        expected=hashlib.sha256(b'correct').hexdigest()
        e.prepare('Town11',expected_sha256=expected)
        before=e.grp
        e.prepare('Town11',expected_sha256=expected)
        self.assertIs(e.grp,before)
        self.assertEqual(e.client.loads,[])

if __name__=='__main__':unittest.main()
