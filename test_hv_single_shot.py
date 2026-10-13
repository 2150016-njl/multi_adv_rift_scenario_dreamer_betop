"""CARLA-free regression tests for strict HV spawn-time contract.

Run: python -m unittest -v test_hv_single_shot.py
The tests exercise uploaded production module via stub CARLA/RIFT interfaces.
"""
import importlib.util
import ast
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch, Mock


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

class Rotation:
    def __init__(self, yaw=0.0):
        self.yaw = float(yaw)

class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

class Vector3D(Location):
    pass

class Actor:
    def __init__(self, actor_id, transform, model="vehicle.tesla.model3"):
        self.id = actor_id
        self.type_id = model
        self.transform = transform
        self.velocity = Vector3D()
        self.is_alive = True
        self.attributes = {"role_name": "generated_hv"}
        self.destroy_calls = 0
    def set_target_velocity(self, v):
        self.velocity = v
    def get_transform(self):
        return self.transform
    def get_velocity(self):
        return self.velocity
    def destroy(self):
        self.destroy_calls += 1
        self.is_alive = False

class World:
    def __init__(self):
        self.ticks = 0
    def tick(self):
        self.ticks += 1
        raise AssertionError("HV initializer must not tick world")
    def get_blueprint_library(self):
        return self
    def find(self, model):
        return types.SimpleNamespace(id=model)

class Provider:
    actors = {}
    spawn_calls = []
    register_calls = []
    cleanup_calls = []
    fail_index = None
    fallback_model = None
    fail_registration = False
    cache_refresh_calls = 0
    locations = {}
    @classmethod
    def reset(cls):
        cls.actors = {}
        cls.spawn_calls = []
        cls.register_calls = []
        cls.cleanup_calls = []
        cls.fail_index = None
        cls.fallback_model = None
        cls.fail_registration = False
        cls.cache_refresh_calls = 0
        cls.locations = {}
    @classmethod
    def request_new_actor(cls, model, transform, **kwargs):
        idx = len(cls.spawn_calls)
        cls.spawn_calls.append((model, transform, kwargs))
        if cls.fail_index == idx:
            raise RuntimeError("synthetic CARLA collision")
        actor = Actor(100+idx,transform,cls.fallback_model or model)
        cls.actors[actor.id] = actor
        cls.locations[actor.id] = None   # registered but not cached, like real provider
        return actor
    @classmethod
    def set_scenario_actors(cls,ego, actors):
        cls.register_calls.append([a.id for a in actors])
        if cls.fail_registration:
            raise RuntimeError("synthetic registration failure")
    @classmethod
    def remove_actor_by_id(cls, actor_id):
        cls.cleanup_calls.append(actor_id)
        cls.actors[actor_id].destroy()
        del cls.actors[actor_id]
        cls.locations.pop(actor_id, None)
    @classmethod
    def get_actor_by_id(cls, actor_id):
        return cls.actors.get(actor_id)
    @classmethod
    def on_carla_tick(cls):
        # This is cache-only; a CARLA world tick is deliberately forbidden.
        cls.cache_refresh_calls += 1
        for actor in cls.actors.values():
            cls.locations[actor.id] = actor.get_transform().location

class Generator:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0
    def generate(self, snapshot):
        self.calls += 1
        if isinstance(self.proposal, BaseException):
            raise self.proposal
        return self.proposal

# Install fake external dependency modules before importing production source.
def install_stubs():
    carla = types.ModuleType("carla")
    carla.Location,carla.Rotation,carla.Transform,carla.Vector3D,carla.Actor = (Location,Rotation,Transform,Vector3D,Actor)
    carla.Vehicle = Actor
    carla.LaneType = types.SimpleNamespace(Driving=1)
    sys.modules["carla"] = carla
    root=types.ModuleType("rift");root.__path__=[];sys.modules["rift"]=root
    cbv=types.ModuleType("rift.cbv");cbv.__path__=[];sys.modules[cbv.__name__]=cbv
    init=types.ModuleType("rift.cbv.initialization");init.__path__=[];sys.modules[init.__name__]=init
    ss=types.ModuleType("rift.cbv.initialization.hv_scene_snapshot")
    class Snapshotter:
        def __init__(self,**kwargs): pass
        def build(self,*args,**kwargs):
            raise AssertionError("post-spawn RIFT audit forbidden")
    ss.HighValueSceneSnapshotter=Snapshotter;sys.modules[ss.__name__]=ss
    sc=types.ModuleType("rift.scenario");sc.__path__=[];sys.modules[sc.__name__]=sc
    tools=types.ModuleType("rift.scenario.tools");tools.__path__=[];sys.modules[tools.__name__]=tools
    provider=types.ModuleType("rift.scenario.tools.carla_data_provider")
    provider.CarlaDataProvider=Provider;sys.modules[provider.__name__]=provider
    err=types.ModuleType("rift.scenario.tools.exception")
    class SpawnRuntimeError(RuntimeError):pass
    err.SpawnRuntimeError=SpawnRuntimeError;sys.modules[err.__name__]=err

install_stubs()
TEST_ROOT=Path(__file__).resolve().parent
# ALWAYS test the deployed module if this is a project checkout;
# do not accidentally test a stale *_strict.py backup in project root.
DEPLOYED=TEST_ROOT/"RIFT/rift/cbv/initialization/hv_online_generation.py"
FILE=(DEPLOYED if DEPLOYED.exists()
      else TEST_ROOT/"hv_online_generation.py" if (TEST_ROOT/"hv_online_generation.py").exists()
      else TEST_ROOT/"hv_online_generation_strict.py")
spec=importlib.util.spec_from_file_location("hv_single_shot_under_test",FILE)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Initializer=module.HighValueGeneratedSceneInitializer


class TestSingleShotHV(unittest.TestCase):
    def setUp(self):
        Provider.reset()
        self.world=World()
        self.obj=Initializer({
            "high_value_init_generation":False,
            "hv_num_generate_agents":2,
            "hv_generation_max_attempts":1,
            "hv_generation_mode":"full_fov_inpainting",
            "hv_generation_partial_context":False,
            "hv_generation_actor_model":"vehicle.tesla.model3",
            "hv_generation_lifecycle_telemetry":False,
        },self.world)
        self.obj.enabled=True
        self.obj._road_waypoint=lambda x,y,z:types.SimpleNamespace(
            transform=Transform(Location(x+1,y-1,0),Rotation(yaw=37)),
            road_id=1,lane_id=-1)
        self.obj._nearest_existing_vehicles=lambda location:[]
        self.ego=types.SimpleNamespace(id=1)
        self.snapshot={"scene_id":"scene0","ego":{"x":100.0,"y":200.0,"z":0.0,"yaw_deg":90.0},"diagnostics":{},"agents":[]}
        self.states=[[12.0, 2.0, .25, 5.0, 10.0, 4.0, 1.0],[20.0,-3.0,-.5,7.0,1.0,.8,1.0]]
        self.obj.generator=Generator({"generated_agent_states":self.states})

    def run_once(self):
        return self.obj._run_generation_attempts(
            self.snapshot,self.ego,None,None,None,[],[],0)

    def test_exactly_one_generation_two_spawn_no_tick_no_snap(self):
        result,proposal,audit=self.run_once()
        self.assertTrue(result["success"])
        self.assertEqual(result["generation_max_attempts"],1)
        self.assertEqual(result["configured_generation_max_attempts"],1)
        self.assertEqual(self.obj.generator.calls,1)
        self.assertEqual(len(Provider.spawn_calls),2)
        self.assertIsNone(audit)
        self.assertEqual(self.world.ticks,0)
        self.assertEqual(Provider.cache_refresh_calls,1)
        self.assertIsNotNone(Provider.locations[100])
        self.assertIsNotNone(Provider.locations[101])
        self.assertEqual(Provider.register_calls,[[100,101]])
        self.assertEqual(result["rift_audit_status"],"deferred_to_normal_cbvs_recog")
        for i,(model,transform,kwargs) in enumerate(Provider.spawn_calls):
            s=self.states[i]
            self.assertEqual(model,"vehicle.tesla.model3")
            self.assertFalse(kwargs["tick"])
            self.assertEqual(kwargs["max_try"],1)
            # yaw 90 degrees: xworld=100-local_y, yworld=200+local_x
            self.assertAlmostEqual(transform.location.x,100-s[1],places=9)
            self.assertAlmostEqual(transform.location.y,200+s[0],places=9)
            self.assertAlmostEqual(transform.rotation.yaw,90+math.degrees(s[2]),places=9)
            self.assertAlmostEqual(transform.location.z,.3,places=9)
            self.assertEqual(result["spawn_attempts"][i]["generated_local_state"],s)
        # Both requested speeds applied directly, not clamped.
        self.assertAlmostEqual(math.hypot(Provider.actors[100].velocity.x,Provider.actors[100].velocity.y),5)
        self.assertAlmostEqual(math.hypot(Provider.actors[101].velocity.x,Provider.actors[101].velocity.y),7)

    def test_stale_pre_tick_actor_pose_is_diagnostic_only(self):
        # Asynchronous CARLA often returns a not-yet-updated actor pose before
        # the next simulation frame. It must not affect the spawn request.
        original = self.obj._spawn_acceptance_fidelity
        def unreliable_observer(actor, attempt):
            return {
                "spawn_acceptance_observation_available": True,
                "spawn_acceptance_observation_phase": "pre_tick_unconfirmed",
                "spawn_acceptance_xy_error_m": 48.0,
                "spawn_acceptance_yaw_error_deg": 90.0,
            }
        self.obj._spawn_acceptance_fidelity = unreliable_observer
        result, _, _ = self.run_once()
        self.assertTrue(result["success"])
        self.assertEqual(self.obj.generator.calls, 1)
        self.assertEqual(len(Provider.spawn_calls), 2)
        self.assertEqual(Provider.cache_refresh_calls, 1)
        self.assertEqual(self.world.ticks, 0)
        self.obj._spawn_acceptance_fidelity = original

    def test_spawn_second_failure_cleans_first_no_retry(self):
        Provider.fail_index=1
        result,_,audit=self.run_once()
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"],"spawn")
        self.assertEqual(self.obj.generator.calls,1)
        self.assertEqual(len(Provider.spawn_calls),2)
        self.assertEqual(Provider.cleanup_calls,[100])
        self.assertEqual(Provider.cache_refresh_calls,0)
        self.assertFalse(Provider.actors)
        self.assertEqual(self.world.ticks,0)
        self.assertIsNone(audit)

    def test_three_states_rejected_before_spawn(self):
        self.obj.generator=Generator({"generated_agent_states":self.states+[self.states[0]]})
        result,_,_=self.run_once()
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"],"proposal_validation")
        self.assertEqual(self.obj.generator.calls,1)
        self.assertEqual(len(Provider.spawn_calls),0)

    def test_nan_rejected_without_repair(self):
        self.obj.generator=Generator({"generated_agent_states":[[float('nan'),2,0,5,4,2,1],self.states[1]]})
        result,_,_=self.run_once()
        self.assertFalse(result["success"])
        self.assertEqual(self.obj.generator.calls,1)
        self.assertEqual(len(Provider.spawn_calls),0)

    def test_generator_exception_has_no_retry(self):
        self.obj.generator=Generator(RuntimeError("sampling failed"))
        result,_,_=self.run_once()
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"],"generator")
        self.assertEqual(self.obj.generator.calls,1)
        self.assertEqual(len(Provider.spawn_calls),0)

    def test_second_invocation_never_generates(self):
        self.run_once()
        result,proposal,audit=self.run_once()
        self.assertFalse(result["success"])
        self.assertTrue(result["skipped"])
        self.assertEqual(result["failure_stage"],"already_consumed")
        self.assertEqual(self.obj.generator.calls,1)
        self.assertEqual(len(Provider.spawn_calls),2)

    def test_bp_fallback_rejected_and_destroyed(self):
        Provider.fallback_model="vehicle.audi.tt"
        result,_,_=self.run_once()
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"],"spawn")
        self.assertEqual(Provider.cleanup_calls,[100])
        self.assertEqual(len(Provider.spawn_calls),1)
        self.assertEqual(self.obj.generator.calls,1)

    def test_registration_failure_is_atomic(self):
        Provider.fail_registration = True
        result,_,_=self.run_once()
        self.assertFalse(result["success"])
        self.assertEqual(self.obj.generator.calls,1)
        self.assertEqual(len(Provider.spawn_calls),2)
        self.assertEqual(Provider.cleanup_calls,[100,101])
        self.assertEqual(Provider.cache_refresh_calls,0)
        self.assertFalse(Provider.actors)

    def test_initialize_without_gate_never_calls_pre_generation_recognition(self):
        observed=[]
        def build(**kwargs):
            observed.append(kwargs["include_rift_labels"])
            self.assertFalse(kwargs["include_rift_labels"])
            return self.snapshot
        self.obj.snapshotter=types.SimpleNamespace(build=build)
        self.obj.gate_on_existing_rift_positive=False
        result=self.obj.initialize(self.ego,None,None,None,[])
        self.assertTrue(result["success"])
        self.assertEqual(observed,[False])
        self.assertEqual(Provider.cache_refresh_calls,1)
        self.assertEqual(self.obj.generator.calls,1)
        self.assertEqual(self.world.ticks,0)

    def test_strict_configuration_guards(self):
        base={"high_value_init_generation":True,"hv_generation_mode":"full_fov_inpainting","hv_generation_partial_context":False}
        cases=[
            {"hv_generation_actor_model":"vehicle.*"},
            {"hv_generation_max_attempts":8},
            {"hv_generation_rolling":True},
            {"hv_generation_mode":"inpainting","hv_generation_partial_context":True},
            {"hv_generation_clamp_generated_states":True},
            {"sd_background_replace_existing":True},
            {"hv_num_generate_agents":3},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                Initializer({**base,**overrides},self.world)

class TestRawInverseNormalization(unittest.TestCase):
    def test_raw_decoder_inverse_does_not_clip(self):
        import torch
        deployed=TEST_ROOT/"scenario-dreamer/high_value_init/high_value_ldm.py"
        path=(deployed if deployed.exists() else TEST_ROOT/"high_value_ldm_strict.py")
        tree=ast.parse(path.read_text())
        model_class=next(n for n in tree.body if isinstance(n,ast.ClassDef)
                         and n.name=="RouteConditionedHighValueLDM")
        method=next(n for n in model_class.body if isinstance(n,ast.FunctionDef)
                    and n.name=="_unnormalize_agents_raw_for_spawn")
        # Execute only the mathematical inverse function; torch-geometric and
        # CARLA are not installed in this unit-test runtime.
        namespace={"torch":torch}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),namespace)
        cfg=types.SimpleNamespace(fov=64.,min_speed=0.,max_speed=29.19,
                                  min_length=.09,max_length=19.77,
                                  min_width=.14,max_width=7.20)
        owner=types.SimpleNamespace(cfg_ldm=types.SimpleNamespace(dataset=cfg))
        decoder=namespace["_unnormalize_agents_raw_for_spawn"]
        values=torch.tensor([[1.5,-1.5,1.5,.5,-.5,-1.5,1.5],
                             [-1.,1.,-1.,.0,1.,-1.,1.]],dtype=torch.float32)
        source=values.clone()
        result=decoder(owner,values)
        self.assertEqual(result.shape,(2,7))
        self.assertAlmostEqual(float(result[0,0]),48.,places=5)
        self.assertAlmostEqual(float(result[0,1]),-48.,places=5)
        self.assertGreater(float(result[0,2]),29.19)
        self.assertLess(float(result[0,5]),.09)
        self.assertGreater(float(result[0,6]),7.20)
        self.assertTrue(torch.equal(values,source))
        self.assertAlmostEqual(float(result[1,0]),-32.,places=5)
        self.assertAlmostEqual(float(result[1,1]),32.,places=5)

if __name__=="__main__":
    unittest.main(verbosity=2)
