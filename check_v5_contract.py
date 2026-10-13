#!/usr/bin/env python3
"""Static lint of V5 no-posthoc/no-online-RIFT interaction contract."""
import ast
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
v5src=ROOT/'scenario-dreamer'/'high_value_init'
wrapper=(v5src/'scene_generator_v5.py').read_text()
model=(v5src/'v5_model.py').read_text()
allsource='\n'.join(p.read_text() for p in v5src.glob('v5*.py'))+'\n'+wrapper
for forbidden in ('world.tick(', 'request_new_actor(', 'CBVs_recog(', 'set_transform(',
                  'set_location(', 'snap_to_lane(', 'set_autopilot(', 'get_rift_candidates('):
    if forbidden in allsource:
        raise AssertionError('Forbidden online code path: '+forbidden)
node=ast.parse(wrapper)
methods=[x for cls in node.body if isinstance(cls,ast.ClassDef) for x in cls.body
         if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=='generate']
assert len(methods)==1
samplings=[x for x in ast.walk(methods[0]) if isinstance(x,ast.Call)
           and isinstance(x.func,ast.Attribute) and x.func.attr=='sample_once']
assert len(samplings)==1, 'Wrapper must call sampler exactly once'
assert 'rift_labels_used_as_input' in wrapper
assert 'right_handed' in wrapper
assert 'num_generate_agents=2' in wrapper
assert 'No HD-Map/ego-route interaction candidates' in wrapper
print(json.dumps({'passed':True,'sampler_calls_in_online_wrapper':len(samplings),
                  'forbidden_runtime_mutators':0,'map_contract':'right_handed',
                  'fixed_target_count':2},indent=2))
