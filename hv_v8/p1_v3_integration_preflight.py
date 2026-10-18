#!/usr/bin/env python3
"""Offline integration regression for V3's population and spawn contracts."""
import ast
import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULE = HERE / 'p1_frozen_mainline_runner_v3.py'

spec = importlib.util.spec_from_file_location('p1frozenv3', MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

# The fixture is bundled in the first-party historical selector patch's
# original self-test: test the *actual* expected 48-KB source anchors.
original_selector = (HERE / 'p1_fixes/p1_target_actor_selection_contract_fix.py').read_text(encoding='utf-8')
fixture_tree = ast.parse(original_selector)
fixture_fn = next(node for node in fixture_tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'self_test')
sample_assigns = [node for node in ast.walk(fixture_fn)
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'sample' for t in node.targets)]
assert len(sample_assigns) == 1
sample = ast.literal_eval(sample_assigns[0].value)
sample += '''
def raw_sedan_spawn(world, carla, actor):
    spawned = world.try_spawn_actor(None, None)
    if spawned is None: return {"status":"rejected"}
    actual = spawned.get_transform()
    return {"status":"accepted", "actual": actual}

def scene_occupancy_spawn(client, world, carla, scene, target_actors):
    return {"status": "unknown"}
'''

patched, changes = runner.reconcile_population_and_pose(sample, ROOT)
assert runner._strict_population_state(patched), changes
assert runner._strict_pose_state(patched), changes
compile(patched, '<p1_v3_integrated_fixture>', 'exec')
second, second_changes = runner.reconcile_population_and_pose(patched, ROOT)
assert second == patched, second_changes

namespace = {'__name__': 'fixture'}
exec(compile(patched, '<p1_v3_integrated_fixture>', 'exec'), namespace)
selector = namespace['select_target_actors']
ids = list(range(100,118))
positive = {101,104,109,117}
scene = {
    'agents': [{'actor_id': actor_id, 'x': float(actor_id), 'y': 0.0} for actor_id in ids],
    'rift_labels': [{'actor_id': actor_id, 'is_interactive': actor_id in positive} for actor_id in ids],
    'diagnostics': {'rift_labeler': {'rift_candidate_actor_ids': sorted(positive)}},
}
targets, selection = selector({}, scene, scene['agents'])
assert len(targets) == 4 and selection['status'] == 'pass'
assert {str(x) for x in positive} == set(selection['selected_target_actor_ids'])
_, bad = selector({'target_actor_ids': [999]}, scene, scene['agents'])
assert bad['status'] == 'fail'
_, bad = selector({}, {'actors': [{'actor_id': 999, 'is_target': True}]}, [])
assert bad['status'] == 'fail'

with tempfile.TemporaryDirectory() as d:
    out = Path(d)
    (out / 'summary.json').write_text(json.dumps({
        'p1_pass': True, 'records': 1,
        'map_identity_all_pass': True,
        'coordinate_roundtrip_all_pass': True,
        'rift_reference_parity_all_pass': True,
        'target_selection_all_pass': True,
        'historical_native_rows': 0,
        'fixed_model3_raw_pose_spawn_rows': 0,
        'fixed_model3_raw_pose_spawn_accepted': 0,
        'rift_reference_parity_pass_count': 0,
        'historical_native_unknown_count': 0,
        'interaction_error_count': 0,
    }))
    ok, failures, _ = runner._strict_acceptance(out, 1)
    assert not ok and failures, failures

print('P1_V3_CONSOLIDATED_OFFLINE_REGRESSION_PASS')
print('strict target: 18 agents / 4 interactive labels => PASS')
print('manifest conflict, legacy actors fallback => FAIL CLOSED')
print('world snapshot pose contract present => PASS')
print('zero native rows with fabricated summary PASS => FAIL CLOSED')
print('integration idempotence => PASS')
print('CARLA was not contacted')
