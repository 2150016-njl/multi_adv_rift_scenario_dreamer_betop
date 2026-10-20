#!/usr/bin/env python3
# P1 occupancy schema and joint lifetime repair. Does not change poses or labels.
"""Guarded, SHA-locked occupancy repair for a NEW frozen P1 capsule.

Dry-run by default. Use --create to create a new, never-overwritten capsule.
Never changes live hv_v8/p1_native_audit.py or existing frozen capsule.
"""
import argparse
import ast
import datetime
import difflib
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys

PARENT_SHA = '34178e62ec0327768135d1882a0d8fd5d78ed12260cc2db1d176e4762f67d7e5'
PARENT_CAPSULE = 'p1_frozen_capsule_route140_20261018_141003_2150'
MARKER = 'P1_OCCUPANCY_AGENTS_JOINT_V1'

# This function may NOT count a spawn RPC accepted as physically verified.
# Its 'complete' is an audit-attempt completion indicator; the independent
# physical feasibility gate remains separate.
REPLACEMENT = '''def scene_occupancy_spawn(
    client: Any,
    world: Any,
    carla: Any,
    scene: dict[str, Any],
    target_actors: list[dict[str, Any]],
) -> dict[str, Any]:
    # P1_OCCUPANCY_AGENTS_JOINT_V1
    # Diagnostically replay exact snapshot states. NO pose correction,
    # retry, transform projection, world.tick(), or native-label fallback.
    # Never treat 'complete' as physical scene or Model3 feasibility PASS.
    result = {
        'schema_version': 'p1_occupancy_agents_joint_v1',
        'status': 'unknown',
        'reason': None,
        'source': 'snapshot.agents + snapshot.rift_labels',
        'background': [], 'targets': [],
        'expected_background_actor_ids': [],
        'expected_target_actor_ids': [],
        'expected_background_count': None,
        'expected_target_count': None,
        'joint_model3_spawn_accepted': None,
        'full_scene_physical_feasibility': 'unknown',
        'source_pose_modified': False,
        'world_tick_called': False,
        'cleanup_verified': None,
    }
    created_ids = []
    ownership_unknown = False
    def fail(reason):
        result['status'] = 'unknown'
        result['reason'] = reason
        return result

    # Validate original source completeness BEFORE spawning anything.
    if not isinstance(scene, dict) or not isinstance(scene.get('ego'), dict):
        return fail('ego_missing_or_invalid')
    ego = scene['ego']
    if not str(ego.get('type_id', '')).startswith('vehicle.'):
        return fail('ego_vehicle_blueprint_missing')
    agents = scene.get('agents')
    labels = scene.get('rift_labels')
    if not isinstance(agents, list) or not isinstance(labels, list):
        return fail('agents_or_rift_labels_missing')
    if not agents or not labels or any(not isinstance(a, dict) for a in agents) or any(not isinstance(l, dict) for l in labels):
        return fail('agents_or_labels_invalid_or_empty')
    if any(a.get('actor_id') is None or not str(a.get('type_id', '')).startswith('vehicle.') for a in agents):
        return fail('agent_id_or_vehicle_type_invalid')
    if any(l.get('actor_id') is None or not isinstance(l.get('is_interactive'), bool) for l in labels):
        return fail('rift_label_id_or_boolean_invalid')
    ids = [str(a['actor_id']) for a in agents]
    label_ids = [str(l['actor_id']) for l in labels]
    if len(ids) != len(set(ids)) or len(label_ids) != len(set(label_ids)) or set(ids) != set(label_ids):
        return fail('agents_labels_identity_conflict')
    positive_ids = {str(l['actor_id']) for l in labels if l['is_interactive']}
    received_ids = [str(t.get('actor_id')) for t in target_actors]
    if (not positive_ids or len(received_ids) != len(set(received_ids))
            or set(received_ids) != positive_ids):
        return fail('target_selection_not_equal_rift_positive')
    if ego.get('actor_id') is not None and str(ego['actor_id']) in set(ids):
        return fail('ego_duplicated_in_agents')

    ego_row = dict(ego)
    ego_row['actor_id'] = '__ego__'
    backgrounds = [ego_row] + [a for a in agents if str(a['actor_id']) not in positive_ids]
    ordered_targets = [a for a in agents if str(a['actor_id']) in positive_ids]
    result['expected_background_actor_ids'] = [str(a['actor_id']) for a in backgrounds]
    result['expected_target_actor_ids'] = [str(a['actor_id']) for a in ordered_targets]
    result['expected_background_count'] = len(backgrounds)
    result['expected_target_count'] = len(ordered_targets)
    # For record 100: exactly 7 background (ego + 6 agents) and 2 targets.
    if len(backgrounds) + len(ordered_targets) != len(agents) + 1:
        return fail('population_count_inconsistent')

    try:
        present = world.get_actors()
        if len(present.filter('vehicle.*')) or len(present.filter('walker.pedestrian.*')):
            return fail('nonempty_world_refuse_spawns')
    except Exception as exc:
        return fail('world_actor_guard_unavailable:%s:%s' % (type(exc).__name__, exc))

    try:
        lib = world.get_blueprint_library()
        pending = []
        for a in backgrounds:
            aid = str(a['actor_id'])
            try:
                bp = lib.find(str(a['type_id']))
                tr = carla_transform(carla, a)
                pending.append((a, carla.command.SpawnActor(bp, tr)))
            except Exception as exc:
                result['background'].append({
                    'actor_id': aid, 'status': 'not_attempted',
                    'reason': 'background_blueprint_or_pose_invalid:%s:%s' % (type(exc).__name__, exc)})
        if len(pending) != len(backgrounds):
            return fail('background_source_cannot_be_replayed')

        replies = client.apply_batch_sync([cmd for a, cmd in pending], False)
        if len(replies) != len(pending):
            # Capture returned actor IDs, but unreported successes may exist.
            created_ids.extend(int(r.actor_id) for r in replies
                               if not getattr(r, 'error', None) and getattr(r, 'actor_id', None) is not None)
            ownership_unknown = True
            return fail('background_batch_response_count_mismatch')
        for (a, _), reply in zip(pending, replies):
            aid = str(a['actor_id'])
            err = getattr(reply, 'error', None)
            new_id = getattr(reply, 'actor_id', None)
            if err or new_id is None:
                result['background'].append({'actor_id': aid, 'status': 'spawn_rejected',
                                             'error': str(err or 'missing_spawned_id')})
            else:
                created_ids.append(int(new_id))
                result['background'].append({'actor_id': aid, 'status': 'spawned',
                                             'spawned_actor_id': int(new_id)})
        if any(r['status'] != 'spawned' for r in result['background']):
            return fail('background_scene_replay_incomplete')

        bp = lib.find('vehicle.tesla.model3')
        # Keep previously spawned targets alive while subsequent targets spawn.
        # This tests joint occupancy rather than isolated success.
        for a in ordered_targets:
            aid = str(a['actor_id'])
            try:
                response = client.apply_batch_sync([
                    carla.command.SpawnActor(bp, carla_transform(carla, a))
                ], False)
                if len(response) != 1:
                    created_ids.extend(int(r.actor_id) for r in response
                                       if not getattr(r, 'error', None) and getattr(r, 'actor_id', None) is not None)
                    ownership_unknown = True
                    result['targets'].append({'actor_id': aid, 'status': 'unknown',
                                              'reason': 'target_response_count_mismatch'})
                    return fail('target_response_count_mismatch')
                reply = response[0]
                err = getattr(reply, 'error', None)
                new_id = getattr(reply, 'actor_id', None)
                if err or new_id is None:
                    result['targets'].append({'actor_id': aid, 'status': 'rejected_in_scene',
                                              'error': str(err or 'missing_spawned_id'),
                                              'raw_pose_correction': False})
                else:
                    created_ids.append(int(new_id))
                    result['targets'].append({'actor_id': aid, 'status': 'accepted_in_scene',
                                              'spawned_actor_id': int(new_id),
                                              'raw_pose_correction': False,
                                              'post_spawn_pose_verified': False})
            except Exception as exc:
                result['targets'].append({'actor_id': aid, 'status': 'unknown',
                                          'reason': 'target_rpc_exception:%s:%s' % (type(exc).__name__, exc)})
                return fail('target_rpc_exception')
        result['joint_model3_spawn_accepted'] = all(
            t['status'] == 'accepted_in_scene' for t in result['targets'])
        result['status'] = 'complete'
        result['reason'] = 'all_original_pose_spawn_requests_have_definitive_outcomes'
        return result
    except Exception as exc:
        return fail('occupancy_rpc_or_blueprint_error:%s:%s' % (type(exc).__name__, exc))
    finally:
        # Synchronous, acknowledged cleanup of only the actors created here.
        # If cleanup cannot be independently verified, never report complete.
        if created_ids:
            try:
                replies = client.apply_batch_sync([
                    carla.command.DestroyActor(i) for i in created_ids
                ], False)
                if len(replies) != len(created_ids) or any(getattr(x, 'error', None) for x in replies):
                    result['cleanup_verified'] = False
                    result['status'] = 'unknown'
                    result['reason'] = 'owned_actor_cleanup_rpc_failed'
                else:
                    remaining = {int(a.id) for a in world.get_actors()}
                    result['cleanup_verified'] = not bool(set(created_ids) & remaining)
                    if not result['cleanup_verified']:
                        result['status'] = 'unknown'
                        result['reason'] = 'owned_actors_remain_after_cleanup'
            except Exception as exc:
                result['cleanup_verified'] = False
                result['status'] = 'unknown'
                result['reason'] = 'owned_actor_cleanup_unverified:%s:%s' % (type(exc).__name__, exc)
        else:
            result['cleanup_verified'] = True
        if ownership_unknown:
            # Never mark cleanup proven when RPC response length was unexpected.
            result['cleanup_verified'] = False
            result['status'] = 'unknown'
            result['reason'] = 'incomplete_rpc_actor_ownership_evidence'
'''


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def patched_source(source):
    if MARKER in source:
        raise RuntimeError('already patched; refusing double patch')
    tree = ast.parse(source)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scene_occupancy_spawn']
    if len(matches) != 1:
        raise RuntimeError('cannot uniquely identify occupancy function')
    node = matches[0]
    lines = source.splitlines(keepends=True)
    body = ''.join(lines[node.lineno-1:node.end_lineno])
    if not re.search(r'scene\.get\(\s*[\'"]actors[\'"]\s*,', body) or 'world.try_spawn_actor' not in body:
        raise RuntimeError('occupancy original source pattern differs; refuse patch')
    # Only the single function AST body is replaced, other logic untouched.
    updated = ''.join(lines[:node.lineno-1]) + REPLACEMENT + '\n' + ''.join(lines[node.end_lineno:])
    compile(updated, 'p1_new_frozen_audit.py', 'exec')
    node2 = [n for n in ast.parse(updated).body if isinstance(n, ast.FunctionDef) and n.name == 'scene_occupancy_spawn']
    if len(node2) != 1:
        raise RuntimeError('replacement AST not unique')
    return updated, body


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--create', action='store_true', help='create a NEW frozen capsule; default dry-run')
    args = p.parse_args()
    root = Path(args.root).resolve()
    parent = root / 'data/hv_init/v7' / PARENT_CAPSULE
    old = parent / 'hv_v8/p1_native_audit.py'
    if not old.is_file() or sha_file(old) != PARENT_SHA:
        raise RuntimeError('frozen parent SHA mismatch / missing: ' + str(old))
    old_bytes = old.read_bytes()
    src, oldfun = patched_source(old_bytes.decode('utf-8'))
    newsha = hashlib.sha256(src.encode('utf-8')).hexdigest()
    print('P1_PARENT_FROZEN_SHA256=' + PARENT_SHA)
    print('P1_OCCUPANCY_CANDIDATE_SHA256=' + newsha)
    print('P1_OCCUPANCY_PATCH_LINES_OLD=' + str(oldfun.count('\n')))
    print('P1_OCCUPANCY_PATCH_LINES_NEW=' + str(REPLACEMENT.count('\n')))
    print('P1_OCCUPANCY_PATCH_DRY_RUN_PASS (NOT CARLA PASS)')
    if not args.create:
        return
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    dest = parent.parent / ('p1_frozen_capsule_occupancy_joint_' + now)
    if dest.exists():
        raise RuntimeError('refusing to overwrite capsule')
    shutil.copytree(parent, dest)
    new_path = dest / 'hv_v8/p1_native_audit.py'
    new_path.write_text(src, encoding='utf-8')
    if sha_file(new_path) != newsha:
        raise RuntimeError('new frozen source write SHA mismatch')
    provp = dest / 'p1_source_provenance.json'
    parent_raw = (parent / 'p1_source_provenance.json').read_bytes()
    prov = json.loads(parent_raw)
    if prov.get('frozen_sha256') != PARENT_SHA:
        raise RuntimeError('parent provenance frozen SHA mismatch')
    (dest / 'p1_source_provenance.parent.json').write_bytes(parent_raw)
    prov['parent_frozen_sha256'] = PARENT_SHA
    prov['parent_source_provenance_sha256'] = hashlib.sha256(parent_raw).hexdigest()
    prov['frozen_sha256'] = newsha
    prov.setdefault('reconciliation_steps', []).append(MARKER)
    prov.setdefault('contracts', {})['occupancy_agents_joint_v1_source_guard'] = True
    prov['occupancy_physical_feasibility_has_not_been_proven'] = True
    provp.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    diff = ''.join(difflib.unified_diff(old_bytes.decode('utf-8').splitlines(keepends=True),src.splitlines(keepends=True),fromfile='parent/p1_native_audit.py',tofile='new/p1_native_audit.py'))
    (dest / 'occupancy_joint.diff').write_text(diff, encoding='utf-8')
    print('P1_OCCUPANCY_NEW_CAPSULE=' + str(dest))
    print('P1_OCCUPANCY_DIFF=' + str(dest / 'occupancy_joint.diff'))
    print('P1_OCCUPANCY_NEW_SOURCE_SHA256=' + sha_file(new_path))
    print('P1_OCCUPANCY_CREATED_CANDIDATE_ONLY_NOT_CARLA_PASS')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('P1_OCCUPANCY_FAIL_CLOSED=%s:%s' % (type(e).__name__, e), file=sys.stderr)
        sys.exit(2)
