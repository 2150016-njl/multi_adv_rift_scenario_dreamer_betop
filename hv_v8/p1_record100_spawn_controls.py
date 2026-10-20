#!/usr/bin/env python3
"""Record-100 CARLA 0.9.15 *diagnostic-only* five-condition spawn control.

Requires an exclusively controlled, already-loaded, empty Town15 world. Each
condition is ONE raw spawn call, followed by verified cleanup if it succeeds.
No load_world, world.tick, transform alteration, fallback, or pose retries.
Not a P1 acceptance test; no actor is retained as a generated CBV.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_SHA = '37bd5c64865372a4627b9a7c9d53f03bbe4b8cb2b3dc7ff8979a3f0d203447c7'
TOWN15_SHA = '8e9c69cfb3e750ace3e5d6b057e46786952a073c3924f319bbabb1a166dc2c17'
SCENE_ID = 'RouteScenario_24333_idx27_rep0_frame145'
EXPECTED_AGENT_COUNT = 8
TARGETS = (11141, 11143)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def snapshot_read(path):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    require(actual == SNAPSHOT_SHA, 'snapshot SHA mismatch {} != {}'.format(actual, SNAPSHOT_SHA))
    scene = json.loads(raw.decode('utf-8'))
    require(scene.get('scene_id') == SCENE_ID, 'scene ID mismatch')
    require(scene.get('town') == 'Town15', 'scene Town mismatch')
    require(len(scene.get('agents') or []) == EXPECTED_AGENT_COUNT, 'agent count mismatch')
    ids = {int(a['actor_id']): a for a in scene['agents']}
    require(len(ids) == EXPECTED_AGENT_COUNT, 'duplicate or missing agents')
    require(set(TARGETS).issubset(ids), 'target IDs missing')
    require(int(scene['ego'].get('actor_id', -1)) == 11129, 'ego actor ID mismatch')
    positive = {int(a['actor_id']) for a in scene.get('rift_labels', []) if a.get('is_interactive') is True}
    require(positive == set(TARGETS), 'native-positive ID mismatch')
    return scene, ids


def to_transform(carla, a):
    return carla.Transform(
        carla.Location(x=float(a['x']), y=float(a['y']), z=float(a['z'])),
        carla.Rotation(pitch=float(a.get('pitch_deg', 0.0)),
                       yaw=float(a['yaw_deg']),
                       roll=float(a.get('roll_deg', 0.0))))


def actor_counts(world):
    actors = world.get_actors()
    return {
        'vehicles': len(actors.filter('vehicle.*')),
        'walkers': len(actors.filter('walker.pedestrian.*')),
        'total_actors': len(actors),
    }


def no_traffic(world, label):
    c = actor_counts(world)
    require(c['vehicles'] == 0 and c['walkers'] == 0,
            '{}: nonempty world, counts={}; no actor was touched'.format(label, c))
    return c


def one_trial(client, world, carla, blueprint_library, name, row, blueprint_id):
    counts_before = no_traffic(world, 'before {}'.format(name))
    bp = blueprint_library.find(blueprint_id)
    original_pose = {k: row.get(k) for k in ('x', 'y', 'z', 'roll_deg', 'pitch_deg', 'yaw_deg')}
    tr = to_transform(carla, row)
    actor_id = None
    result = {
        'case': name,
        'snapshot_actor_id': row['actor_id'],
        'blueprint_id': blueprint_id,
        'original_pose': original_pose,
        'counts_before': counts_before,
        'attempts': 1,
    }
    try:
        # Unlike try_spawn_actor(), batch SpawnActor returns CARLA's error string.
        replies = client.apply_batch_sync([carla.command.SpawnActor(bp, tr)], False)
        require(len(replies) == 1, '{}: unexpected response count {}'.format(name, len(replies)))
        reply = replies[0]
        if reply.error:
            result.update({'status': 'rejected', 'carla_error': str(reply.error), 'new_actor_id': None})
        else:
            actor_id = int(reply.actor_id)
            result.update({'status': 'spawn_rpc_accepted', 'carla_error': None, 'new_actor_id': actor_id})
    finally:
        if actor_id is not None:
            replies = client.apply_batch_sync([carla.command.DestroyActor(actor_id)], False)
            require(len(replies) == 1 and not replies[0].error,
                    '{}: cleanup failed for new actor {}; STOP to protect world'.format(name, actor_id))
        counts_after = no_traffic(world, 'after {}'.format(name))
        result['counts_after'] = counts_after
    print('PHYSICAL_CONTROL_CASE=' + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output', required=True, help='new JSON path; never overwrites')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=31120)
    p.add_argument('--rpc-timeout', type=float, default=120)
    args = p.parse_args()
    input_path = Path(args.snapshot).resolve()
    output_path = Path(args.output).resolve()
    require(not output_path.exists(), 'output path exists; refuse overwrite')
    scene, ids = snapshot_read(input_path)

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.rpc_timeout)
    require(client.get_server_version() == '0.9.15', 'server version mismatch')
    require(client.get_client_version() == '0.9.15', 'client version mismatch')
    world = client.get_world()
    carla_map = world.get_map()
    town = str(carla_map.name).replace('\\', '/').split('/')[-1]
    require(town == 'Town15', 'current map is {}, no map switch permitted'.format(carla_map.name))
    actual_map_sha = hashlib.sha256(carla_map.to_opendrive().encode('utf-8')).hexdigest()
    require(actual_map_sha == TOWN15_SHA, 'Town15 OpenDRIVE SHA mismatch')
    initial = no_traffic(world, 'initial')
    print('P1_DIAGNOSTIC_PREFLIGHT=Town15_SHA_MATCH_0_VEHICLES_0_WALKERS', flush=True)

    # One trial per experimental condition, not retries of a failed proposal.
    cases = [
        ('ego_original', scene['ego'], str(scene['ego']['type_id'])),
        ('11141_original', ids[11141], str(ids[11141]['type_id'])),
        ('11141_model3', ids[11141], 'vehicle.tesla.model3'),
        ('11143_original', ids[11143], str(ids[11143]['type_id'])),
        ('11143_model3', ids[11143], 'vehicle.tesla.model3'),
    ]
    trials = []
    library = world.get_blueprint_library()
    for case_name, row, bp_name in cases:
        trials.append(one_trial(client, world, carla, library, case_name, row, bp_name))
    final = no_traffic(world, 'final')
    report = {
        'diagnostic_only_not_p1_pass': True,
        'record_index': 100, 'scene_id': SCENE_ID,
        'snapshot_path': str(input_path), 'snapshot_sha256': SNAPSHOT_SHA,
        'town': carla_map.name, 'opendrive_sha256': actual_map_sha,
        'initial_counts': initial, 'final_counts': final,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'trials': trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('P1_PHYSICAL_CONTROL_REPORT=' + str(output_path))
    print('P1_PHYSICAL_DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE_PASS')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('P1_PHYSICAL_DIAGNOSTIC_STOP={} {}'.format(type(e).__name__, e), file=sys.stderr)
        sys.exit(2)
