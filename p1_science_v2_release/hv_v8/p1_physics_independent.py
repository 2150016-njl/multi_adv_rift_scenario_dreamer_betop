#!/usr/bin/env python3
"""CARLA 0.9.15 dedicated, one-record, one-phase physical evidence.

Never called as part of native P1; uses a fresh episode for each phase.
No GT modifications, no height retry, no tick. This is NOT full training gate.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--record-index',type=int,required=True)
    p.add_argument('--phase',choices=['raw','occupancy'],required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=31120)
    p.add_argument('--output',required=True);p.add_argument('--ack-dedicated',action='store_true')
    a=p.parse_args()
    if not a.ack_dedicated: p.error('dedicated CARLA acknowledgment required')
    root=Path(a.root);dist=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
    if not dist.exists():raise SystemExit('PINNED_CARLA_EGG_MISSING')
    sys.path.insert(0,str(dist));import carla
    if not str(carla.__file__).startswith(str(dist)+'/'):raise SystemExit('WRONG_CARLA_MODULE_ORIGIN')
    from . import p1_native_audit as science
    from .common import require
    manifest=root/'data/hv_init/v7/manifests/p1_calibration95.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    match=[row for row in rows if int(row['record_index'])==a.record_index]
    require(len(match)==1,'record index not unique in manifest')
    row=match[0];path=Path(row['snapshot_path']);raw=path.read_bytes()
    expected_sha=row.get('snapshot_sha256')
    if expected_sha:require(hashlib.sha256(raw).hexdigest()==expected_sha,'snapshot SHA mismatch')
    scene=json.loads(raw);town=scene['town']
    mm=root/'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    maps={x['town']:x['opendrive_sha256'] for x in json.loads(mm.read_text())['maps']}
    require(town in maps,'untrusted town')
    client=carla.Client(a.host,a.port);client.set_timeout(120)
    require(client.get_client_version()=='0.9.15' and client.get_server_version()=='0.9.15',
            'CARLA version mismatch')
    before=client.get_world(); actors=before.get_actors()
    require(not actors.filter('vehicle.*') and not actors.filter('walker.pedestrian.*'),
            'nonempty world BEFORE phase, refuse load')
    from .p1_world_guard import load_verified_world,assert_same_episode_map
    out=Path(a.output);require(not out.exists(),'output exists')
    world,cmap,xsha,trans=load_verified_world(client,town,maps[town],record_id=a.record_index,
       evidence_path=out.with_suffix('.transition.json'))
    require(not world.get_actors().filter('vehicle.*') and not world.get_actors().filter('walker.pedestrian.*'),
            'nonempty fresh world')
    targets,sel=science.select_target_actors(row,scene,scene['agents'])
    require(sel.get('status')=='pass','invalid target identities')
    if a.phase=='raw':
        # A raw attempt for each target, but not jointly with background.
        results=[science.raw_sedan_spawn(world,carla,dict(t)) for t in targets]
        payload={'phase':'raw','results':results,'full_scene_proven':False}
    else:
        evidence=science.scene_occupancy_spawn(client,world,carla,scene,targets)
        payload={'phase':'occupancy','results':evidence,
                 'full_scene_proven':False,'note':'RPC outcome is not a full dynamic collision/pose certificate'}
    assert_same_episode_map(client,trans['verified_episode_id'],town,xsha)
    remain=client.get_world().get_actors()
    payload.update({'schema':'p1_independent_physics_v1','record_index':a.record_index,
        'town':town,'map_sha256':xsha,'map_episode_id':trans['verified_episode_id'],
        'ending_vehicle_count':len(remain.filter('vehicle.*')),
        'ending_walker_count':len(remain.filter('walker.pedestrian.*')),
        'scientific_P1_pass':False})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2,ensure_ascii=False);f.write('\n')
    print('P1_INDEPENDENT_PHYSICS_REPORT='+str(out))
    print('P1_PHYSICS_NOT_NATIVE_P1_PASS_OR_MODEL3_FULL_SCENE_TRAINING_GATE')
    if payload['ending_vehicle_count'] or payload['ending_walker_count']:
        raise SystemExit('PHYSICS_ACTOR_CLEANUP_UNVERIFIED')
    return 0

if __name__=='__main__':sys.exit(main())
