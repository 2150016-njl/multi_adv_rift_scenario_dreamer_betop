#!/usr/bin/env python3
"""Explicit dedicated-server CARLA map switch, verified new episode & SHA.
Does not tick, spawn, destroy, or modify scientific input data.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .p1_world_guard import load_verified_world

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--town',required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=31120)
    p.add_argument('--output',required=True);p.add_argument('--ack-dedicated',action='store_true')
    a=p.parse_args();
    if not a.ack_dedicated:raise SystemExit('REFUSE_NO_DEDICATED_SERVER_ACK')
    root=Path(a.root);eg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
    sys.path.insert(0,str(eg));import carla
    if not str(carla.__file__).startswith(str(eg)+'/'):raise SystemExit('WRONG_CARLA_MODULE')
    mm=json.loads((root/'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json').read_text())
    expected={r['town']:r['opendrive_sha256'] for r in mm['maps']}
    if a.town not in expected:raise SystemExit('UNKNOWN_TOWN')
    client=carla.Client(a.host,a.port);client.set_timeout(120)
    if (client.get_client_version(),client.get_server_version())!=('0.9.15','0.9.15'):
        raise SystemExit('CARLA_VERSION_MISMATCH')
    world=client.get_world();agents=world.get_actors()
    if agents.filter('vehicle.*') or agents.filter('walker.pedestrian.*'):
        raise SystemExit('NONEMPTY_WORLD_REFUSE_MAP_SWITCH')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    world,cmap,sha,evidence=load_verified_world(client,a.town,expected[a.town],record_id='P2_MAP_READ_ONLY',evidence_path=out)
    if world.get_actors().filter('vehicle.*') or world.get_actors().filter('walker.pedestrian.*'):
        raise SystemExit('NEW_WORLD_NONEMPTY')
    print('P2_TOWN_VERIFIED='+a.town)
    print('P2_TOWN_SHA='+sha)
    print('P2_MAP_EPISODE='+str(evidence['verified_episode_id']))
    print('P2_MAP_READY_NOT_SCIENCE_PASS')
if __name__=='__main__':main()
