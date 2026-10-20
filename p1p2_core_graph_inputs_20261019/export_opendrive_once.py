#!/usr/bin/env python3
"""Capture the ACTUAL five CARLA OpenDRIVE maps once, SHA/episode fenced.

This command WILL load different Towns if --ack-dedicated is supplied.
It never spawns actors, ticks the world, changes z/pose, or runs RIFT.
Requires a dedicated empty CARLA server before ANY map switch.
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import sys
import tarfile
import time

EXPECTED_MANIFEST_SHA='9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'
EXPECTED_TOWNS=('Town15','Town12','Town11','Town13','Town04')

class ContractError(RuntimeError):pass

def need(ok,why):
    if not ok:raise ContractError(why)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def read_manifest(root):
    path=root/'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    raw=path.read_bytes()
    need(sha(raw)==EXPECTED_MANIFEST_SHA,'frozen map manifest SHA differs')
    data=json.loads(raw.decode('utf-8'))
    m={v['town']:v['opendrive_sha256'] for v in data['maps']}
    for town in EXPECTED_TOWNS:need(town in m,'missing immutable Town '+town)
    return m,raw

def episode(world):return int(world.id)

def fetch_same_episode(client,world,expected_town,expected_sha):
    wid=episode(world)
    cmap=world.get_map()
    need(cmap.name.split('/')[-1]==expected_town,'Town mismatch in episode')
    raw=cmap.to_opendrive().encode('utf-8')
    need(sha(raw)==expected_sha,'OpenDRIVE SHA mismatch in episode '+expected_town)
    actual=client.get_world()
    need(episode(actual)==wid,'Episode changed during read '+expected_town)
    cmap2=actual.get_map()
    need(cmap2.name.split('/')[-1]==expected_town and sha(cmap2.to_opendrive().encode('utf-8'))==expected_sha,
         'Map changed during read '+expected_town)
    return raw,wid

def verified_world(client,town,expected_sha,ack):
    old=client.get_world();oldtown=old.get_map().name.split('/')[-1]
    if oldtown==town:
        raw,eid=fetch_same_episode(client,old,town,expected_sha)
        return raw,eid,False
    need(ack,'different Town; explicit --ack-dedicated required to load world')
    actors=old.get_actors()
    v=len(actors.filter('vehicle.*'));w=len(actors.filter('walker.pedestrian.*'))
    need(v==0 and w==0,'unsafe world nonempty: vehicles=%d walkers=%d'%(v,w))
    oldid=episode(old)
    client.load_world(town)
    deadline=time.monotonic()+120.0
    previous=None; stable=0
    while time.monotonic()<deadline:
        nw=client.get_world();eid=episode(nw)
        cmap=nw.get_map();actual_town=cmap.name.split('/')[-1]
        if eid==oldid or actual_town!=town:
            time.sleep(0.5);continue
        raw=cmap.to_opendrive().encode('utf-8')
        need(sha(raw)==expected_sha,'post-load wrong OpenDRIVE SHA for '+town)
        if previous==eid:stable+=1
        else:stable=1;previous=eid
        if stable>=2:
            data,eid2=fetch_same_episode(client,nw,town,expected_sha)
            need(eid2==eid,'episode instability')
            return data,eid,True
        time.sleep(0.5)
    raise ContractError('map switch episode/SHA timeout: '+town)

def run(args):
    root=pathlib.Path(args.root).resolve()
    out=pathlib.Path(args.out).resolve()
    need(not out.exists() and not out.with_suffix('.tar.gz').exists(),'output already exists')
    expect,manifest=read_manifest(root)
    egg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
    need(egg.is_file(),'pinned CARLA egg missing')
    sys.path.insert(0,str(egg))
    import carla
    need(str(carla.__file__).startswith(str(egg)+'/'),'wrong CARLA Python binding imported')
    client=carla.Client(args.host,args.port);client.set_timeout(120)
    need(client.get_server_version()==client.get_client_version()=='0.9.15','CARLA client/server version mismatch')
    out.mkdir(parents=True)
    (out/'frozen_map_manifest.json').write_bytes(manifest)
    rows=[]
    for town in args.towns:
        need(town in expect,'Town not in map manifest '+town)
        raw,eid,switched=verified_world(client,town,expect[town],args.ack_dedicated)
        f=out/(town+'.xodr.gz')
        with gzip.open(str(f),'wb') as g:g.write(raw)
        row={'town':town,'episode_id':eid,'switched_world':switched,
             'raw_opendrive_sha256':sha(raw),'raw_bytes':len(raw),
             'gzip_sha256':sha(f.read_bytes()),'gzip_bytes':f.stat().st_size}
        rows.append(row)
        print('MAP_EXPORTED='+town+' XODR_SHA256='+row['raw_opendrive_sha256']+' EPISODE='+str(eid),flush=True)
    (out/'map_exports.json').write_text(json.dumps({'schema':'actual_CARLA_world_OpenDRIVE_map_export_v1',
             'map_manifest_sha256':EXPECTED_MANIFEST_SHA,'exports':rows},indent=2)+'\n',encoding='utf-8')
    tarpath=out.with_suffix('.tar.gz')
    with tarfile.open(str(tarpath),'w:gz') as t:t.add(str(out),arcname=out.name)
    print('CARLA_MAP_EXPORT_COMPLETE=1')
    print('UPLOAD_FILE='+str(tarpath))
    print('UPLOAD_SHA256='+sha(tarpath.read_bytes()))
    print('UPLOAD_BYTES='+str(tarpath.stat().st_size))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=31120)
    p.add_argument('--towns',nargs='+',choices=EXPECTED_TOWNS,default=list(EXPECTED_TOWNS))
    p.add_argument('--ack-dedicated',action='store_true')
    args=p.parse_args()
    try:run(args)
    except (ContractError,OSError,ValueError,KeyError,ImportError,RuntimeError) as e:
        print('MAP_EXPORT_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(2)
