#!/usr/bin/env python3
"""Read-only P1 four-link witness across actual CARLA Towns; optional dedicated map loads.
Produces source-XODR geometry, exact lookup, directed OpenDRIVE link for 4 classes.
No P1 PASS based on this evidence alone.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,sys
from pathlib import Path
from p1_od_link_guard import verify_pair

CASES={0:18,155:9,1039:14,1279:19}
SHA_MANIFEST='92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
SHA_MAP='9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'

def need(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(w):return [int(w.road_id),int(w.section_id),int(w.lane_id)] if w else None
def angle(a,b):return abs((float(a)-float(b)+180)%360-180)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=31120)
    ap.add_argument('--town',required=True,choices=['Town11','Town12','Town13'])
    ap.add_argument('--output',required=True);ap.add_argument('--ack-dedicated',action='store_true')
    a=ap.parse_args();p=Path(a.output);need(not p.exists(),'REFUSE_OVERWRITE')
    root=Path(a.root).resolve();base=root/'data/hv_init/v7'
    man=base/'manifests/p1_calibration95.jsonl'
    mp=base/'manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    need(sha(man)==SHA_MANIFEST and sha(mp)==SHA_MAP,'manifest hash changed')
    expected={r['town']:r['opendrive_sha256'] for r in read(mp)['maps']}
    rows={int(r['record_index']):r for r in map(json.loads,man.read_text().splitlines())}
    egg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
    need(egg.is_file(),'pinned carla egg missing')
    sys.path.insert(0,str(egg));import carla
    need(str(carla.__file__).startswith(str(egg)+'/'),'imported wrong CARLA extension')
    client=carla.Client(a.host,a.port);client.set_timeout(120)
    need(client.get_client_version()==client.get_server_version()=='0.9.15','wrong CARLA version')
    prior=client.get_world();actual_town=prior.get_map().name.split('/')[-1]
    if actual_town!=a.town:
        need(a.ack_dedicated,'different Town: requires --ack-dedicated to load world')
        actors=prior.get_actors()
        need(len(actors.filter('vehicle.*'))==0 and len(actors.filter('walker.pedestrian.*'))==0,
             'refuse switch: dynamic actors still present')
        guardpath=root/'p1_science_v2_release/hv_v8/p1_world_guard.py'
        need(guardpath.is_file(),'world_guard missing')
        spec=importlib.util.spec_from_file_location('p1_temp_world_guard',str(guardpath));module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        world,cmap,_,transition=module.load_verified_world(client,a.town,expected[a.town])
    else:
        world=prior;cmap=world.get_map();transition={'no_map_load':True,'episode_id':world.id}
    expected_sha=expected[a.town];observed_sha=hashlib.sha256(cmap.to_opendrive().encode()).hexdigest()
    need(observed_sha==expected_sha,'MAP_SHA_MISMATCH')
    cases=[];xml=cmap.to_opendrive()
    for idx,i in CASES.items():
        row=rows[idx]
        need(sha(Path(row['snapshot_path']))==row['snapshot_sha256'],'snapshot modified '+str(idx))
        scene=read(row['snapshot_path'])
        if scene['town']!=a.town:continue
        pair=scene['ego_route'][i-1:i+1]
        need(len(pair)==2,'missing pair '+str(idx))
        wps=[];anchors=[]
        for point in pair:
            w=cmap.get_waypoint_xodr(int(point['road_id']),int(point['lane_id']),float(point['s']))
            if w is None:
                anchors.append({'status':'missing_xodr'});wps.append(None);continue
            pos=w.transform.location;loc=carla.Location(x=float(point['x']),y=float(point['y']),z=float(point['z']))
            geom={'status':'verified' if key(w)==[int(point['road_id']),int(point['section_id']),int(point['lane_id'])] and
                 pos.distance(loc)<=0.05 and angle(w.transform.rotation.yaw,point['yaw_deg'])<=0.5 and
                 abs(float(w.s)-float(point['s']))<=1e-4 and w.lane_type==carla.LaneType.Driving else 'mismatch',
                  'road_key':key(w),'s_delta_m':abs(float(w.s)-float(point['s'])),
                  'xyz_delta_m':float(pos.distance(loc)),
                  'yaw_delta_deg':angle(w.transform.rotation.yaw,point['yaw_deg'])}
            anchors.append(geom);wps.append(w)
        topo=verify_pair(xml,wps[0],wps[1]) if all(wps) and all(e['status']=='verified' for e in anchors) else {'status':'unproven','reason':'source_anchor_not_verified'}
        cases.append({'record_index':idx,'route_index':i,'town':a.town,
                      'original_pairs':pair,'source_anchors':anchors,'directed_od_link':topo})
    now=client.get_world();nm=now.get_map()
    need(now.id==world.id and nm.name==cmap.name and hashlib.sha256(nm.to_opendrive().encode()).hexdigest()==expected_sha,
         'world episode changed during validation')
    doc={'schema':'p1_four_link_od_evidence_v1','town':a.town,'map_sha256':expected_sha,
         'episode_id':world.id,'transition':transition,'cases':cases,'not_p1_pass':True}
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    for row in cases:print('P1_LINK_CASE='+json.dumps({'idx':row['record_index'],'link':row['directed_od_link']},ensure_ascii=False))
    print('P1_LINK_EVIDENCE_OUTPUT='+str(p))
    print('P1_LINK_EVIDENCE_NOT_P1_PASS')
if __name__=='__main__':main()
