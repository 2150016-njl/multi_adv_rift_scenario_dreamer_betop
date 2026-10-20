#!/usr/bin/env python3
"""Read-only CARLA/OpenDRIVE sampling of GT-free intersection/merge geometry.
Samples road centerlines and compares them to the provided ego future routes.
Geometry conflict is a NECESSARY cue, NOT proof of eventual RIFT interaction.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,sys
from pathlib import Path


def uid(w):return (int(w.road_id),int(w.section_id),int(w.lane_id))
def xyz(w):
    p=w.transform.location
    return float(p.x),float(p.y),float(p.z)
def heading_diff(a,b):return abs((float(a)-float(b)+180.0)%360.0-180.0)

def make_geometry(cmap,conditions,town,step=2.0):
    points={};ranges=collections.defaultdict(list)
    visited=set();errs=[]
    # Scan lane directed segments at <=2m spacing; exclude sidewalks.
    for start,end in cmap.get_topology():
        a,b=uid(start),uid(end)
        if a!=b: continue
        p=start;target_s=float(end.s);sign=1 if target_s>=float(start.s) else -1
        for j in range(min(10000,int(abs(target_s-float(start.s))/step)+6)):
            if uid(p)!=a:break
            ss=float(p.s)
            if (ss-target_s)*sign>0.1:break
            token=(a,round(ss,2))
            if token in visited:break
            visited.add(token)
            x,y,z=xyz(p);ranges[a].append(ss)
            points[token]={'lane':list(a),'s':ss,'x':x,'y':y,'z':z,
                          'yaw_deg':float(p.transform.rotation.yaw),'junction':bool(p.is_junction)}
            if abs(ss-target_s)<step*0.5:break
            try:
                nexts=[n for n in p.next(step) if uid(n)==a and (float(n.s)-ss)*sign>1e-3]
                if len(nexts)!=1:break
                p=nexts[0]
            except Exception as e:
                errs.append(type(e).__name__+': '+str(e));break
    grid=collections.defaultdict(list)
    cell=4.0
    for point in points.values():
        grid[(int(math.floor(point['x']/cell)),int(math.floor(point['y']/cell)))].append(point)
    records=[]
    for row in conditions:
        if row['town']!=town:continue
        rid=int(row['record_index']); ego=row['ego_route'];matches=[];dedup=set()
        for index,e in enumerate(ego):
            ek=(int(e['road_id']),int(e['section_id']),int(e['lane_id']))
            ex,ey,ez=float(e['x']),float(e['y']),float(e['z']);key=(int(math.floor(ex/cell)),int(math.floor(ey/cell)))
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for v in grid.get((key[0]+dx,key[1]+dy),[]):
                        lk=tuple(v['lane'])
                        if lk==ek:continue
                        horizontal=math.hypot(v['x']-ex,v['y']-ey)
                        zdist=abs(v['z']-ez)
                        hd=heading_diff(v['yaw_deg'],float(e['yaw_deg']))
                        if horizontal>3.0 or zdist>0.7 or hd<25.0 or hd>155.0:continue
                        tag=(index,lk)
                        if tag in dedup:continue
                        dedup.add(tag)
                        matches.append({'ego_route_index':index,'ego_lane':list(ek),'other_lane':list(lk),
                            'other_s_sample':v['s'],'other_xyz':[v['x'],v['y'],v['z']],
                            'ego_xyz':[ex,ey,ez],'xy_separation_m':horizontal,'z_separation_m':zdist,
                            'heading_difference_deg':hd,'anchor_kind':'geometric_crossing_candidate',
                            'native_interaction_verified':False})
        records.append({'record_index':rid,'town':town,'crossing_anchor_candidates':matches})
    return {'schema':'p2_gt_free_map_geometry_v1','town':town,'records':records,
        'sampled_lane_station_ranges':[{'lane':list(k),'s_min':min(vals),'s_max':max(vals),
                                        'sample_count':len(vals)} for k,vals in sorted(ranges.items())],
        'sampled_waypoint_count':len(points),'sample_errors':errs,'GT_used':False,
        'continuous_spawn_legality_verified':False,'geometric_crossings_are_not_native_RIFT_interactions':True}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',required=True);a.add_argument('--town',required=True)
    a.add_argument('--host',default='127.0.0.1');a.add_argument('--port',type=int,default=31120)
    a.add_argument('--conditions',required=True);a.add_argument('--output',required=True)
    arg=a.parse_args();root=Path(arg.root);out=Path(arg.output)
    if out.exists():raise SystemExit('REFUSE_OUTPUT_OVERWRITE')
    mm=json.loads((root/'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json').read_text())
    trusted={m['town']:m['opendrive_sha256'] for m in mm['maps']}
    if arg.town not in trusted:raise SystemExit('UNTRUSTED_TOWN')
    egg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
    sys.path.insert(0,str(egg));import carla
    if not str(carla.__file__).startswith(str(egg)+'/'):raise SystemExit('WRONG_CARLA_BINARY')
    client=carla.Client(arg.host,arg.port);client.set_timeout(90)
    if (client.get_client_version(),client.get_server_version())!=('0.9.15','0.9.15'):
        raise SystemExit('CARLA_VERSION_MISMATCH')
    world=client.get_world();cmap=world.get_map()
    if cmap.name.split('/')[-1]!=arg.town:raise SystemExit('WRONG_MAP_REFUSE_SWITCH')
    sha=hashlib.sha256(cmap.to_opendrive().encode()).hexdigest()
    if sha!=trusted[arg.town]:raise SystemExit('MAP_SHA_MISMATCH')
    conditions=[json.loads(x) for x in Path(arg.conditions).read_text().splitlines() if x.strip()]
    # Prove condition rows contain no GT actor information.
    for c in conditions:
        if c.get('gt_target_positions_in_condition') is not False or any(k in c for k in ('agents','rift_labels','targets')):
            raise SystemExit('GT_LEAKAGE_IN_P2_CONDITION')
    report=make_geometry(cmap,conditions,arg.town)
    w2=client.get_world();m2=w2.get_map()
    if w2.id!=world.id or m2.name!=cmap.name or hashlib.sha256(m2.to_opendrive().encode()).hexdigest()!=sha:
        raise SystemExit('CARLA_EPISODE_CHANGED_DURING_SAMPLE')
    report.update({'map_sha256':sha,'conditions_sha256':hashlib.sha256(Path(arg.conditions).read_bytes()).hexdigest()})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('P2_GEOMETRY_WAYPOINTS='+str(report['sampled_waypoint_count']))
    print('P2_GEOMETRY_CROSSING_ANCHORS='+str(sum(len(x['crossing_anchor_candidates']) for x in report['records'])))
    print('P2_GEOMETRY_OUTPUT='+str(out))
    print('P2_GEOMETRY_NOT_NATIVE_OR_MODEL3_PASS')
    return 0 if not report['sample_errors'] else 3

if __name__=='__main__':sys.exit(main())
