#!/usr/bin/env python3
"""CARLA 0.9.15 read-only directed, permission-gated lane-change interval census.

Uses only map topology + lane-change markings + geometric adjacency. NEVER uses
GT actor states/labels; NEVER spawns, ticks or changes maps. Returns conservative
lane-key/s intervals for investigation, NOT a validated vehicle trajectory.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

class EvidenceError(RuntimeError):pass
def need(v,msg):
    if not v:raise EvidenceError(msg)
def key(w):return int(w.road_id),int(w.section_id),int(w.lane_id)
def heading_error(a,b):return abs(((float(a)-float(b)+180.0)%360.0)-180.0)
def distance(a,b):
    x,y=a.transform.location,b.transform.location
    return math.hypot(float(x.x)-float(y.x),float(x.y)-float(y.y))
def permission(w,side,carla):
    allowed=(carla.LaneChange.Left,carla.LaneChange.Both) if side=='left' else (carla.LaneChange.Right,carla.LaneChange.Both)
    return w.lane_change in allowed
def valid_lateral(w,side,carla):
    if bool(w.is_junction) or w.lane_type!=carla.LaneType.Driving or not permission(w,side,carla):return None
    other=w.get_left_lane() if side=='left' else w.get_right_lane()
    if other is None or other.lane_type!=carla.LaneType.Driving or bool(other.is_junction):return None
    if (int(w.road_id),int(w.section_id))!=(int(other.road_id),int(other.section_id)):return None
    if int(w.lane_id)*int(other.lane_id)<=0:return None
    if heading_error(w.transform.rotation.yaw,other.transform.rotation.yaw)>30:return None
    dz=abs(float(w.transform.location.z)-float(other.transform.location.z))
    if dz>0.6:return None
    d=distance(w,other)
    if not (0.8 <= d <= 7.0):return None
    return other

def collect(cmap,carla,step=2.0,minimum_interval=8.0):
    # Segments only from CARLA's lane-level topology. We bound sampling by s
    # and do not assume any lane change extends along the entire lane.
    evidence=collections.defaultdict(list);sample_errors=[];sample_count=0
    for j,(start,end) in enumerate(cmap.get_topology()):
        if key(start)!=key(end):continue
        cur=start
        start_s=float(start.s);end_s=float(end.s)
        direction=1 if end_s>=start_s else -1
        max_steps=int(abs(end_s-start_s)/step)+8
        seen=set()
        for iteration in range(max_steps):
            if key(cur)!=key(start):break
            s=float(cur.s)
            if (s-start_s)*direction< -0.01 or (s-end_s)*direction>0.01:break
            tag=(key(cur),round(s,3))
            if tag in seen:break
            seen.add(tag);sample_count+=1
            for side in ('left','right'):
                other=valid_lateral(cur,side,carla)
                if other is not None:
                    evidence[(key(cur),key(other),side)].append(s)
            if abs(s-end_s)<step*0.5:break
            try:
                choices=[n for n in cur.next(step) if key(n)==key(start)]
                if len(choices)!=1:break
                nxt=choices[0]
                if (float(nxt.s)-s)*direction<=1e-4:break
                cur=nxt
            except Exception as e:
                sample_errors.append({'segment':j,'reason':type(e).__name__+': '+str(e)})
                break
    intervals=[]
    for (src,dst,side),ss in sorted(evidence.items()):
        ss=sorted(set(round(s,5) for s in ss));groups=[];active=[]
        for s in ss:
            if active and s-active[-1]>step*2.1:
                groups.append(active);active=[]
            active.append(s)
        if active:groups.append(active)
        for g in groups:
            if len(g)<3 or g[-1]-g[0]<minimum_interval:continue
            intervals.append({'from_lane':list(src),'to_lane':list(dst),'direction':side,
                              'source_s_min':g[0],'source_s_max':g[-1],
                              'sample_count':len(g),'source_marking_permission_checked':True,
                              'geometry_checks_passed_at_samples':True,
                              'continuous_interval_between_samples_proven':False,
                              'vehicle_collision_clearance_proven':False})
    return {'intervals':intervals,'waypoints_sampled':sample_count,'sample_errors':sample_errors,
            'topology_segment_count':len(cmap.get_topology())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=31120)
    p.add_argument('--town',required=True);p.add_argument('--step-m',type=float,default=2.0)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    try:
        need(1.0<=a.step_m<=5.0,'sample spacing out of [1,5] m')
        root=Path(a.root).resolve();mmap=root/'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json'
        need(hashlib.sha256(mmap.read_bytes()).hexdigest()=='9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf','map manifest hash mismatch')
        maps={row['town']:row['opendrive_sha256'] for row in json.loads(mmap.read_text())['maps']}
        need(a.town in maps,'unknown Town')
        egg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
        need(egg.is_file(),'pinned CARLA py3.7 egg missing')
        sys.path.insert(0,str(egg));import carla
        need(str(carla.__file__).startswith(str(egg)+'/'),'imported wrong CARLA egg')
        client=carla.Client(a.host,a.port);client.set_timeout(90)
        need(client.get_server_version()=='0.9.15' and client.get_client_version()=='0.9.15','CARLA version mismatch')
        before=client.get_world();bmap=before.get_map();name=bmap.name.split('/')[-1]
        need(name==a.town,'Town mismatch; refusing map switch: '+name)
        actual=hashlib.sha256(bmap.to_opendrive().encode('utf-8')).hexdigest()
        need(actual==maps[a.town],'OpenDRIVE mismatch')
        result=collect(bmap,carla,a.step_m)
        after=client.get_world();amap=after.get_map()
        need(after.id==before.id and amap.name==bmap.name and hashlib.sha256(amap.to_opendrive().encode()).hexdigest()==actual,
             'world episode/map changed during census')
        doc={'schema':'p2_lateral_legal_segment_census_v0','town':a.town,'opendrive_sha256':actual,
             'GT_used':False,'not_model3_physical_spawn_pass':True,**result}
        with Path(a.output).open('x',encoding='utf-8') as f:json.dump(doc,f,ensure_ascii=False,indent=2);f.write('\n')
        print('P2_LATERAL_SEGMENTS='+str(len(result['intervals'])))
        print('P2_LATERAL_SAMPLED='+str(result['waypoints_sampled']))
        print('P2_LATERAL_ERRORS='+str(len(result['sample_errors'])))
        print('P2_LATERAL_REPORT='+str(Path(a.output).resolve()))
        print('P2_LATERAL_SAMPLE_EVIDENCE_ONLY_NOT_PHYSICS_OR_110_RECALL')
        return 0 if not result['sample_errors'] else 3
    except (EvidenceError,ValueError,KeyError,TypeError,OSError,ImportError) as e:
        print('P2_LATERAL_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
