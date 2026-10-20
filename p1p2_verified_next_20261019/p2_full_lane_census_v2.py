#!/usr/bin/env python3
"""P2 v2 read-only, GT-free CARLA full-lane census (not just topology same-lane edges).
Produces compatibility inputs for existing p2_approach_v1.py.
Does not load/switch world, spawn actors, or infer physical spawn legality.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,sys
from pathlib import Path

EXPECTED_SHA_MANIFEST='9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'
class EvidenceError(RuntimeError):pass
def need(v,m):
    if not v:raise EvidenceError(m)
def key(w):return int(w.road_id),int(w.section_id),int(w.lane_id)
def heading(a,b):return abs((float(a)-float(b)+180.0)%360.0-180.0)
def meta(w):
    tr=w.transform;p=tr.location
    return {'lane':list(key(w)),'s':float(w.s),'x':float(p.x),'y':float(p.y),
        'z':float(p.z),'yaw_deg':float(tr.rotation.yaw),'junction':bool(w.is_junction)}
def allowed(w,side,carla):
    options=(carla.LaneChange.Left,carla.LaneChange.Both) if side=='left' else (carla.LaneChange.Right,carla.LaneChange.Both)
    return w.lane_change in options

def collect(cmap,carla,conditions,town,spacing):
    # generate_waypoints() covers road lanes even when topology segment endpoints
    # cross road/lane identities. This avoids the original 169-point blind spot.
    raw=list(cmap.generate_waypoints(spacing))
    samples={}
    for w in raw:
        if w.lane_type!=carla.LaneType.Driving:continue
        d=meta(w)
        need(all(math.isfinite(d[k]) for k in ('x','y','z','s','yaw_deg')),'nonfinite CARLA map waypoint')
        samples[(tuple(d['lane']),round(d['s'],3))]=(w,d)
    need(samples,'no Driving waypoints')
    ranges=collections.defaultdict(list); lateral_samples=collections.defaultdict(list)
    xygrid=collections.defaultdict(list)
    cell=4.0
    for (lk,_),(w,d) in samples.items():
        ranges[lk].append(d['s'])
        xygrid[(math.floor(d['x']/cell),math.floor(d['y']/cell))].append(d)
        if d['junction']:continue
        for side in ('left','right'):
            if not allowed(w,side,carla):continue
            other=w.get_left_lane() if side=='left' else w.get_right_lane()
            if other is None or other.lane_type!=carla.LaneType.Driving or other.is_junction:continue
            nk=key(other)
            if lk[:2]!=nk[:2] or lk[2]*nk[2]<=0:continue
            if heading(d['yaw_deg'],other.transform.rotation.yaw)>30:continue
            q=other.transform.location
            dz=abs(d['z']-q.z)
            dist=math.hypot(d['x']-q.x,d['y']-q.y)
            if dz>0.6 or not 0.8<=dist<=7.0:continue
            lateral_samples[(lk,nk,side)].append(d['s'])
    intervals=[]
    for (src,dst,side),ss in sorted(lateral_samples.items()):
        ss=sorted(set(round(s,4) for s in ss));groups=[];run=[]
        for s in ss:
            if run and s-run[-1] > spacing*2.1:
                groups.append(run);run=[]
            run.append(s)
        if run:groups.append(run)
        for group in groups:
            if len(group)<3 or group[-1]-group[0]<8:continue
            intervals.append({'from_lane':list(src),'to_lane':list(dst),'direction':side,
                'source_s_min':group[0],'source_s_max':group[-1],
                'sample_count':len(group),'source_marking_permission_checked':True,
                'geometry_checks_passed_at_samples':True,
                'continuous_interval_between_samples_proven':False,
                'vehicle_collision_clearance_proven':False})
    records=[]
    for c in conditions:
        if c['town']!=town:continue
        need(c.get('gt_target_positions_in_condition') is False and
             not any(x in c for x in ('agents','rift_labels','targets')),
             'GT labels leaked into conditions')
        matches=[];dedup=set()
        for index,e in enumerate(c['ego_route']):
            ek=(int(e['road_id']),int(e['section_id']),int(e['lane_id']))
            x,y,z=float(e['x']),float(e['y']),float(e['z']);yaw=float(e['yaw_deg'])
            near=(math.floor(x/cell),math.floor(y/cell))
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for v in xygrid.get((near[0]+dx,near[1]+dy),[]):
                        lk=tuple(v['lane'])
                        if lk==ek:continue
                        hd=heading(yaw,v['yaw_deg']);hdxy=math.hypot(v['x']-x,v['y']-y)
                        if hdxy>3.0 or abs(v['z']-z)>0.7 or not 25<=hd<=155:continue
                        token=(index,lk)
                        if token in dedup:continue
                        dedup.add(token)
                        matches.append({'ego_route_index':index,'ego_lane':list(ek),
                            'other_lane':list(lk),'other_s_sample':v['s'],
                            'other_xyz':[v['x'],v['y'],v['z']],
                            'ego_xyz':[x,y,z], 'xy_separation_m':hdxy,
                            'z_separation_m':abs(v['z']-z),'heading_difference_deg':hd,
                            'anchor_kind':'geometric_crossing_candidate',
                            'native_interaction_verified':False})
        records.append({'record_index':int(c['record_index']),'town':town,
                        'crossing_anchor_candidates':matches})
    need(records,'no conditions for town')
    station=[{'lane':list(k),'s_min':min(v),'s_max':max(v),'sample_count':len(v)}
             for k,v in sorted(ranges.items())]
    lateral={'schema':'p2_lateral_legal_segment_census_v0','town':town,
            'intervals':intervals,'waypoints_sampled':len(samples),'sample_errors':[],
            'GT_used':False,'not_model3_physical_spawn_pass':True,
            'sampling_method':'CARLA map.generate_waypoints','note':'sampled legality, not continuous lane-change path proof'}
    geom={'schema':'p2_gt_free_map_geometry_v1','town':town,'records':records,
          'sampled_lane_station_ranges':station,'sampled_waypoint_count':len(samples),
          'sample_errors':[],'GT_used':False,'continuous_spawn_legality_verified':False,
          'geometric_crossings_are_not_native_RIFT_interactions':True,
          'sampling_method':'CARLA map.generate_waypoints'}
    return lateral,geom,{'driving_samples':len(samples),'distinct_lane_keys':len(ranges),
                         'lateral_intervals':len(intervals),
                         'crossing_candidates':sum(len(z['crossing_anchor_candidates']) for z in records),
                         'conditions_in_town':len(records)}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',required=True);a.add_argument('--conditions',required=True)
    a.add_argument('--topology',required=True)
    a.add_argument('--town',required=True);a.add_argument('--output-dir',required=True)
    a.add_argument('--host',default='127.0.0.1');a.add_argument('--port',type=int,default=31120)
    a.add_argument('--step-m',type=float,default=2.0)
    x=a.parse_args()
    try:
        need(1<=x.step_m<=5,'step_m outside [1,5]')
        out=Path(x.output_dir)
        need(not out.exists(),'output directory already exists; refuse overwrite')
        root=Path(x.root).resolve()
        mp=root/'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json'
        need(hashlib.sha256(mp.read_bytes()).hexdigest()==EXPECTED_SHA_MANIFEST,'map manifest changed')
        doc=json.loads(mp.read_text());sha_by={a['town']:a['opendrive_sha256'] for a in doc['maps']}
        need(x.town in sha_by,'unknown Town')
        egg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
        need(egg.is_file(),'missing CARLA pinned egg')
        sys.path.insert(0,str(egg));import carla
        need(str(carla.__file__).startswith(str(egg)+'/'),'CARLA wrong Python egg')
        client=carla.Client(x.host,x.port);client.set_timeout(120)
        need(client.get_client_version()=='0.9.15' and client.get_server_version()=='0.9.15','CARLA version wrong')
        world=client.get_world();cmap=world.get_map()
        need(cmap.name.split('/')[-1]==x.town,'current world wrong Town; refuse map switch')
        sha=hashlib.sha256(cmap.to_opendrive().encode()).hexdigest()
        need(sha==sha_by[x.town],'actual OpenDRIVE SHA mismatch')
        conditions=[json.loads(s) for s in Path(x.conditions).read_text().splitlines() if s.strip()]
        topology=json.loads(Path(x.topology).read_text())
        need(topology.get('town')==x.town and topology.get('map_sha256')==sha,
             'TOPOLOGY_MAP_SHA_OR_TOWN_MISMATCH')
        node_keys={tuple(int(z) for z in row) for row in topology['node_keys']}
        lateral,geom,counts=collect(cmap,carla,conditions,x.town,x.step_m)
        sampled_keys={tuple(r['lane']) for r in geom['sampled_lane_station_ranges']}
        route_keys={tuple(int(r[z]) for z in ('road_id','section_id','lane_id'))
                    for c in conditions if c['town']==x.town for r in c['ego_route']}
        missing_route=(route_keys & node_keys)-sampled_keys
        counts['topology_lane_keys']=len(node_keys)
        counts['topology_lanes_missing_samples']=len(node_keys-sampled_keys)
        counts['ego_route_lanes_missing_station_samples']=[list(k) for k in sorted(missing_route)]
        counts['ego_route_lanes_missing_from_topology']=[list(k) for k in sorted(route_keys-node_keys)]
        geom['topology_source_sha256']=hashlib.sha256(Path(x.topology).read_bytes()).hexdigest()

        after=client.get_world();amap=after.get_map()
        need(after.id==world.id and amap.name==cmap.name and
             hashlib.sha256(amap.to_opendrive().encode()).hexdigest()==sha,
             'world/map changed during census')
        lateral['opendrive_sha256']=sha
        geom['map_sha256']=sha
        geom['conditions_sha256']=hashlib.sha256(Path(x.conditions).read_bytes()).hexdigest()
        out.mkdir(parents=True,exist_ok=False)
        for name,obj in [('lateral.json',lateral),('geometry.json',geom),('census_summary.json',counts)]:
            (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
        print('P2_V2_MAP_DRIVING_SAMPLES='+str(counts['driving_samples']))
        print('P2_V2_DISTINCT_LANE_KEYS='+str(counts['distinct_lane_keys']))
        print('P2_V2_LATERAL_INTERVALS='+str(counts['lateral_intervals']))
        print('P2_V2_TOPOLOGY_LANES_MISSING_SAMPLES='+str(counts['topology_lanes_missing_samples']))
        print('P2_V2_ROUTE_LANES_MISSING_SAMPLES='+str(counts['ego_route_lanes_missing_station_samples']))
        print('P2_V2_CROSSINGS='+str(counts['crossing_candidates']))
        print('P2_V2_OUTPUT='+str(out))
        print('P2_SAMPLE_EVIDENCE_ONLY_NOT_110_RECALL_NOT_SPAWN_PASS')
        if missing_route:print('P2_INCOMPLETE_ROUTE_ANCHOR_SAMPLING_FAIL_CLOSED')
        return 3 if missing_route else 0
    except (EvidenceError,OSError,ValueError,KeyError,TypeError,ImportError) as e:
        print('P2_V2_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr)
        return 2
    return 0
if __name__=='__main__':sys.exit(main())
