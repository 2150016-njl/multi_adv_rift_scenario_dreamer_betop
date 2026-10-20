#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only CARLA directed lane topology and diagnostic GT lane reachability.

This is P2 foundation/diagnostic, NOT P2 GT approach recall. No world load, tick,
spawn, delete or actor mutations. Map graph uses only CARLA map.get_topology() and
waypoint.next(). GT labels are opened only AFTER constructing the topology.
Python 3.8+ stdlib + CARLA 0.9.15 PythonAPI.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

EXPECTED_MANIFEST = '92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
EXPECTED_MAP_MANIFEST = '9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'

class StopEvidence(RuntimeError):
    pass

def need(x, msg):
    if not x:
        raise StopEvidence(msg)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def townkey(s):
    return str(s or '').replace('\\','/').split('/')[-1]

def key(w):
    return (int(w.road_id), int(w.section_id), int(w.lane_id))

def angle(a,b):
    return abs((float(a)-float(b)+180.0)%360.0 - 180.0)

def xy(a,b):
    return math.hypot(float(a.x)-float(b.x),float(a.y)-float(b.y))

def validate_xodr(cmap, row):
    required = ('x','y','z','yaw_deg','road_id','section_id','lane_id','s')
    if not isinstance(row,dict) or any(k not in row for k in required):
        return {'status':'missing_fields'}
    try:
        wp = cmap.get_waypoint_xodr(int(row['road_id']),int(row['lane_id']),float(row['s']))
        if wp is None:
            return {'status':'missing_xodr'}
        p=wp.transform.location
        pos3=math.sqrt((p.x-float(row['x']))**2+(p.y-float(row['y']))**2+(p.z-float(row['z']))**2)
        sd=abs(float(wp.s)-float(row['s']))
        yd=angle(wp.transform.rotation.yaw,row['yaw_deg'])
        identity = key(wp)==(int(row['road_id']),int(row['section_id']),int(row['lane_id']))
        valid=identity and sd<=0.05 and pos3<=0.30 and yd<=2.0
        return {'status':'verified' if valid else 'geometry_mismatch',
                'lane':list(key(wp)), 's_error_m':sd,'xyz_error_m':pos3,'yaw_error_deg':yd}
    except Exception as exc:
        return {'status':'error','reason':'{}:{}'.format(type(exc).__name__,str(exc))}

def forward_reachable(edges, start, goal, cap=12):
    if start == goal:
        return True
    queue=collections.deque([(start,0)])
    visited={start}
    while queue:
        cur,depth=queue.popleft()
        if depth>=cap:
            continue
        for other in edges.get(cur,()):
            if other==goal:
                return True
            if other not in visited:
                visited.add(other)
                queue.append((other,depth+1))
    return False

def construct_graph(cmap, step_m=2.0):
    topology=list(cmap.get_topology())
    nodes=set()
    edges=collections.defaultdict(set)
    errors=[]
    for j,(a,b) in enumerate(topology):
        ka,kb=key(a),key(b)
        nodes.update((ka,kb))
        if ka!=kb:
            edges[ka].add(kb)
        try:
            nxt=b.next(step_m)
            for w in nxt:
                kw=key(w)
                nodes.add(kw)
                if kw!=kb:
                    edges[kb].add(kw)
        except Exception as exc:
            errors.append({'segment':j,'error':'{}:{}'.format(type(exc).__name__,str(exc))})
    return nodes,edges,{'topology_segment_count':len(topology),
                        'unique_directed_lane_keys':len(nodes),
                        'directed_lane_transitions':sum(len(v) for v in edges.values()),
                        'waypoint_next_errors':errors}

def run(args):
    root=Path(args.root).resolve()
    manifest=root/'data/hv_init/v7/manifests/p1_calibration95.jsonl'
    mpath=root/'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    need(sha(manifest)==EXPECTED_MANIFEST,'95 manifest SHA mismatch')
    need(sha(mpath)==EXPECTED_MAP_MANIFEST,'map identity SHA mismatch')
    mapdoc=json.loads(mpath.read_text(encoding='utf-8'))
    hashes={townkey(m['town']):m['opendrive_sha256'] for m in mapdoc['maps']}
    api=root/'carla/PythonAPI/carla'
    dist=api/'dist'
    if dist.is_dir():
        for egg in sorted(dist.glob('carla-0.9.15-*.egg')):
            sys.path.insert(0,str(egg))
    sys.path.insert(0,str(api))
    import carla
    client=carla.Client(args.host,args.port)
    client.set_timeout(120.0)
    need(client.get_server_version()=='0.9.15' and client.get_client_version()=='0.9.15','CARLA version != 0.9.15')
    world=client.get_world()
    cmap=world.get_map()
    town=townkey(cmap.name)
    need(town in hashes,'not one of five provenance Towns: '+town)
    if args.town:
        need(town==args.town,'current world town mismatch, refusing switch: '+town)
    actual=hashlib.sha256(cmap.to_opendrive().encode('utf-8')).hexdigest()
    need(actual==hashes[town],'OpenDRIVE SHA mismatch {} {}'.format(actual,hashes[town]))
    nodes,edges,graph_info=construct_graph(cmap)
    # All topology above is built without seeing any snapshot target or label.
    rows=[json.loads(t) for t in manifest.read_text(encoding='utf-8').splitlines() if t.strip()]
    evaluated=[]
    for ordinal,row in enumerate(rows):
        scene_path=Path(row['snapshot_path'])
        need(scene_path.is_file(),'missing snapshot '+str(scene_path))
        if row.get('snapshot_sha256'):
            need(sha(scene_path)==row['snapshot_sha256'],'snapshot sha mismatch: '+str(ordinal))
        scene=json.loads(scene_path.read_text(encoding='utf-8'))
        if townkey(scene.get('town') or scene.get('map_name'))!=town:
            continue
        route=scene.get('ego_route')
        agents=scene.get('agents')
        labels=scene.get('rift_labels')
        need(isinstance(route,list) and isinstance(agents,list) and isinstance(labels,list), 'incomplete scene '+str(ordinal))
        byid={str(a['actor_id']):a for a in agents}
        need(len(byid)==len(agents),'duplicate agent ids')
        route_evidence=[validate_xodr(cmap,point) for point in route]
        route_lanes={tuple(x['lane']) for x in route_evidence if x['status']=='verified'}
        targets=[]
        for label in labels:
            if label.get('is_interactive') is not True:
                continue
            aid=str(label.get('actor_id'))
            need(aid in byid,'label target missing agent '+aid)
            a=byid[aid]
            # Target-dependent queries belong ONLY to evaluation diagnostics.
            actor_lane=None
            actor_lookup_error=None
            try:
                wp=cmap.get_waypoint(carla.Location(float(a['x']),float(a['y']),float(a['z'])),project_to_road=False,lane_type=carla.LaneType.Driving)
                if wp is not None:
                    actor_lane=key(wp)
            except Exception as exc:
                actor_lookup_error='{}:{}'.format(type(exc).__name__,str(exc))
            anchor=validate_xodr(cmap,label.get('interaction_waypoint'))
            anchor_lane=tuple(anchor['lane']) if anchor['status']=='verified' else None
            # This is merely coarse directed-lane connectivity, not a spatially bounded
            # collision anchor, continuous spawn interval or native-positive replay.
            graph_reaches=(forward_reachable(edges,actor_lane,anchor_lane,args.max_lane_hops)
                           if actor_lane is not None and anchor_lane is not None else None)
            targets.append({'actor_id':aid,'actor_lane':list(actor_lane) if actor_lane else None,
                            'actor_lookup_error':actor_lookup_error,
                            'gt_anchor_verification':anchor,
                            'gt_anchor_lane_on_verified_ego_route':anchor_lane in route_lanes if anchor_lane else None,
                            'directed_lane_graph_reaches_gt_anchor':graph_reaches})
        evaluated.append({'ordinal':ordinal,'record_index':row['record_index'],
                          'scene_id_in_manifest':row.get('scene_id'),
                          'scene_id_in_snapshot':scene.get('scene_id'),
                          'route_xodr_verified_points':sum(x['status']=='verified' for x in route_evidence),
                          'route_point_count':len(route),'gt_targets':targets})
    alltargets=[x for rec in evaluated for x in rec['gt_targets']]
    result={'schema':'p2_lane_topology_probe_v1','scientific_P2_pass':False,
            'P2_GT_approach_recall_calculated':False,'town':town,'map_sha256':actual,
            'graph_source':'CARLA map.get_topology and waypoint.next (condition only)',
            'gt_only_used_for_posthoc_diagnostics':True,'max_lane_hops':args.max_lane_hops,
            'graph':graph_info,'node_keys':[list(k) for k in sorted(nodes)],
            'directed_edges':[{'from':list(a),'to':list(b)} for a in sorted(edges) for b in sorted(edges[a])],
            'evaluated_manifest_records':len(evaluated),
            'gt_evaluated_target_count':len(alltargets),
            'gt_lane_connectivity_true':sum(x['directed_lane_graph_reaches_gt_anchor'] is True for x in alltargets),
            'gt_lane_connectivity_false':sum(x['directed_lane_graph_reaches_gt_anchor'] is False for x in alltargets),
            'gt_lane_connectivity_unknown':sum(x['directed_lane_graph_reaches_gt_anchor'] is None for x in alltargets),
            'records':evaluated,
            'limitations':'Lane-level directed reachability diagnostic, not anchor construction, valid spawn interval, crossing-conflict detection, 110-front-GT recall, or P2 gate PASS.'}
    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print('P2_TOWN={} SHA_VERIFIED={}'.format(town,actual))
    print('P2_GRAPH_SEGMENTS={} LANE_KEYS={} DIRECTED_EDGES={}'.format(graph_info['topology_segment_count'],graph_info['unique_directed_lane_keys'],graph_info['directed_lane_transitions']))
    print('P2_DIAGNOSTIC_RECORDS={} GT_TARGETS={} CONNECTED={} NOT_CONNECTED={} UNKNOWN={}'.format(len(evaluated),len(alltargets),result['gt_lane_connectivity_true'],result['gt_lane_connectivity_false'],result['gt_lane_connectivity_unknown']))
    print('P2_PROBE_REPORT='+str(out))
    print('P2_NOT_APPROACH_RECALL_NOT_P2_GATE_PASS')
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',default=31120,type=int)
    p.add_argument('--town',default=None)
    p.add_argument('--max-lane-hops',type=int,default=12)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    try:
        need(1<=args.max_lane_hops<=100,'max_lane_hops outside [1,100]')
        return run(args)
    except (StopEvidence,OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        print('P2_PROBE_FAIL_CLOSED={} {}'.format(type(exc).__name__,str(exc)),file=sys.stderr)
        return 2

if __name__=='__main__':
    sys.exit(main())
