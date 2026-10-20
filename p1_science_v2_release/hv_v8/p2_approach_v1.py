#!/usr/bin/env python3
"""GT-free P2 directed approach candidate builder + independent evaluation.

Make: conditions + CARLA directed topology + CARLA lateral marking samples +
      CARLA XY/Z crossing evidence. No GT rows read in make.
Evaluate: only after a frozen candidate file exists; diagnostic probe or fixed110
          file with exactly 110 verified distinct IDs. No 'native PASS' claimed.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,sys
from pathlib import Path

class EvidenceError(RuntimeError):pass
def need(cond,msg):
    if not cond:raise EvidenceError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def jl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lane(a):
    if isinstance(a,(list,tuple)):
        need(len(a)==3,'invalid lane key');return tuple(int(x) for x in a)
    return tuple(int(a[k]) for k in ('road_id','section_id','lane_id'))
def write(p,obj):
    p=Path(p);need(not p.exists(),'output already exists');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')

def make(topology,conditions,lateral,geometry,max_hops=12):
    need(1<=max_hops<=40,'invalid max_hops')
    need(topology['town']==lateral['town']==geometry['town'],'Town mismatch')
    need(topology['map_sha256']==lateral['opendrive_sha256']==geometry['map_sha256'],'OpenDRIVE SHA mismatch')
    need(lateral.get('GT_used') is False and geometry.get('GT_used') is False,'GT data in map inputs')
    need(all(c.get('gt_target_positions_in_condition') is False and
             not any(x in c for x in ('agents','rift_labels','targets')) for c in conditions),
         'GT labels in candidate conditions')
    nodes={lane(x) for x in topology['node_keys']}
    rev=collections.defaultdict(list)
    for edge in topology['directed_edges']:
        source,target=lane(edge['from']),lane(edge['to'])
        need(source in nodes and target in nodes,'invalid longitudinal edge')
        rev[target].append((source,'longitudinal',None))
    # Lateral transitions are conditional: only where sampled markings permit.
    for interval in lateral['intervals']:
        source,target=lane(interval['from_lane']),lane(interval['to_lane'])
        need(source in nodes and target in nodes,'lateral source/target not in graph')
        need(interval.get('source_marking_permission_checked') is True and
             interval.get('geometry_checks_passed_at_samples') is True,'unverified lateral marking')
        need(float(interval['source_s_max'])>float(interval['source_s_min']),'empty lateral s interval')
        rev[target].append((source,'lateral_marking_samples',
            {'s_min':float(interval['source_s_min']),'s_max':float(interval['source_s_max'])}))
    geo_by={int(r['record_index']):r for r in geometry['records']}
    ranges={lane(r['lane']):(float(r['s_min']),float(r['s_max']))
            for r in geometry['sampled_lane_station_ranges']}
    records=[]
    for c in conditions:
        if c['town']!=topology['town']:continue
        idx=int(c['record_index']); route={lane(r) for r in c['ego_route']}
        need(idx in geo_by,'geometry not sampled for record %d'%idx)
        anchors={}
        for r in c['ego_route']:
            k=lane(r)
            if k in nodes:
                anchors[k]={'lane':list(k),'type':'ego_route_same_lane_following',
                            'ego_route_index':None,'crossing_3d_checked':False}
        for k in route:
            if k not in nodes:continue
            incoming={x for x,kind,_ in rev[k] if kind=='longitudinal'}
            if len(incoming)>=2:
                anchors[k]={'lane':list(k),'type':'topology_merge_candidate',
                            'ego_route_index':None,'crossing_3d_checked':False}
        for a in geo_by[idx]['crossing_anchor_candidates']:
            k=lane(a['other_lane'])
            if k in nodes:
                anchors[k]={'lane':list(k),'type':'3d_geometric_crossing_candidate',
                    'ego_route_index':int(a['ego_route_index']),
                    'crossing_3d_checked':True}
        candidate={}
        for anchor,meta in sorted(anchors.items()):
            queue=collections.deque([(anchor,0,[list(anchor)],[],None)])
            seen={anchor}
            while queue:
                k,h,path,edge_types,last_lateral=queue.popleft()
                interval=ranges.get(k)
                if interval and interval[1]-interval[0]>0:
                    val={'lane':list(k),'anchor_lane':list(anchor),'anchor_type':meta['type'],
                         'distance_hops_to_anchor':h,'lane_path_to_anchor':path,
                         'path_edge_types':edge_types,'source_s_range_observed':[interval[0],interval[1]],
                         'lateral_transition_s_bounds':last_lateral,
                         'GT_used':False,'physical_spawn_verified':False,
                         'continuous_lane_change_trajectory_verified':False}
                    if k not in candidate or h<candidate[k]['distance_hops_to_anchor']:
                        candidate[k]=val
                if h>=max_hops:continue
                for source,typ,bound in rev.get(k,[]):
                    if source in seen:continue
                    seen.add(source)
                    queue.append((source,h+1,[list(source)]+path,[typ]+edge_types,
                                  bound if typ.startswith('lateral') else last_lateral))
        records.append({'record_index':idx,'town':topology['town'],
            'anchors':list(anchors.values()),'approaches':sorted(candidate.values(),key=lambda r:tuple(r['lane'])),
            'route_lanes_missing_from_topology':[list(x) for x in sorted(route-nodes)],
            'not_fully_legally_spawnable':True})
    need(records,'no target Town condition rows')
    return {'schema':'p2_gtfree_approach_support_v1','town':topology['town'],
        'records':records,'GT_used_in_make':False,'formal_110_recall_calculated':False,
        'native_interaction_proven':False,'spawn_proven':False,
        'limitations':['lane-level upstream reachability','lateral permissions are sampled not trajectory proof',
                       '3d crossings not dynamic RIFT interaction proof','candidate s range is observed not certified spawn range']}

def evaluate(doc,gt_rows,formal110):
    byidx={int(r['record_index']):r for r in doc['records']}
    actual=[];seen=set()
    for g in gt_rows:
        idx=int(g['record_index']);actor=str(g['actor_id'])
        key=(idx,actor);need(key not in seen,'duplicate GT record/actor identity');seen.add(key)
        if idx not in byidx:continue
        selected=byidx[idx];k=g.get('actor_lane') or g.get('lane')
        actual_lanes={tuple(x['lane']) for x in selected['approaches']}
        covered=(tuple(int(v) for v in k) in actual_lanes) if k is not None else None
        actual.append({'record_index':idx,'actor_id':actor,'actor_lane':k,
            'lane_identity_hit':covered,'is_approach_recall_pass':False,
            'uncovered_reason':('lane_not_reached' if covered is False else
                                'lane_id_not_verified' if covered is None else None)})
    if formal110:
        need(len(gt_rows)==110 and len(actual)==110,'fixed110 must contain 110 fully evaluated unique targets')
        need(all(r['actor_lane'] is not None for r in actual),'fixed110 contains unknown actor lane')
    cnt=collections.Counter('hit' if x['lane_identity_hit'] is True else 'miss' if x['lane_identity_hit'] is False else 'unknown'
                             for x in actual)
    return {'schema':'p2_lane_approach_diagnostic_v1','town':doc['town'],
        'counts':dict(cnt),'evaluated_gt':len(actual),'formal_fixed_110_input':bool(formal110),
        'formal_GT_approach_recall_proven':False,
        'lane_identity_hit_fraction':(cnt['hit']/(cnt['hit']+cnt['miss']) if cnt['hit']+cnt['miss'] else None),
        'cases':actual}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('make');a.add_argument('--topology',required=True);a.add_argument('--conditions',required=True)
    a.add_argument('--lateral',required=True);a.add_argument('--geometry',required=True)
    a.add_argument('--max-hops',type=int,default=12);a.add_argument('--output',required=True)
    b=sub.add_parser('evaluate');b.add_argument('--candidates',required=True)
    group=b.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe',help='CARLA topology GT diagnostic, NOT fixed110')
    group.add_argument('--fixed110',help='original frozen fixed110 rows, each with actor_lane')
    b.add_argument('--output',required=True)
    a=p.parse_args()
    try:
        if a.cmd=='make':
            g=read(a.topology);l=read(a.lateral);geo=read(a.geometry)
            need(g.get('schema')=='p2_lane_topology_probe_v1','unsupported CARLA topology schema')
            need(l.get('schema')=='p2_lateral_legal_segment_census_v0','unsupported lateral schema')
            need(geo.get('schema')=='p2_gt_free_map_geometry_v1','unsupported geometry schema')
            doc=make(g,jl(a.conditions),l,geo,a.max_hops)
            doc['source_sha256']={k:sha(getattr(a,k)) for k in ('topology','lateral','geometry','conditions')}
        else:
            doc=read(a.candidates)
            need(doc.get('schema')=='p2_gtfree_approach_support_v1','invalid candidates schema')
            if a.probe:
                probe=read(a.probe);need(probe['town']==doc['town'],'GT probe wrong Town')
                rows=[{'record_index':r['record_index'],'actor_id':t['actor_id'],'actor_lane':t.get('actor_lane')}
                      for r in probe['records'] for t in r.get('gt_targets',[])]
            else: rows=jl(a.fixed110)
            doc=evaluate(doc,rows,bool(a.fixed110));doc['candidate_sha256']=sha(a.candidates)
            doc['GT_source_sha256']=sha(a.fixed110 or a.probe)
        write(a.output,doc)
        if a.cmd=='make':
            print('P2_ANCHORS='+str(sum(len(r['anchors']) for r in doc['records'])))
            print('P2_APPROACH_RECORDS='+str(len(doc['records'])))
        else:print('P2_EVAL_COUNTS='+json.dumps(doc['counts']))
        print('P2_OUTPUT='+str(a.output))
        print('P2_FULL_APPROACH_RECALL_AND_SPAWN_GATE_NOT_PROVEN')
        return 0
    except (EvidenceError,KeyError,ValueError,TypeError,OSError) as e:
        print('P2_FAIL_CLOSED='+repr(e),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
