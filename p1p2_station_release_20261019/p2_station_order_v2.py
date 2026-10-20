#!/usr/bin/env python3
"""GT-free s-order *necessary-condition* screen for existing P2 V1 lane paths.

Inputs: frozen ego-route condition JSONL, REAL CARLA lane-s range census, and
V1 GT-free candidate paths. Not an OpenDRIVE lane-link proof, a continuous
lane-change proof, a native interaction test, or a spawn feasibility test.

Stage 'make' DOES NOT read a GT file. Stage 'evaluate' reads topology GT probe
ONLY AFTER a complete immutable, hashed screening candidate file exists.
Python 3.8+ / standard library only.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, sys
from pathlib import Path

class ContractError(RuntimeError): pass

def need(test, reason):
    if not test: raise ContractError(reason)

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def jsonl(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lane(obj):
    if isinstance(obj,(list,tuple)):
        need(len(obj)==3,'expected 3-item lane key')
        return tuple(int(v) for v in obj)
    return tuple(int(obj[k]) for k in ('road_id','section_id','lane_id'))
def mark(path,doc):
    p=Path(path);need(not p.exists(),'refuse overwrite '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(doc,f,indent=2,ensure_ascii=False,sort_keys=True); f.write('\n')
def safe_range(x):
    lo,hi=(float(x[0]),float(x[1]))
    need(math.isfinite(lo) and math.isfinite(hi) and lo<=hi,'invalid station interval')
    return lo,hi

def intersect(*parts):
    lo=max(a for a,b in parts); hi=min(b for a,b in parts)
    return (lo,max(lo,hi)) if hi+1.e-6>=lo else None

def upstream(s_min,s_max,goal,positive_s):
    """All source stations that could reach any goal in direction of travel."""
    return intersect((s_min,s_max),(s_min,goal[1]) if positive_s else (goal[0],s_max))

def anchor_for_path(cond, path):
    """Bind previously lane-only anchor to actual ego route waypoint index/s.
    Choose last future sample on that anchor lane; log that this is a broad
    same-lane *support* anchor, not a verified conflict point.
    """
    ak=lane(path['anchor_lane'])
    picks=[(i,r) for i,r in enumerate(cond['ego_route']) if lane(r)==ak]
    if not picks:return None
    i,r=picks[-1]
    return {'ego_route_index':i,'ego_s':float(r['s']),'ego_xyz':[float(r[k]) for k in ('x','y','z')],
            'anchor_lane':list(ak),'anchor_semantics':'last_sample_on_ego_route_lane_NOT_conflict_verified',
            'is_junction':bool(r.get('is_junction',False))}

def verify_path(path, anchor, ranges, longitudinal, lateral_by, contact_by=None):
    contact_by=contact_by or {}
    ks=[lane(x) for x in path['lane_path_to_anchor']]
    types=path['path_edge_types']
    need(len(ks)>=1 and len(types)==len(ks)-1,'malformed V1 path')
    need(ks[0]==lane(path['lane']) and ks[-1]==lane(path['anchor_lane']),
         'source/anchor path lane mismatch')
    if anchor is None:
        return {'status':'anchor_not_on_exact_route','station_screen_possible':False,
                'reason':'no_exact_ego_waypoint_for_anchor_lane','edges':[]}
    if ks[-1] not in ranges:
        return {'status':'unknown_missing_anchor_samples','station_screen_possible':None,
                'reason':'no_station_samples_at_anchor','edges':[]}
    a_s=float(anchor['ego_s']);need(math.isfinite(a_s),'nonfinite ego anchor s')
    # CARLA sampled s-extents are observations, not true OpenDRIVE road limits.
    # Ego snapshot s is verified separately and may be up to spacing outside the samples.
    ga=ranges[ks[-1]]; tol=3.0
    if a_s < ga[0]-tol or a_s > ga[1]+tol:
        return {'status':'anchor_station_outside_sampled_lane','station_screen_possible':None,
                'reason':'anchor station absent from sampled s range','edges':[]}
    feasible=upstream(ga[0],ga[1],(a_s,a_s),ks[-1][2]<0)
    if feasible is None:
        return {'status':'unknown_anchor_before_sampled_station_start',
                'station_screen_possible':None,
                'reason':'verified_ego_anchor_s_not_in_sampled_upstream_station_support',
                'edges':[]}
    edges=[]; unknown=False
    for j in range(len(types)-1,-1,-1):
        src,dst=ks[j],ks[j+1];typ=types[j]
        src_range,dst_range=ranges.get(src),ranges.get(dst)
        if src_range is None or dst_range is None:
            return {'status':'unknown_missing_lane_station_samples','station_screen_possible':None,
                    'reason':'source_or_target_lane_not_sampled','edges':list(reversed(edges))}
        positive=(src[2]<0)
        # For the single mapped road's right/left traffic orientation, lanes
        # with same sign have same station travel sign; no opposite-direction LC.
        if typ=='lateral_marking_samples':
            if src[:2]!=dst[:2] or src[2]*dst[2]<=0:
                return {'status':'invalid_lateral_identity','station_screen_possible':False,
                        'reason':'lateral_edge_not_same_road_section_and_direction','edges':list(reversed(edges))}
            matches=lateral_by.get((src,dst),[])
            if not matches:
                return {'status':'lateral_interval_not_in_real_census','station_screen_possible':False,
                        'reason':'missing_authorized_marking_interval','edges':list(reversed(edges))}
            usable=[]
            for e in matches:
                if not (e.get('source_marking_permission_checked') is True and
                        e.get('geometry_checks_passed_at_samples') is True):continue
                s=safe_range((e['source_s_min'],e['source_s_max']))
                possible=intersect(s,src_range,dst_range,feasible)
                if possible:usable.append(possible)
            if not usable:
                return {'status':'station_order_conflict','station_screen_possible':False,
                        'reason':'legal_lateral_interval_not_reachable_before_downstream_constraint',
                        'failed_edge':j,'edges':list(reversed(edges))}
            # union over valid switching windows, sufficient only to keep an
            # existential *necessary* station ordering interval, not a trajectory.
            switch=(min(x[0] for x in usable),max(x[1] for x in usable))
            out=upstream(src_range[0],src_range[1],switch,positive)
            if out is None:
                return {'status':'station_order_conflict','station_screen_possible':False,
                        'reason':'source_station_range_not_upstream_of_lateral','failed_edge':j,
                        'edges':list(reversed(edges))}
            edges.append({'from_lane':list(src),'to_lane':list(dst),'type':typ,
                          'admissible_lateral_station_window':list(switch),
                          'continuous_marking_interval_proven':False,
                          'continuous_trajectory_proven':False})
            feasible=out
        elif typ=='longitudinal':
            if (src,dst) not in longitudinal:
                return {'status':'longitudinal_edge_missing_from_frozen_graph','station_screen_possible':False,
                        'reason':'path_edge_not_in_CARLA_topology','failed_edge':j,'edges':list(reversed(edges))}
            contact_windows=[]
            # Optional REAL CARLA next() source->target station samples. If
            # sample is not present, no fabricated boundary station is used.
            for c in contact_by.get((src,dst),[]):
                sc=float(c['source']['s']); tc=float(c['target']['s'])
                if intersect((tc,tc),feasible,dst_range) is None:continue
                up=upstream(src_range[0],src_range[1],(sc,sc),positive)
                if up is not None:contact_windows.append((up,sc,tc))
            if contact_windows:
                feasible=(min(t[0][0] for t in contact_windows),max(t[0][1] for t in contact_windows))
                edges.append({'from_lane':list(src),'to_lane':list(dst),'type':typ,
                              'local_CARLA_next_contact_samples_compatible':len(contact_windows),
                              'sampled_source_contact_s':[min(t[1] for t in contact_windows),max(t[1] for t in contact_windows)],
                              'sampled_target_contact_s':[min(t[2] for t in contact_windows),max(t[2] for t in contact_windows)],
                              'complete_longitudinal_connectivity_proven':False})
                unknown=True
            else:
                # Absence of a short next() sample is NOT disproof. Same goes
                # for sampled contacts incompatible with the current broad
                # station range: record uncertainty, do not silently FAIL.
                edges.append({'from_lane':list(src),'to_lane':list(dst),'type':typ,
                              'local_CARLA_next_contact_samples_compatible':0,
                              'station_contact_proven':False,
                              'note':'no compatible sampled contact; upstream interval conservatively widened'})
                feasible=src_range
                unknown=True
        else:
            return {'status':'unsupported_edge_type','station_screen_possible':None,
                    'reason':typ,'edges':list(reversed(edges))}
    origin=ranges.get(ks[0])
    if origin is None:
        return {'status':'unknown_missing_origin_samples','station_screen_possible':None,
                'reason':'origin_not_sampled','edges':list(reversed(edges))}
    if len(ks)==1:
        # Source station is upstream of route anchor on exactly the same lane.
        feasible=upstream(origin[0],origin[1],(a_s,a_s),ks[0][2]<0)
        if feasible is None:
            return {'status':'no_upstream_sampled_source_station','station_screen_possible':False,
                    'reason':'ego_route_anchor_upstream_of_all_sampled_stations','edges':[]}
    n_lateral=sum(v=='lateral_marking_samples' for v in types)
    status=('necessary_station_order_with_unverified_longitudinal' if unknown else
            'necessary_station_order_lateral_sampled_only' if n_lateral else
            'same_lane_upstream_station_range_observed')
    return {'status':status,'station_screen_possible':True,
            'origin_station_feasible_observed':list(feasible),
            'longitudinal_station_contact_proven':not unknown,
            'lateral_change_count':n_lateral,
            'all_lateral_samples_continuous_proven':False if n_lateral else None,
            'actor_init_station_verified':False,'native_interaction_verified':False,
            'Model3_spawn_verified':False,'edges':list(reversed(edges))}

def make(topology,conditions,geometry,lateral,approaches,contacts=None):
    town=topology['town']; msha=topology['map_sha256']
    need(geometry['town']==lateral['town']==approaches['town']==town,'TOWN_MISMATCH')
    need(geometry['map_sha256']==lateral['opendrive_sha256']==msha,'MAP_SHA_MISMATCH')
    need(geometry.get('GT_used') is False and lateral.get('GT_used') is False and
         approaches.get('GT_used_in_make') is False,'GT_PROVENANCE_CONTAMINATION')
    inputs={int(c['record_index']):c for c in conditions if c['town']==town}
    need(all(c.get('gt_target_positions_in_condition') is False and
             not any(k in c for k in ('targets','agents','rift_labels')) for c in inputs.values()),
         'GT_IN_P2_MAKE_CONDITION')
    frozen={int(r['record_index']):r for r in approaches['records']}
    need(inputs.keys()==frozen.keys(),'CONDITION_AND_CANDIDATE_IDENTITY_MISMATCH')
    ranges={lane(r['lane']):safe_range([r['s_min'],r['s_max']]) for r in geometry['sampled_lane_station_ranges']}
    long_edges={(lane(e['from']),lane(e['to'])) for e in topology['directed_edges']}
    contact_by=collections.defaultdict(list)
    if contacts is not None:
        need(contacts.get('schema')=='p2_CARLA_local_directed_contact_samples_v1' and
             contacts.get('town')==town and contacts.get('map_sha256')==msha and
             contacts.get('GT_used') is False,
             'CARLA_CONTACT_SOURCE_OR_MAP_NOT_VERIFIED')
        for e in contacts['samples']:
            edge=(lane(e['from_lane']),lane(e['to_lane']))
            need(edge in long_edges,'contact edge absent in pinned graph')
            contact_by[edge].append(e)
    lat_by=collections.defaultdict(list)
    for e in lateral['intervals']:
        lat_by[(lane(e['from_lane']),lane(e['to_lane']))].append(e)
    records=[]
    for idx in sorted(frozen):
        selected=frozen[idx]; cond=inputs[idx]; checks=[]
        for p in selected['approaches']:
            an=anchor_for_path(cond,p)
            ck=verify_path(p,an,ranges,long_edges,lat_by,contact_by)
            checks.append({'lane':p['lane'],'anchor_lane':p['anchor_lane'],
                           'lane_path_to_anchor':p['lane_path_to_anchor'],
                           'path_edge_types':p['path_edge_types'],
                           'ego_anchor':an,'station_check':ck})
        records.append({'record_index':idx,'town':town,'candidate_station_checks':checks})
    return {'schema':'p2_gtfree_station_necessary_screen_v2','town':town,
            'map_sha256':msha,'GT_used_in_make':False,
            'all_station_order_and_longitudinal_links_proven':False,
            'fixed110_approach_recall_proven':False,
            'trajectory_continuity_proven':False,
            'Model3_spawn_proven':False,
            'CARLA_contact_samples_used':contacts is not None,
            'records':records}

def evaluate(screen,probe):
    need(screen['town']==probe['town'],'GT_PROBE_TOWN_MISMATCH')
    records={int(r['record_index']):r for r in screen['records']}
    result=[]
    for row in probe['records']:
        idx=int(row['record_index']);need(idx in records,'GT_INDEX_NOT_IN_CANDIDATES')
        for g in row.get('gt_targets',[]):
            key=lane(g['actor_lane']);matching=[c for c in records[idx]['candidate_station_checks']
                                             if lane(c['lane'])==key]
            possible=any(c['station_check']['station_screen_possible'] is True for c in matching)
            unresolved=any(c['station_check']['station_screen_possible'] is None for c in matching)
            result.append({'record_index':idx,'actor_id':str(g['actor_id']),
                           'actor_lane':list(key),'lane_identity_hit':bool(matching),
                           'necessary_station_screen_hit':possible,
                           'has_unknown_station_evidence':unresolved,
                           'candidate_paths_count':len(matching),
                           'native_interaction_proven':False,
                           'full_approach_recall_pass':False})
    cnt=collections.Counter(('screen_hit' if r['necessary_station_screen_hit'] else
                             'unknown' if r['has_unknown_station_evidence'] else 'screen_miss') for r in result)
    return {'schema':'p2_posthoc_station_screen_diagnostic_v2','town':screen['town'],
            'GT_observations':len(result),'station_screen_counts':dict(cnt),
            'fully_validated_approach_count':0,
            'formal_fixed110_recall_proven':False,
            'full_approach_recall_per_target_proven':False,
            'cases':result}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('make');a.add_argument('--contacts',help='optional real CARLA local directed next() station witness')
    for k in ('topology','conditions','geometry','lateral','approaches','output'):
        a.add_argument('--'+k,required=True)
    b=sub.add_parser('evaluate');
    for k in ('screen','probe','output'): b.add_argument('--'+k,required=True)
    x=p.parse_args()
    try:
        if x.cmd=='make':
            doc=make(load(x.topology),jsonl(x.conditions),load(x.geometry),load(x.lateral),load(x.approaches),
                     load(x.contacts) if x.contacts else None)
            doc['input_sha256']={k:digest(getattr(x,k)) for k in
                                 ('topology','conditions','geometry','lateral','approaches')}
            if x.contacts:doc['contact_sha256']=digest(x.contacts)
            mark(x.output,doc)
            counts=collections.Counter(c['station_check']['status'] for r in doc['records'] for c in r['candidate_station_checks'])
            print('P2_STATION_COUNTS='+json.dumps(dict(counts),ensure_ascii=False,sort_keys=True))
            print('P2_STATION_OUTPUT='+x.output)
        else:
            doc=evaluate(load(x.screen),load(x.probe)); doc['screen_sha256']=digest(x.screen);doc['GT_probe_sha256']=digest(x.probe)
            mark(x.output,doc)
            print('P2_STATION_GT26_COUNTS='+json.dumps(doc['station_screen_counts'],sort_keys=True))
            print('P2_STATION_GT26_NOT_FORMAL_APPROACH_RECALL='+str(doc['GT_observations']))
            print('P2_STATION_GT26_OUTPUT='+x.output)
        return 0
    except (ContractError,KeyError,ValueError,TypeError,OverflowError,OSError,IndexError) as e:
        print('P2_STATION_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
