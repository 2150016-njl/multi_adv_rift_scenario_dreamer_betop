#!/usr/bin/env python3
"""P2 *condition-only* lane-level upstream support baseline. NO GT in make().

Input: genuine CARLA topology probe graph (nodes, longitudinal directed edges)
and P2 CONDITIONS ONLY. Output is lane identity support, NOT collision/approach
geometry, lateral lane-change permissions, or 110-front-GT recall.
Evaluation is a separate subcommand; its outputs are never consumed by make().
Python 3.8+ stdlib.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

class EvidenceError(RuntimeError):pass

def need(ok,msg):
    if not ok:raise EvidenceError(msg)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def readjl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write_new(p,obj):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
def lane_key(obj):
    if isinstance(obj,list):
        need(len(obj)==3,'malformed lane key')
        return tuple(int(x) for x in obj)
    return (int(obj['road_id']),int(obj['section_id']),int(obj['lane_id']))

def upstream(reverse,goal,depth):
    # shortest directed upstream hops from anchor, restricted to first occurrence
    q=collections.deque([(goal,0)]);seen={goal:0}
    while q:
        k,h=q.popleft()
        if h>=depth:continue
        for prev in sorted(reverse.get(k,())):
            if prev not in seen:
                seen[prev]=h+1;q.append((prev,h+1))
    return seen

def make(graph_doc,condition_rows,max_hops=8):
    need(graph_doc.get('schema')=='p2_lane_topology_probe_v1','requires genuine CARLA p2 topology report')
    town=graph_doc['town'];nodes={lane_key(x) for x in graph_doc['node_keys']}
    need(nodes and graph_doc.get('directed_edges'),'no directed CARLA topology')
    reverse=collections.defaultdict(set)
    for e in graph_doc['directed_edges']:
        a,b=lane_key(e['from']),lane_key(e['to'])
        need(a in nodes and b in nodes,'edge outside node set')
        reverse[b].add(a)
    output=[];seen_indices=set()
    for row in condition_rows:
        if row.get('town')!=town:continue
        idx=int(row['record_index'])
        need(idx not in seen_indices,'duplicate condition record index')
        seen_indices.add(idx)
        route=row.get('ego_route')
        need(isinstance(route,list) and bool(route),'missing ego route')
        route_keys={lane_key(x) for x in route}
        in_graph=route_keys & nodes
        # Candidate entry anchors from route and its directed predecessors.
        # A conflict/merge is not yet proven: this is an upper-bound support set.
        candidate_lanes={}
        anchor_rows=[]
        for anchor in sorted(in_graph):
            access=upstream(reverse,anchor,max_hops)
            upstream_entries=sorted(x for x in reverse.get(anchor,set()) if x not in route_keys)
            for key,h in access.items():
                if key not in candidate_lanes or h<candidate_lanes[key]:candidate_lanes[key]=h
            anchor_rows.append({'anchor_lane':list(anchor),'incoming_non_route_lanes':[list(x) for x in upstream_entries],
                                'reachable_upstream_lane_count':len(access),
                                'is_interaction_conflict_verified':False})
        output.append({'record_index':idx,'town':town,'manifest_scene_id':row.get('manifest_scene_id'),
             'ego_route_lane_count':len(route_keys),'route_lane_keys_not_in_topology':[list(x) for x in sorted(route_keys-nodes)],
             'anchor_rows':anchor_rows,
             'candidate_lane_hops':[{'lane':list(k),'hops_to_route_anchor':h} for k,h in sorted(candidate_lanes.items())],
             'has_geometric_anchor':False,'has_legal_lateral_edges':False,'has_validated_spawn_intervals':False})
    need(output,'no conditions for graph Town '+town)
    return {'schema':'p2_directional_lane_support_v0','town':town,'GT_used_in_generator':False,
        'approach_recall_110_computed':False,'legally_spawnable_support_proven':False,
        'only_longitudinal_topology':True,'max_lane_hops':max_hops,'records':output}

def cmd_make(a):
    need(a.max_hops>=1 and a.max_hops<=32,'max hops out of bounds')
    g=readj(a.topology);need(g['town']==a.town,'Town mismatch')
    # deliberately never access g['records'] containing GT diagnostics
    c=readjl(a.conditions)
    doc=make(g,c,a.max_hops)
    doc['topology_sha256']=sha(a.topology);doc['conditions_sha256']=sha(a.conditions)
    write_new(a.output,doc)
    print('P2_CONDITION_ONLY_RECORDS='+str(len(doc['records'])))
    print('P2_LANE_SUPPORT_REPORT='+str(Path(a.output).resolve()))
    print('P2_NOT_110_GT_RECALL_NOT_CONFLICT_ANCHOR_NOT_SPAWN_PASS')
    return 0

def cmd_eval(a):
    cand=readj(a.candidates);probe=readj(a.topology)
    need(cand.get('schema')=='p2_directional_lane_support_v0','candidate schema mismatch')
    need(cand.get('topology_sha256')==sha(a.topology),'evaluating different topology')
    # The GT block is read ONLY during evaluation, after fixed candidate file exists.
    by_idx={int(r['record_index']):r for r in cand['records']}
    totals=collections.Counter();cases=[]
    for row in probe['records']:
        if int(row['record_index']) not in by_idx:continue
        pred=by_idx[int(row['record_index'])]
        keys={tuple(x['lane']) for x in pred['candidate_lane_hops']}
        for t in row.get('gt_targets',[]):
            lane=t.get('actor_lane');valid=lane is not None
            covered=bool(valid and tuple(lane) in keys)
            totals['all_diagnostic_GT']+=1
            totals['evaluable']+=int(valid)
            totals['covered_lane_identity']+=int(covered)
            totals['unknown_lane']+=int(not valid)
            cases.append({'record_index':int(row['record_index']),'actor_id':t['actor_id'],
                          'actor_lane':lane,'candidate_lane_membership':covered if valid else None,
                          'status':'lane_support_hit' if covered else ('unknown_lane' if not valid else 'lane_support_miss')})
    need(cases,'no GT diagnostic rows to evaluate: wrong Town or graph')
    result={'schema':'p2_lane_support_diagnostic_v0','town':cand['town'],'topology_sha256':sha(a.topology),
            'candidates_sha256':sha(a.candidates),'counts':dict(totals),
            'lane_support_hit_rate_diagnostic':totals['covered_lane_identity']/totals['evaluable'] if totals['evaluable'] else None,
            'GT_used_only_after_candidates_frozen':True,'is_formal_110_GT_approach_recall':False,
            'is_legal_road_state_recall':False,'cases':cases}
    write_new(a.output,result)
    print('P2_LANE_DIAGNOSTIC='+json.dumps(result['counts']))
    print('P2_EVAL_REPORT='+str(Path(a.output).resolve()))
    print('P2_NOT_110_GT_RECALL_NOT_MODEL3_PHYSICAL_PASS')
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('make');a.add_argument('--topology',required=True);a.add_argument('--conditions',required=True)
    a.add_argument('--town',required=True);a.add_argument('--max-hops',type=int,default=8);a.add_argument('--output',required=True)
    b=sub.add_parser('evaluate');b.add_argument('--topology',required=True);b.add_argument('--candidates',required=True);b.add_argument('--output',required=True)
    args=p.parse_args()
    try:return cmd_make(args) if args.cmd=='make' else cmd_eval(args)
    except (EvidenceError,ValueError,KeyError,TypeError,FileNotFoundError) as e:
        print('P2_BASELINE_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
