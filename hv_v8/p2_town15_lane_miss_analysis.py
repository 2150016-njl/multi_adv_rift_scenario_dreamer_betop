#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline diagnostic: distinguish longitudinal lane graph misses from adjacent-lane cases.

Reads an existing ground-truth evaluation report; NEVER treats observed GT anchors
as input to an online generator. Does not use CARLA or touch a simulation.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json
from pathlib import Path

def analyze(data):
    counts=collections.Counter()
    out=[]
    for rec in data['records']:
        for item in rec['gt_targets']:
            start=item.get('actor_lane')
            anc=item.get('gt_anchor_verification') or {}
            end=anc.get('lane') if anc.get('status')=='verified' else None
            reaches=item.get('directed_lane_graph_reaches_gt_anchor')
            neighbor=(start is not None and end is not None
                      and len(start)==len(end)==3
                      and start[:2]==end[:2]
                      and abs(start[2]-end[2])==1)
            if reaches is True:
                category='longitudinal_reachability_true'
            elif reaches is False and neighbor:
                category='adjacent_lane_lateral_rule_not_evaluated'
            elif reaches is False:
                category='other_longitudinal_nonreachability'
            else:
                category='not_evaluable'
            counts[category]+=1
            out.append({'ordinal':rec['ordinal'], 'record_index':rec['record_index'],
                        'actor_id':item['actor_id'], 'actor_lane':start,
                        'gt_anchor_lane':end,'longitudinal_reachable':reaches,
                        'coarse_category':category,
                        'lane_change_legality':'NOT_EVALUATED',
                        'spatial_approach_coverage':'NOT_EVALUATED',
                        'p2_gt_approach_recalled':'NOT_EVALUATED'})
    nonreach_by_actor=collections.defaultdict(list)
    for r in out:
        if r['coarse_category']=='adjacent_lane_lateral_rule_not_evaluated':
            nonreach_by_actor[str(r['actor_id'])].append(r['record_index'])
    repeated=[{'actor_id':aid, 'record_indices':sorted(v), 'occurrences':len(v)}
              for aid,v in sorted(nonreach_by_actor.items())]
    return {'source_schema':data.get('schema'),
            'source_town':data.get('town'), 'source_sha256':None,
            'gt_target_count':len(out),'counts':dict(sorted(counts.items())),
            'adjacent_lane_nonreach_actor_id_groups':repeated,
            'is_P2_recall':False,'results':out,
            'interpretation':'Adjacent-lane classification does not prove legal lane changes, a continuous approach, or a safe initial spawn.'}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--probe-report',required=True)
    a.add_argument('--out-prefix',required=True)
    args=a.parse_args()
    source=Path(args.probe_report)
    data=json.loads(source.read_text(encoding='utf-8'))
    assert data.get('schema')=='p2_lane_topology_probe_v1' and data.get('scientific_P2_pass') is False
    report=analyze(data)
    report['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    prefix=Path(args.out_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    jp=Path(str(prefix)+'.json'); cp=Path(str(prefix)+'.csv')
    with jp.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    with cp.open('x',encoding='utf-8',newline='') as f:
        cols=['ordinal','record_index','actor_id','actor_lane','gt_anchor_lane',
              'longitudinal_reachable','coarse_category','lane_change_legality',
              'spatial_approach_coverage','p2_gt_approach_recalled']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for row in report['results']:
            w.writerow({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],list) else row[k] for k in cols})
    print('P2_DIAGNOSTIC_COUNTS='+json.dumps(report['counts'],ensure_ascii=False))
    print('P2_DIAGNOSTIC_JSON='+str(jp))
    print('P2_DIAGNOSTIC_CSV='+str(cp))
    print('P2_NOT_LATERAL_LEGALITY_NOT_APPROACH_RECALL')

if __name__=='__main__':main()
