#!/usr/bin/env python3
"""Read-only V5 eligibility counterfactuals. Never writes training data/checkpoints.

Purpose: distinguish too-weak *labels* from wrong candidate/chart assumptions.
Counterfactual thresholds are diagnostic sensitivity, NOT deployment/training policy.
All actual candidates are map+ego-route-only; RIFT labels are read exclusively for
post hoc comparison with GT. Some ablations use >32 candidates and therefore
are NOT compatible with the current trained V5 network.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import numpy as np

from high_value_init.v5_candidates import make_candidates, assign_pair, closest, get_lanes, _array


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--train-jsonl')
    p.add_argument('--val-jsonl')
    p.add_argument('--test-jsonl')
    p.add_argument('--jsonl', action='append', help='Additional arbitrary named input; can repeat')
    p.add_argument('--output-json',required=True)
    p.add_argument('--max-records-per-split',type=int,default=2500,help='0 = all')
    p.add_argument('--min-rift-score',type=float,default=.4)
    p.add_argument('--max-gt-lane-distance',type=float,default=1.25)
    p.add_argument('--max-anchor-distance',type=float,default=12.)
    return p.parse_args()


def positive_metrics(states, labels):
    if len(states)!=2 or len(labels)!=2:
        return None
    try:
        scores=[float(l['interaction_strength']) for l in labels]
        gaps=[float(l['route_distance_difference_m']) for l in labels]
        thresholds=[float(l['route_distance_difference_threshold_m']) for l in labels]
        anchors=[l['conflict_point_local_xy'] for l in labels]
        xyavs=[[float(v) for v in s[:4]] for s in states]
        if any(len(a)<2 for a in anchors):return None
        values=scores+gaps+thresholds+[v for s in xyavs for v in s]
        if not all(math.isfinite(v) for v in values): return None
        if any(t<=0 or g<0 or g>t or s<0 or s>1 for s,g,t in zip(scores,gaps,thresholds)):
            return None
        return scores,gaps,thresholds,xyavs
    except (ValueError,TypeError,KeyError,IndexError):
        return None


def raw_lane_match(lanes, state, max_pos, max_yaw):
    """Ego-local raw map lane compatibility, before V5 candidate construction."""
    nearest=float('inf'); best_dh=float('inf'); along_heading=False; samples=0
    for lane in lanes:
        poly=_array(lane)
        if poly is None:continue
        samples+=1
        d,s,h,p=closest(poly,state[:2])
        nearest=min(nearest,d)
        e=abs(math.atan2(math.sin(state[2]-h),math.cos(state[2]-h)))
        if d<=max_pos:
            best_dh=min(best_dh,e)
            if e<=max_yaw:along_heading=True
    return nearest, best_dh, along_heading, samples


def assignment(candidates, states, labels, max_pos=1.25, max_heading=.38, max_anchor=12.):
    if not candidates:return False
    return assign_pair(candidates,states,labels,max_pos_m=max_pos,
                       max_heading_rad=max_heading,max_anchor_m=max_anchor) is not None


def fmt_fraction(x,y):return round(x/y,5) if y else None


def process_one(name,path,args):
    count=Counter();towns=defaultdict(Counter);examples=defaultdict(list)
    hist_min_score=[0,0,0,0,0]
    vals={'min_rift_strength':[],'max_target_speed_mps':[],
          'closest_raw_lane_m':[],'target_anchor_to_closest_candidate_m':[],
          'candidates_32':[], 'rift_anchor_ego_route_arclength_m':[],
          'rift_anchor_distance_to_ego_route_m':[]}
    with open(path,encoding='utf-8') as f:
      for row_idx,line in enumerate(f):
        if args.max_records_per_split>0 and row_idx>=args.max_records_per_split:break
        if not line.strip():continue
        d=json.loads(line);count['total']+=1
        cond=d.get('condition') or {};tar=d.get('target') or {}
        states=tar.get('generated_agent_states') or []
        labels=tar.get('route_interaction_supervision') or []
        parsed=positive_metrics(states,labels)
        if parsed is None:
            count['invalid_two_target_rift_labels']+=1
            continue
        count['valid_rift_labels']+=1
        scores,gaps,thresholds,ss=parsed
        strong=min(scores)>=args.min_rift_score
        vals['min_rift_strength'].append(min(scores)); vals['max_target_speed_mps'].append(max(s[3] for s in ss))
        for t in (0.,.2,.4,.6,.8):
            if min(scores)>=t:count['score_pair_ge_%s'%t]+=1
        if not strong:
            count['failed_score_gate']+=1
            continue
        count['passed_score_gate']+=1
        if all(0<=s[3]<12 for s in ss):count['both_speed_lt_12']+=1
        if all(0<=s[3]<25 for s in ss):count['both_speed_lt_25']+=1
        if all(10<=math.hypot(s[0],s[1])<=45 for s in ss):count['both_radius_10_45']+=1
        town=str(d.get('town') or 'unknown');top=str((d.get('topology_v3') or {}).get('bucket') or 'unknown')
        towns[town]['strong']+=1
        if any(not 0<=s[3]<12 for s in ss):towns[town]['speed_outside_12']+=1
        lane_rows=get_lanes(cond)
        raw=[raw_lane_match(lane_rows,s,args.max_gt_lane_distance,.38) for s in ss]
        if any(n==0 for _,_,_,n in raw):count['raw_map_lanes_missing']+=1
        for near,e,correct,_ in raw:
            if math.isfinite(near):vals['closest_raw_lane_m'].append(near)
        if all(r[0]<=args.max_gt_lane_distance for r in raw):count['both_near_raw_hdmap_centerline']+=1
        if all(r[2] for r in raw):count['both_heading_match_raw_hdmap']+=1
        route=_array((cond.get('ego_route') or {}).get('polyline') or [])
        if route is not None:
            for lab in labels:
                anchor=lab.get('conflict_point_local_xy') or []
                if len(anchor)>=2:
                    dist,progress,angle,point=closest(route, anchor[:2])
                    vals['rift_anchor_ego_route_arclength_m'].append(progress)
                    vals['rift_anchor_distance_to_ego_route_m'].append(dist)
                    if progress >31.:count['rift_anchor_beyond_v5_31m_route_window']+=1
        c32=make_candidates(cond,max_candidates=32)
        c96=make_candidates(cond,max_candidates=96)
        vals['candidates_32'].append(len(c32))
        if not c32:count['no_map_route_candidates']+=1
        if c32:count['has_map_route_candidates']+=1
        trials={
            'baseline_32': (c32,1.25,.38,12.),
            'candidate_cap_96_diagnostic_only': (c96,1.25,.38,12.),
            'anchor_24m_diagnostic_only': (c32,1.25,.38,24.),
            'anchor_unbounded_diagnostic_only': (c32,1.25,.38,1.e9),
            'lane_dist_2m_diagnostic_only': (c32,2.,.38,12.),
            'heading_0p65_diagnostic_only': (c32,1.25,.65,12.),
            'joint_relax_diagnostic_only': (c96,2.,.65,1.e9),
        }
        selected={}
        for k,(c,dp,dy,da) in trials.items():
            ok=assignment(c,ss,labels,max_pos=dp,max_heading=dy,max_anchor=da)
            selected[k]=ok
            if ok:count['assign_'+k]+=1
        if all(0<=s[3]<12 for s in ss) and all(10<=math.hypot(s[0],s[1])<=45 for s in ss):
            count['strong_speed_radius_eligible']+=1
            if selected['baseline_32']:count['strong_speed_radius_baseline_assign']+=1
        if not selected['baseline_32']:
            count['strong_candidate_failure']+=1
            if len(examples['candidate_failure'])<12:
                anchors=[]
                for a in labels:
                    anchors.append(a.get('conflict_point_local_xy'))
                examples['candidate_failure'].append({
                    'scene_id':d.get('scene_id'),'town':town,'topology':top,
                    'score':[round(x,3) for x in scores],
                    'states':[[round(x,2) for x in s] for s in ss],
                    'rift_anchors':anchors,
                    'nearest_map_m':[round(r[0],2) if math.isfinite(r[0]) else None for r in raw],
                    'heading_to_map_rad':[round(r[1],2) if math.isfinite(r[1]) else None for r in raw],
                    'candidate_count':len(c32),
                    'sensitivity_successes':[k for k,v in selected.items() if v]})
        if selected['baseline_32']:towns[town]['assign_baseline']+=1
    summary={'name':name,'source_jsonl':str(path),
             'max_records_per_split':args.max_records_per_split,
             'counts':dict(sorted(count.items())),
             'rates':{k:fmt_fraction(v,count['passed_score_gate']) for k,v in count.items()
                      if k.startswith(('assign_','both_','strong_candidate_failure','no_map_route_candidates'))},
             'by_town':{t:dict(c) for t,c in sorted(towns.items())},
             'numeric_distributions':{k:({'mean':float(np.mean(v)),'p50':float(np.percentile(v,50)),
                                        'p90':float(np.percentile(v,90)),'max':float(np.max(v))} if v else None)
                                      for k,v in vals.items()},
             'failure_examples':dict(examples),
             'warnings':[
                'Assignment ablations read GT RIFT labels for DIAGNOSTICS ONLY, never online candidate construction.',
                'candidate_cap_96 is not compatible with existing V5 network MAX_CANDIDATES=32.',
                'Relaxed thresholds do NOT establish CARLA or RIFT native interaction feasibility.',
                'Only labels that are already present in V4C datasets are evaluated; no resampling or data rewriting.',
             ]}
    return summary


def main():
    args=arguments(); paths={}
    for split in ('train','val','test'):
        p=getattr(args,split+'_jsonl')
        if p:paths[split]=p
    for i,p in enumerate(args.jsonl or []):paths['custom_'+str(i)]=p
    if not paths:raise SystemExit('Provide at least one --*-jsonl / --jsonl')
    report={'schema_version':'hv_v5_candidate_coverage_diagnostic_v1',
            'configured_score_gate':args.min_rift_score,
            'configured_lane_distance_m':args.max_gt_lane_distance,
            'configured_anchor_distance_m':args.max_anchor_distance,
            'no_training_or_state_modification':True,'splits':{}}
    for name,path in paths.items():
        print('Profiling',name,str(path),flush=True)
        v=process_one(name,path,args)
        report['splits'][name]=v
        c=v['counts']
        print(name,'total=',c.get('total'),'strong=',c.get('passed_score_gate'),
              'map_orient=',c.get('both_heading_match_raw_hdmap'),
              'candidate32=',c.get('assign_baseline_32'),
              'candidate96=',c.get('assign_candidate_cap_96_diagnostic_only'),flush=True)
    out=Path(args.output_json);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Written:',out)


if __name__=='__main__':main()
