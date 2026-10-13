#!/usr/bin/env python3
"""V5 dataset reuse: target-only RIFT V4C labels; NO new CARLA collection.

Fail closed on missing RIFT interaction supervision. The candidate generator uses
ONLY condition['map'] and condition['ego_route'], never the target annotations.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

from high_value_init.v5_candidates import make_candidates, assign_pair, signature, closest
from high_value_init.v5_model import CARLAJointRawGenerator, candidate_tensors, SIZE_FIXED
from eval_high_value_scene_generator_v5 import _overlap


def args_parser():
    p=argparse.ArgumentParser()
    for name in ('train','val','test'):
        p.add_argument('--%s-jsonl'%name,required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--min-rift-score',type=float,default=0.4)
    p.add_argument('--max-anchor-distance',type=float,default=12.0)
    p.add_argument('--max-gt-lane-distance',type=float,default=1.25)
    p.add_argument('--min-train-records',type=int,default=1000)
    p.add_argument('--min-val-records',type=int,default=100)
    p.add_argument('--min-test-records',type=int,default=100)
    p.add_argument('--max-records-per-split',type=int,default=-1,help='diagnostic-only limit; do not use for final training')
    return p


_ORACLE=None

def _oracle_model():
    global _ORACLE
    if _ORACLE is None:
        _ORACLE=CARLAJointRawGenerator(hidden=16)
    return _ORACLE

def transform(record, args):
    condition=record.get('condition') or {}
    target=record.get('target') or {}
    states=target.get('generated_agent_states') or []
    supervision=target.get('route_interaction_supervision') or []
    if len(states)!=2 or len(supervision)!=2:
        return None,'missing_two_rift_supervised_targets'
    if any(not x or not isinstance(x,dict) or not x.get('conflict_point_local_xy')
           for x in supervision):
        return None,'incomplete_conflict_anchor'
    for label in supervision:
        try:
            score=float(label.get('interaction_strength'))
            gap=float(label.get('route_distance_difference_m'))
            threshold=float(label.get('route_distance_difference_threshold_m'))
            if score < args.min_rift_score or gap > threshold or threshold <=0:
                return None,'weak_rift_interaction'
        except (ValueError,TypeError):
            return None,'invalid_rift_strength'
    if any(not (0.0 <= float(state[3]) < 12.0) for state in states):
        return None,'target_speed_outside_v5_chart'
    if any(not (10.0 <= math.hypot(float(s[0]),float(s[1])) <= 45.0) for s in states):
        return None,'target_outside_rift_candidate_distance_range'
    ego=condition.get('ego_state') or [0.,0.,0.,0.,5.176,2.297]
    if _overlap(states[0],states[1]):
        return None,'target_pair_obb_overlap'
    for state in states:
        if _overlap(state,ego,length_b=float(ego[4]),width_b=float(ego[5])):
            return None,'target_ego_obb_overlap'
        for observed in condition.get('observed_agent_states') or []:
            if _overlap(state,observed,length_b=float(observed[4]),width_b=float(observed[5])):
                return None,'target_observed_obb_overlap'
    candidates=make_candidates(condition)
    if not candidates:
        return None,'no_map_route_candidates'
    assigned=assign_pair(candidates,states,supervision,
                         max_pos_m=args.max_gt_lane_distance,max_heading_rad=0.38,
                         max_anchor_m=args.max_anchor_distance)
    if assigned is None:
        return None,'target_not_representable_on_route_candidate'
    indices=[int(entry[0]) for entry in assigned]
    # Verify the EXACT target is representable by the intrinsic V5 chart.
    # No silent target clamping, truncation or GT state rewriting.
    tester=_oracle_model()
    try:
        truth=tester.target_latents(candidates,states,indices)
        t=candidate_tensors(candidates)
        reconstructed=tester.decode(truth,indices,t).detach().numpy()
    except (ValueError,TypeError,IndexError,RuntimeError) as error:
        return None,'invalid_v5_chart_target'
    gt=[[float(x) for x in state[:4]] for state in states]
    import numpy as np
    g=np.asarray(gt)
    pos_err=float(np.abs(reconstructed[:,:2]-g[:,:2]).max())
    yaw_err=float(np.abs(np.arctan2(np.sin(reconstructed[:,2]-g[:,2]),np.cos(reconstructed[:,2]-g[:,2]))).max())
    speed_err=float(np.abs(reconstructed[:,3]-g[:,3]).max())
    if pos_err>0.03 or yaw_err>0.01 or speed_err>0.02:
        return None,'representation_roundtrip_failed'
    # Retain labels for OFFLINE audit/loss only. They are never passed to the model condition.
    cleaned_condition={
        'ego_state':condition.get('ego_state'),
        'ego_route':condition.get('ego_route'),
        'map':{'lane_polylines':(condition.get('map') or {}).get('lane_polylines') or []},
        'observed_agent_states':condition.get('observed_agent_states') or [],
        'num_generate_agents':2,
    }
    out={
      'schema_version':'hv_v5_joint_raw_condition_v1',
      'town':record.get('town'),
      'scene_id':record.get('scene_id'),
      'episode_key':record.get('episode_key'),
      'condition':cleaned_condition,
      'candidates':candidates,
      'targets':gt,
      'assigned_indices':indices,
      'rift_supervision_for_offline_training_only':[
          {'interaction_strength':float(a['interaction_strength']),
           'route_distance_difference_m':float(a['route_distance_difference_m']),
           'route_distance_difference_threshold_m':float(a['route_distance_difference_threshold_m']),
           'target_route_distance_to_conflict_m':a.get('target_route_distance_to_conflict_m'),
           'ego_route_distance_to_conflict_m':a.get('ego_route_distance_to_conflict_m')}
          for a in supervision],
      'intrinsic_chart_roundtrip':{'max_pos_m':pos_err,'max_heading_rad':yaw_err,'max_speed_mps':speed_err},
      'content_sha256':signature(record),
    }
    return out,'accepted'


def run(args):
    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    report={'schema_version':'hv_v5_dataset_audit_1',
            'contract':{'num_targets':2,'data_source':'V4C RIFT target labels',
                        'online_candidate_source':'ego route + CARLA-correct HD Map only',
                        'min_rift_score':args.min_rift_score,
                        'target_state_rewriting':False},'splits':{}}
    seen_hashes={}
    seen_episodes={}
    for split in ('train','val','test'):
        counts=Counter(); towns=Counter(); topologies=Counter(); candidate_counts=Counter(); score_bins=Counter(); score_values=[]; gap_values=[]
        inpath=Path(getattr(args,'%s_jsonl'%split))
        outpath=outdir/(split+'.jsonl')
        with inpath.open(encoding='utf-8') as source,outpath.open('w',encoding='utf-8') as dest:
            for ix,line in enumerate(source):
                if args.max_records_per_split>0 and ix>=args.max_records_per_split:
                    break
                if not line.strip():continue
                record=json.loads(line)
                counts['input']+=1
                normalized,reason=transform(record,args)
                counts[reason]+=1
                if normalized is None:
                    continue
                h=normalized['content_sha256']
                if h in seen_hashes:
                    counts['duplicate_content']+=1
                    if seen_hashes[h]!=split:
                        raise RuntimeError('Content hash leakage across splits: %s'%h)
                    continue
                episode=normalized.get('episode_key')
                if episode and episode in seen_episodes and seen_episodes[episode]!=split:
                    raise RuntimeError('Episode leakage across splits: %s (%s, %s)'%
                                       (episode,seen_episodes[episode],split))
                seen_hashes[h]=split
                if episode:seen_episodes[episode]=split
                dest.write(json.dumps(normalized,ensure_ascii=False,separators=(',',':'))+'\n')
                counts['written']+=1
                towns[str(normalized['town'])]+=1
                topologies[str((record.get('topology_v3') or {}).get('bucket') or (record.get('topology_v4') or {}).get('bucket') or 'unknown')]+=1
                candidate_counts[len(normalized['candidates'])]+=1
                for sup in normalized['rift_supervision_for_offline_training_only']:
                    score_values.append(sup['interaction_strength'])
                    score_bins['[0.4,0.6)' if sup['interaction_strength']<0.6 else '[0.6,0.8)' if sup['interaction_strength']<0.8 else '[0.8,1.0]']+=1
                    gap_values.append(sup['route_distance_difference_m'])
        report['splits'][split]={
            'input':counts['input'],'written':counts['written'],
            'acceptance_rate':counts['written']/max(counts['input'],1),
            'rejection_counts':dict(sorted(counts.items())),
            'town_counts':dict(sorted(towns.items())),
            'topology_counts':dict(sorted(topologies.items())),
            'interaction_strength_histogram':dict(sorted(score_bins.items())),
            'candidate_count_distribution':dict(sorted(candidate_counts.items())),
            'mean_rift_strength':sum(score_values)/len(score_values) if score_values else None,
            'mean_rift_route_gap_m':sum(gap_values)/len(gap_values) if gap_values else None,
            'output_jsonl':str(outpath),
        }
        print('%s: %d/%d rows written; rejections=%s'%
              (split,counts['written'],counts['input'],dict(counts)),flush=True)
    report['unique_content_hashes']=len(seen_hashes)
    report['unique_episodes']=len(seen_episodes)
    report['ready_to_train']=(report['splits']['train']['written']>=args.min_train_records
                              and report['splits']['val']['written']>=args.min_val_records
                              and report['splits']['test']['written']>=args.min_test_records)
    (outdir/'dataset_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('ready_to_train:',report['ready_to_train'])
    if not report['ready_to_train']:
        raise SystemExit('INSUFFICIENT ELIGIBLE TRAIN/VAL/TEST; analyze rejection histogram; do not train')

if __name__=='__main__':
    run(args_parser().parse_args())
