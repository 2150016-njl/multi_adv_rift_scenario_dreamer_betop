#!/usr/bin/env python3
"""V5 representation oracle and ONE-SAMPLE full-generation feasibility evaluation.
No CARLA/RIFT here: all route interaction geometry is explicitly a proxy.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from high_value_init.v5_model import (CARLAJointRawGenerator, candidate_tensors,
                                       SIZE_FIXED, _angle)
from high_value_init.v5_candidates import closest


def records(path,max_records=0):
    with open(path,encoding='utf-8') as f:
        for i,line in enumerate(f):
            if max_records>0 and i>=max_records:break
            if line.strip():yield json.loads(line)


def oracle(model,row):
    indices=row['assigned_indices']
    states=row['targets']
    with torch.no_grad():
        z=model.target_latents(row['candidates'],states,indices)
        tensors=candidate_tensors(row['candidates'],device=next(model.parameters()).device)
        rec=model.decode(z,indices,tensors).cpu().numpy()
    gt=np.asarray(states,dtype=np.float64)
    dpos=np.abs(rec[:,:2]-gt[:,:2])
    dyaw=np.abs(np.arctan2(np.sin(rec[:,2]-gt[:,2]),np.cos(rec[:,2]-gt[:,2])))
    dspeed=np.abs(rec[:,3]-gt[:,3])
    return {'position_max_m':float(dpos.max()),'heading_max_rad':float(dyaw.max()),
            'speed_max_mps':float(dspeed.max())}


def _overlap(a,b,length_a=SIZE_FIXED[0],width_a=SIZE_FIXED[1],
             length_b=SIZE_FIXED[0],width_b=SIZE_FIXED[1]):
    # Oriented bounding box separating axis theorem; use fixed CARLA blueprint
    # approximation only as OFFLINE metric, never for inference rejection.
    ayaw=float(a[2]);byaw=float(b[2]);
    au=np.asarray([math.cos(ayaw),math.sin(ayaw)])
    av=np.asarray([-au[1],au[0]])
    bu=np.asarray([math.cos(byaw),math.sin(byaw)])
    bv=np.asarray([-bu[1],bu[0]])
    d=np.asarray(b[:2],dtype=float)-np.asarray(a[:2],dtype=float)
    for axis in (au,av,bu,bv):
        ra=length_a/2*abs(axis@au)+width_a/2*abs(axis@av)
        rb=length_b/2*abs(axis@bu)+width_b/2*abs(axis@bv)
        if abs(d@axis)>ra+rb:return False
    return True


def raw_metrics(row,sampled):
    states=sampled['generated_agent_states']; candidates=row['candidates']; ids=sampled['candidate_indices'];
    condition=row['condition']
    lane_dist=[];heading_errors=[];sync=[]
    for s,i in zip(states,ids):
        c=candidates[i]
        d,progress,angle,_=closest(np.asarray(c['polyline']),s[:2])
        lane_dist.append(d)
        heading_errors.append(abs(math.atan2(math.sin(float(s[2])-angle),math.cos(float(s[2])-angle))))
        sync.append(abs((c['anchor_s']-progress)-c['ego_route_s']))
    overlap=_overlap(states[0],states[1])
    ego=condition.get('ego_state') or [0.,0.,0.,0.,5.176,2.297]
    for s in states:
        overlap=overlap or _overlap(s,ego,length_b=float(ego[4]),width_b=float(ego[5]))
    for observed in condition.get('observed_agent_states') or []:
        for s in states:
            overlap=overlap or _overlap(s,observed,length_b=float(observed[4]),width_b=float(observed[5]))
    radius=[math.hypot(float(s[0]),float(s[1])) for s in states]
    actor_valid=[bool(np.isfinite(s[:4]).all() if isinstance(s,np.ndarray) else np.isfinite(s[:4]).all())
                 and 10.0<=radius[k]<=45.0 and lane_dist[k]<=1.75
                 and heading_errors[k]<=0.45 and 0<=float(s[3])<=12.0
                 and abs(float(s[0]))<=32 and abs(float(s[1]))<=32 for k,s in enumerate(states)]
    both_road_valid=all(actor_valid)
    any_static_interaction=any(v<5.0 for v in sync)
    return {
      'both_road_heading_fov_distance_valid':float(both_road_valid),
      'no_obb_overlap':float(not overlap),
      'both_joint_feasible':float(both_road_valid and not overlap),
      'at_least_one_static_route_sync_proxy':float(any_static_interaction),
      'joint_plus_route_proxy':float(both_road_valid and not overlap and any_static_interaction),
      'nearest_lane_distance_m':float(np.mean(lane_dist)),
      'heading_to_chosen_lane_rad':float(np.mean(heading_errors)),
      'static_route_gap_m':float(np.mean(sync)),
      'pair_separation_m':math.dist(states[0][:2],states[1][:2]),
      'speed_mps':float(np.mean([a[3] for a in states])),
      'coordinate_frame':'ego_local_rh_x_forward',
      'native_rift_conflict_checked':False,
    }


def evaluate(model,rows,seed=1,stochastic=True,check_oracle=True):
    samples=[]
    for i,row in enumerate(rows):
        if check_oracle:
            o=oracle(model,row)
            if o['position_max_m']>.03 or o['heading_max_rad']>.01 or o['speed_max_mps']>.02:
                raise RuntimeError('Representation oracle failed at row %d: %s'%(i,o))
        # exactly ONE model sampling per condition, no best-of-N or resampling.
        torch.manual_seed((seed+1000003*i)%(2**32))
        with torch.no_grad():
            sampled=model.sample_once(row['condition'],row['candidates'])
        metrics=raw_metrics(row,sampled)
        samples.append({'scene_id':row.get('scene_id'),'episode_key':row.get('episode_key'),
                        'town':row.get('town'),'metrics':metrics,
                        'raw_generated_states':sampled['generated_agent_states'],
                        'sampled_candidate_indices':sampled['candidate_indices']})
    keys=[k for k,v in samples[0]['metrics'].items() if isinstance(v,(int,float))] if samples else []
    summary={k:float(np.mean([s['metrics'][k] for s in samples])) for k in keys}
    return {'schema_version':'hv_v5_single_sample_eval_v1','num_conditions':len(samples),
            'sampling_calls':len(samples),'samples_per_condition':1,
            'raw_output_rewriting':False,'does_not_claim_native_rift_recognition':True,
            'fixed_model_blueprint':'vehicle.tesla.model3',
            'oracle_representation_checked':bool(check_oracle),
            'summary':summary,'samples':samples}


def load_model(checkpoint,device='cpu'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('schema_version')!='hv_v5_raw_checkpoint_v1':
        raise ValueError('Refuse non-V5 checkpoint: '+str(checkpoint))
    model=CARLAJointRawGenerator(hidden=int(ck['config']['hidden']))
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--jsonl',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-json',required=True)
    parser.add_argument('--max-records',type=int,default=256)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(),4))
    model=load_model(args.checkpoint,args.device)
    result=evaluate(model,list(records(args.jsonl,args.max_records)),args.seed)
    path=Path(args.output_json)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
