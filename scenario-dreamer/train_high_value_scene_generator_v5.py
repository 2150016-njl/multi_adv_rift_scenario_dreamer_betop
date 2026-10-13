#!/usr/bin/env python3
"""Train a new CARLA raw-state generator; no ScenarioDreamer AE/LDM weights.

Consumes RIFT-positive V4C derived V5 records and chooses checkpoints with
ONE stochastic full-sample per held-out condition (not teacher-forced x0).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from eval_high_value_scene_generator_v5 import records,evaluate
from high_value_init.v5_model import CARLAJointRawGenerator


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--lr',type=float,default=3.e-4)
    p.add_argument('--batch-size',type=int,default=8,help='gradient accumulation over examples')
    p.add_argument('--max-train-records',type=int,default=0,help='0 means all; only reduce for smoke')
    p.add_argument('--s1-val-records',type=int,default=128)
    p.add_argument('--patience',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-joint-s1-rate',type=float,default=0.45)
    return p.parse_args()


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(state,path):
    tmp=str(path)+'.pending'
    torch.save(state,tmp)
    Path(tmp).replace(path)


def main():
    a=parse()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('GPU requested but not available')
    data=Path(a.data_dir)
    audit=json.loads((data/'dataset_audit.json').read_text())
    if not audit.get('ready_to_train'):
        raise RuntimeError('Refuse dataset without passing V5 split audit')
    train_path=data/'train.jsonl';val_path=data/'val.jsonl'
    train=list(records(train_path,a.max_train_records))
    val=list(records(val_path,a.s1_val_records))
    if len(train)<4 or len(val)<4:
        raise RuntimeError('Not enough valid train/val records')
    output=Path(a.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Refuse to overwrite nonempty checkpoint directory '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    config=vars(a).copy()
    config.update({'dataset_train_sha256':digest_file(train_path),
                   'dataset_val_sha256':digest_file(val_path),
                   'model_type':CARLAJointRawGenerator.version,
                   'output_format':'two [x,y,yaw_rad,speed,blueprint_L,blueprint_W,1] ego-local',
                   'map_mode':'right_handed','training_oracle_targets_leak_online':False,
                   'ckpt_selector':'heldout joint feasible + at least one static route-sync proxy, S1'})
    (output/'train_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    model=CARLAJointRawGenerator(a.hidden).to(a.device)
    optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1.e-4)
    rng=random.Random(a.seed)
    best=-1.e9; no_gain=0;history=[]
    for epoch in range(a.epochs):
        model.train()
        order=list(range(len(train)));rng.shuffle(order)
        accum=0
        sums={}
        optim.zero_grad(set_to_none=True)
        for k,index in enumerate(order):
            try:
                losses=model.training_losses(train[index])
                if not bool(torch.isfinite(losses['loss'])):
                    raise RuntimeError('nonfinite loss at index '+str(index))
                (losses['loss']/a.batch_size).backward()
            except Exception as error:
                raise RuntimeError('V5 training record %d failed (%s): %s'%(index,train[index].get('scene_id'),error)) from error
            for name,value in losses.items():
                sums[name]=sums.get(name,0.)+float(value.detach().cpu())
            accum+=1
            if accum>=a.batch_size or k==len(order)-1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optim.step();optim.zero_grad(set_to_none=True);accum=0
        with torch.no_grad():
            summary=evaluate(model.eval(),val,seed=a.seed+2026,check_oracle=True)
        s=summary['summary']
        # A feasibility-first checkpoint objective. Both valid + no collision
        # plus *at least one* static route-progress synchronization proxy.
        # This proxy is NOT native RIFT recognition and never a sampling reranker.
        joint=s['joint_plus_route_proxy']
        selector=joint+0.25*s['both_joint_feasible']-0.005*s['static_route_gap_m']
        row={'epoch':epoch,'s1_num_records':len(val),'selector':selector,
             'train':{k:val/len(train) for k,val in sums.items()},
             'val_s1':s}
        history.append(row)
        (output/'history.json').write_text(json.dumps(history,indent=2,ensure_ascii=False))
        payload={'schema_version':'hv_v5_raw_checkpoint_v1','model':model.state_dict(),
                 'config':config,'epoch':epoch,'selection_score':selector,
                 'validation_s1':s}
        atomic_save(payload,output/'last.ckpt')
        if selector>best+1.e-6:
            best=selector;no_gain=0
            atomic_save(payload,output/'best.ckpt')
        else:
            no_gain+=1
        print(json.dumps({'epoch':epoch,'best_score':best,'selector':selector,
                          'joint_s1':s['both_joint_feasible'],
                          'joint_route_proxy_s1':joint,
                          'route_gap_m':s['static_route_gap_m'],
                          'train_loss':row['train']['loss']},ensure_ascii=False),flush=True)
        if no_gain>=a.patience:
            break
    final=torch.load(output/'best.ckpt',map_location='cpu',weights_only=False)
    s=final['validation_s1']
    readiness={'best_epoch':final['epoch'],'best_s1':s,
               'ready_for_limited_CARLA_test':bool(s['both_joint_feasible']>=a.min_joint_s1_rate
                                            and s['joint_plus_route_proxy']>=.25)}
    (output/'training_readiness.json').write_text(json.dumps(readiness,indent=2))
    print('TRAINING_READINESS',json.dumps(readiness,indent=2))
    if not readiness['ready_for_limited_CARLA_test']:
        print('GATE FAILED: do not start CARLA; fix model/data with saved diagnostics')

if __name__=='__main__':main()
