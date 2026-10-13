#!/usr/bin/env python3
"""Fail-fast V5 data/representation/train-inference condition parity audit.

No GPU, no CARLA; includes no target information in online model conditions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from eval_high_value_scene_generator_v5 import records,oracle
from high_value_init.v5_candidates import make_candidates
from high_value_init.v5_model import CARLAJointRawGenerator


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--max-per-split',type=int,default=0,help='0=all; complete audit recommended')
    ap.add_argument('--output-json',required=True)
    ap.add_argument('--diagnostic-unready',action='store_true',help='Audit already-accepted rows even when builder readiness failed; write report but NEVER pass training gate')
    a=ap.parse_args()
    torch.set_num_threads(2)
    root=Path(a.data_dir)
    ref=json.loads((root/'dataset_audit.json').read_text())
    if not ref.get('ready_to_train') and not a.diagnostic_unready:
        raise SystemExit('Builder readiness failed (use --diagnostic-unready for observation only)')
    model=CARLAJointRawGenerator(hidden=16)
    report={'schema_version':'hv_v5_representation_gate_v1','splits':{},
            'builder_ready_to_train':bool(ref.get('ready_to_train')),
            'diagnostic_only':not bool(ref.get('ready_to_train')),
            'online_input_leakage_count':0,'condition_candidate_parity_failures':0,
            'representation_failures':0}
    seen=set()
    for split in ('train','val','test'):
        stats=Counter();episodes=set()
        for row in records(root/(split+'.jsonl'),a.max_per_split):
            stats['rows']+=1
            condition=row['condition']
            # The only extra target/RIFT supervision fields are at ROW root;
            # model.distribution receives condition and candidates exclusively.
            cond_string=json.dumps(condition,sort_keys=True)
            if any(key in cond_string for key in ('rift_labels','interaction_waypoint',
                    'target_route_to_interaction','route_interaction_supervision','actor_id')):
                report['online_input_leakage_count']+=1
            if row.get('schema_version')!='hv_v5_joint_raw_condition_v1' or len(row['targets'])!=2:
                raise ValueError('Invalid V5 schema in split %s'%split)
            calculated=make_candidates(condition)
            if calculated != row['candidates']:
                report['condition_candidate_parity_failures']+=1
            with torch.no_grad(): rec=oracle(model,row)
            for key,value in rec.items():
                stats[key+'_sum']+=value
                stats[key+'_max']=max(stats[key+'_max'],value)
            if (rec['position_max_m']>.03 or rec['heading_max_rad']>.01
                    or rec['speed_max_mps']>.02):
                report['representation_failures']+=1
            h=row['content_sha256']
            if h in seen:raise RuntimeError('Repeated canonical content across V5 records: '+h)
            seen.add(h)
            e=row.get('episode_key')
            if e and e in episodes:stats['same_episode_records']+=1
            episodes.add(e)
        if stats['rows']==0:
            raise RuntimeError('Empty split '+split)
        report['splits'][split]={
          'checked':stats['rows'],
          'position_max_error_m':stats['position_max_m_max'],
          'heading_max_error_rad':stats['heading_max_rad_max'],
          'speed_max_error_mps':stats['speed_max_mps_max'],
          'mean_position_max_error_m':stats['position_max_m_sum']/stats['rows'],
          'same_episode_records':stats['same_episode_records'],
        }
    report['representation_passed']=(report['online_input_leakage_count']==0 and
                                    report['condition_candidate_parity_failures']==0 and
                                    report['representation_failures']==0)
    report['passed']=bool(ref.get('ready_to_train')) and report['representation_passed']
    output=Path(a.output_json)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2))
    if not report['passed']:
        raise SystemExit('V5 gate NOT ready: report saved for observation only; refuse training')

if __name__=='__main__':main()
