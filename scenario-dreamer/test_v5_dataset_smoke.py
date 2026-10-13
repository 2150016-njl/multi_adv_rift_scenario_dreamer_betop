"""Synthesized RIFT-target schema smoke only; these labels are NOT real RIFT evidence."""
import json
import tempfile
import math
from pathlib import Path
import subprocess
import sys

from test_high_value_v5 import assigned_record


def main():
    work=Path(tempfile.mkdtemp(prefix='v5_code_smoke_'))
    r=assigned_record()
    cond=r['condition']
    for split,num in [('train',12),('val',6),('test',6)]:
        with (work/(split+'_source.jsonl')).open('w') as f:
            for i in range(num):
                shift=0.1*(i+(20 if split=='val' else 40 if split=='test' else 0))
                targets=[[10.+shift,0.1,0.,3.],[20.1,-11.+shift,math.pi/2.,2.]]
                sup=[{'interaction_strength':.8,
                       'route_distance_difference_m':1.0,
                       'route_distance_difference_threshold_m':6.0,
                       'conflict_point_local_xy':[20.,0.],
                       'target_route_distance_to_conflict_m':8.,
                       'ego_route_distance_to_conflict_m':9.} for _ in range(2)]
                row={'scene_id':split+'_fixture_'+str(i),'episode_key':split+'_episode_'+str(i),
                     'town':'Town12','condition':cond,
                     'target':{'generated_agent_states':targets,
                               'route_interaction_supervision':sup}}
                f.write(json.dumps(row)+'\n')
    cmd=[sys.executable,'build_high_value_scene_dataset_v5.py','--train-jsonl',str(work/'train_source.jsonl'),
         '--val-jsonl',str(work/'val_source.jsonl'),'--test-jsonl',str(work/'test_source.jsonl'),
         '--output-dir',str(work/'dataset'),'--min-train-records','4',
         '--min-val-records','4','--min-test-records','4',
         '--min-rift-score','0.4']
    print('building dataset',work,flush=True)
    subprocess.run(cmd,check=True)
    cmd=[sys.executable,'train_high_value_scene_generator_v5.py',
         '--data-dir',str(work/'dataset'),'--output-dir',str(work/'model'),
         '--device','cpu','--hidden','24','--epochs','2','--batch-size','4',
         '--s1-val-records','6','--min-joint-s1-rate','0.0']
    print('training 2-epoch smoke',flush=True)
    subprocess.run(cmd,check=True)
    cmd=[sys.executable,'eval_high_value_scene_generator_v5.py',
         '--jsonl',str(work/'dataset'/'test.jsonl'),'--checkpoint',str(work/'model'/'best.ckpt'),
         '--output-json',str(work/'eval.json'),'--max-records','6','--device','cpu']
    subprocess.run(cmd,check=True)
    ev=json.loads((work/'eval.json').read_text())
    assert ev['sampling_calls']==6 and ev['samples_per_condition']==1
    assert ev['oracle_representation_checked']
    print('SMOKE SUCCESS',work,flush=True)

if __name__=='__main__':main()
