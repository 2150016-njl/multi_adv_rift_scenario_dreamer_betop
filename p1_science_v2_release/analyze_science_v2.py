#!/usr/bin/env python3
"""Read-only per-record and per-target P1 report. No summary PASS inflation."""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,math
from pathlib import Path

def obj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_new(path,data):
    path=Path(path)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,sort_keys=True,ensure_ascii=False);f.write('\n')
def run(src,out):
    src,out=Path(src),Path(out)
    if out.exists():raise FileExistsError(out)
    ledger=obj(src/'ledger.json');out.mkdir(parents=True)
    rows=[];targets=[];counts=collections.Counter();native_counts=collections.Counter()
    for entry in ledger['records']:
        idx=int(entry['record_index']);town=entry['town']
        file=src/'cases'/('record_%04d'%idx)/'audit'/('record_%04d.json'%idx)
        row={'record_index':idx,'town':town,'execution_status':entry.get('execution_status'),
             'native_subgate_pass':entry.get('native_subgate_pass'),
             'failure_class':entry.get('failure_class'),'result_exists':file.is_file(),
             'native_oracle_parity_fail':0,'native_unknown_numeric':0,
             'native_historical_negative':0,'raw_spawn_accepted':0,'raw_spawn_rejected':0,
             'occupancy_status':'not_run','physical_joint_proven':False}
        counts[entry.get('execution_status')]+=1
        if file.is_file():
            r=obj(file)
            row['occupancy_status']=r['fixed_model3_scene_occupancy'].get('status')
            for n in r.get('native_rows',[]):
                actor=str(n.get('actor_id'));par=n.get('rift_reference_parity_status')
                truth=n.get('native_positive_status')
                c=n.get('candidate_semantics',{})
                native_counts[(par,truth)]+=1
                if par!='pass':row['native_oracle_parity_fail']+=1
                if c.get('candidate_filter_pass') is None or c.get('heading_numeric_status')=='unknown_nonfinite':
                    row['native_unknown_numeric']+=1
                if n.get('native_positive') is False:row['native_historical_negative']+=1
                targets.append({'record_index':idx,'town':town,'actor_id':actor,
                    'reference_parity_status':par,'native_truth_status':truth,
                    'native_positive':n.get('native_positive'),
                    'native_selected_index':n.get('interaction',{}).get('selected_ego_route_index'),
                    'reference_selected_index':n.get('rift_reference_parity',{}).get('selected_ego_route_index'),
                    'heading_numeric_status':c.get('heading_numeric_status'),
                    'raw_legal_training_positive':False})
            for s in r.get('fixed_model3_raw_pose_spawn',[]):
                if s.get('status')=='accepted':row['raw_spawn_accepted']+=1
                if s.get('status')=='rejected':row['raw_spawn_rejected']+=1
        rows.append(row)
    report={'schema':'p1_science_v2_traceable_summary','run':str(src),
        'ledger_sha256':sha(src/'ledger.json'),'observed_records':len(rows),
        'written_records':sum(r['result_exists'] for r in rows),
        'native_subgate_verified':sum(r['native_subgate_pass'] is True for r in rows),
        'execution_counts':dict(counts),'reference_parity_native_statuses':[
           {'parity':str(k[0]),'native_status':str(k[1]),'count':v} for k,v in sorted(native_counts.items())],
        'physics_full_scene_positive_proven':0,
        'scientific_P1_pass':False,'model3_training_ready':False}
    save_new(out/'summary.json',report)
    for name,collection in [('records.csv',rows),('targets.csv',targets)]:
        if collection:
            with (out/name).open('x',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(collection[0].keys()))
                w.writeheader();w.writerows(collection)
    print('P1_REPORT='+str(out/'summary.json'))
    print('P1_WRITTEN='+str(report['written_records'])+'/95')
    print('P1_NATIVE_SUBGATE='+str(report['native_subgate_verified'])+'/95')
    print('P1_NOT_SCIENTIFIC_95_PASS_OR_MODEL3_TRAINING_READY')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',required=True);ap.add_argument('--output-dir',required=True)
    x=ap.parse_args();run(x.run_dir,x.output_dir)
if __name__=='__main__':main()
