#!/usr/bin/env python3
"""Run matched GT/V2/V3 diagnostics on a dedicated CARLA world and summarize."""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import shutil


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stage_signature(command):
    """Compare experiment inputs/settings, independent of output and transport."""
    result=[];i=2
    while i<len(command):
        token=command[i]
        if token in ('--output-dir','--resume-from','--host','--port','--timeout'):
            i+=2;continue
        if token=='--manifest':
            result.extend([token,sha(Path(command[i+1]))]);i+=2;continue
        result.append(token);i+=1
    return result


def selected_samples(records, indices):
    found={};wanted=set(indices)
    for record in records:
        index=record.get('record_idx',record.get('record_index'))
        if index not in wanted:continue
        if record.get('sample_idx',record.get('sample_index'))!=0 or index in found:
            raise ValueError('Requires exactly one raw S1 output for index %s'%index)
        found[index]=record
    if set(found)!=wanted:raise ValueError('Missing raw S1 indices: '+str(sorted(wanted-set(found))))
    return found


def rebind_generated(base_manifest,dataset,generated,output):
    """Reuse verified source identity; validate every V3 pair without rescanning RAW."""
    from build_hv_snapshot_audit_manifest import pair_job
    from hv_audit_inputs import manifest_jobs
    base=manifest_jobs(base_manifest)
    evidence=json.loads(Path(str(base_manifest)+'.audit.json').read_text(encoding='utf-8'))
    if sha(dataset)!=evidence['dataset_sha256']:raise ValueError('Dataset changed since source pairing')
    indices=[j['record_index'] for j in base];wanted=set(indices);records={}
    with dataset.open(encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            if index in wanted:records[index]=json.loads(line)
            if index>=max(wanted):break
    if set(records)!=wanted:raise ValueError('Dataset is missing selected indices')
    samples=selected_samples(read_jsonl(generated),indices)
    jobs=[]
    for previous in base:
        index=previous['record_index'];path=Path(previous['snapshot_path']);data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=previous['snapshot_sha256']:
            raise ValueError('Source snapshot changed for index %s'%index)
        if samples[index].get('town') not in (None,records[index].get('town')):
            raise ValueError('Generated town differs for index %s'%index)
        job=pair_job(path,data,records[index],samples[index]);job['record_index']=index
        for key in ('snapshot_sha256','target_actor_ids','context_actor_ids',
                    'reference_target_states','candidate_condition'):
            if job[key]!=previous[key]:raise ValueError('Matched comparison differs at %s, index %s'%(key,index))
        jobs.append(job)
    if output.exists() or Path(str(output)+'.audit.json').exists():raise FileExistsError(output)
    output.write_text(''.join(json.dumps(j,ensure_ascii=False,allow_nan=False)+'\n' for j in jobs),encoding='utf-8')
    scope=dict(evidence,manifest_sha256=sha(output),generated_sha256=sha(generated),
               source_manifest=str(base_manifest),source_manifest_sha256=sha(base_manifest),
               selection='same_verified_indices_as_v2',selected_dataset_indices=indices)
    Path(str(output)+'.audit.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
    return {'jobs':len(jobs),'source_scan':'only bound snapshots; no RAW directory scan'}


def raw_source_manifest(base_manifest,output):
    from hv_audit_inputs import manifest_jobs
    groups={}
    for job in manifest_jobs(base_manifest):
        key=job.get('snapshot_replay_sha256') or job['snapshot_sha256']
        groups.setdefault(key,[]).append(job)
    jobs=[];aliases={}
    for key,group in groups.items():
        source=group[0]
        jobs.append({k:source[k] for k in ('snapshot_path','snapshot_sha256','record_index','episode_key','town') if k in source})
        aliases[key]=[j['record_index'] for j in group]
    if output.exists() or Path(str(output)+'.audit.json').exists():raise FileExistsError(output)
    output.write_text(''.join(json.dumps(j)+'\n' for j in jobs),encoding='utf-8')
    scope={'passed':True,'manifest_sha256':sha(output),'requested_records':len(jobs),'written_jobs':len(jobs),
           'evaluation_scope':'deduplicated_raw_sources_of_verified_pairs','full_evaluation_eligible':False,
           'source_manifest_sha256':sha(base_manifest),'source_record_groups':aliases}
    Path(str(output)+'.audit.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
    return len(jobs)


def truth(value):
    return value if isinstance(value,bool) else None


def combine(values,any_true=False):
    if not values:return None
    if any_true:
        return True if True in values else (None if None in values else False)
    return False if False in values else (None if None in values else True)


def actor_gates(actor,row,cohort):
    native=actor.get('native') or {};road=actor.get('road_geometry') or {}
    spawn=(row.get('spawn') or {}).get('raw_fleet' if cohort=='raw_pool' else cohort) or {}
    aid=actor['actor_id']
    attempt=next((a for a in spawn.get('attempts',[]) if a.get('source_actor_id')==aid and a.get('role')=='target'),None)
    geometry=spawn.get('geometry_3d') or {}
    geometry=geometry.get(str(aid),geometry.get(aid)) or {}
    raw_spawn=(truth(attempt.get('accepted')) if attempt is not None else None)
    if spawn.get('context_complete') is not True:raw_spawn=None
    return {
        'forward':truth(actor.get('forward_fov')),
        'center_driving':truth(road.get('center_driving')),
        'preview_driving':truth(road.get('preview_driving')),
        'native':truth(native.get('native_candidate')) if native.get('evaluated') is True else None,
        'candidate_coverage':truth(actor.get('candidate_geometry_supported')),
        'v6_approach_coverage':truth(actor.get('v6_approach_geometry_supported')),
        'occupancy_2d_clear':truth((actor.get('occupancy_2d') or {}).get('clear')),
        'raw_spawn':raw_spawn,
        'whole_polygon':truth((geometry.get('whole_road') or {}).get('whole_box_in_sampled_driving_polygon')),
        'peer_3d_clear':not geometry['peer_3d_overlap_ids'] if 'peer_3d_overlap_ids' in geometry else None,
    }


def cohort_diagnosis(rows,cohort,mode):
    counts=Counter();failures=Counter();events=[]
    native_order=('forward','preview_driving','native')
    joint_order=('forward','preview_driving','native','raw_spawn','whole_polygon','peer_3d_clear')
    order=joint_order if mode in ('spawn','physics') else native_order
    for row in rows:
        actors=row.get(cohort,[]);native_events=[];joint_events=[]
        for actor in actors:
            gates=actor_gates(actor,row,cohort);counts['actors']+=1
            for name,value in gates.items():counts[name+'_'+str(value).lower()]+=1
            if gates['forward'] is True and gates['native'] is True:
                counts['native_forward_positive']+=1
                counts['positive_candidate_'+str(gates['candidate_coverage']).lower()]+=1
                counts['positive_v6_approach_'+str(gates['v6_approach_coverage']).lower()]+=1
            native_events.append(combine([gates['forward'],gates['native']]))
            joint_events.append(combine([gates[k] for k in joint_order]))
            label='all_gates_true'
            for name in order:
                if gates[name] is not True:
                    label=name+('_unknown' if gates[name] is None else '_false');break
            failures[label]+=1
        event={'record_index':row.get('record_index'),
               'block':row.get('episode_key') or '%s:%s:%s:%s'%(row.get('town'),row.get('route_id'),
                    row.get('route_index'),row.get('repetition_index')),
               'snapshot_sha256':row.get('source_sha256'),
               'map_sha256':row.get('carla_opendrive_sha256'),
               'context_sha256':row.get('candidate_condition_sha256'),
               'at_least_one_front_native':combine(native_events,True),
               'at_least_one_joint':combine(joint_events,True),
               'both_joint':combine(joint_events) if len(joint_events)==2 else None}
        events.append(event)
    scene_counts={}
    for metric in ('at_least_one_front_native','at_least_one_joint','both_joint'):
        values=[e[metric] for e in events];hist=Counter(str(v).lower() for v in values)
        scene_counts[metric]={'planned_scenes':len(rows),'true':hist['true'],'false':hist['false'],
            'unknown':hist['none'],'confirmed_true_fraction_of_planned':hist['true']/len(rows) if rows else None,
            'complete_rate':hist['true']/len(rows) if rows and not hist['none'] else None}
    return {'actor_counts':dict(counts),'first_failure_or_unknown':dict(failures),
            'scene_metrics':scene_counts,'events':events}


def paired_comparison(left,right,metric,replicates=1000):
    for events in (left,right):
        indices=[e['record_index'] for e in events]
        if None in indices or len(indices)!=len(set(indices)):
            raise ValueError('Paired comparison requires unique original record indices')
    a={e['record_index']:e for e in left};b={e['record_index']:e for e in right}
    indices=sorted(set(a)|set(b));blocks=defaultdict(list);discordant=Counter();unknown=0
    for index in indices:
        x=a.get(index);y=b.get(index)
        if x is None or y is None:unknown+=1;continue
        if (x['snapshot_sha256'],x['context_sha256'])!=(y['snapshot_sha256'],y['context_sha256']):
            raise ValueError('V2/V3 source or condition mismatch at index %s'%index)
        if x.get('map_sha256') and y.get('map_sha256') and x['map_sha256']!=y['map_sha256']:
            raise ValueError('V2/V3 OpenDRIVE map mismatch at index %s'%index)
        if x[metric] is None or y[metric] is None:unknown+=1;continue
        difference=int(y[metric])-int(x[metric]);blocks[x['block']].append(difference)
        discordant['both_true' if x[metric] and y[metric] else 'v2_only' if x[metric]
                   else 'v3_only' if y[metric] else 'both_false']+=1
    values=[v for group in blocks.values() for v in group];ci=None
    if len(blocks)>=2:
        rng=random.Random(0);keys=sorted(blocks);samples=[]
        for _ in range(replicates):
            selected=[v for key in rng.choices(keys,k=len(keys)) for v in blocks[key]]
            samples.append(sum(selected)/len(selected))
        samples.sort();ci=[samples[int(.025*replicates)],samples[int(.975*replicates)]]
    return {'metric':metric,'paired_planned_records':len(indices),'known_both_records':len(values),
            'unknown_or_missing_records':unknown,'episode_blocks':len(blocks),'outcomes':dict(discordant),
            'v3_minus_v2_on_known_pairs':sum(values)/len(values) if values else None,
            'episode_block_bootstrap_95_interval':ci,
            'scope':'diagnostic known pairs; sparse copies/adjacent frames are not independent; no full-test claim'}


def analyze(experiment):
    stages={};warnings=[]
    for folder in sorted(experiment.iterdir()):
        if not folder.is_dir() or not (folder/'rows.jsonl').is_file():continue
        rows=read_jsonl(folder/'rows.jsonl')
        jobs=read_jsonl(folder/'jobs.jsonl') if (folder/'jobs.jsonl').is_file() else []
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8')) if (folder/'summary.json').is_file() else {}
        planned=max(len(jobs),summary.get('planned_jobs',0),len(rows))
        if len(rows)<planned:
            warnings.append(folder.name+': incomplete row ledger')
            rows.extend({'errors':['Unwritten planned record']} for _ in range(planned-len(rows)))
        for i,row in enumerate(rows):
            if i<len(jobs):
                row.setdefault('record_index',jobs[i].get('record_index'))
                row.setdefault('episode_key',jobs[i].get('episode_key'))
        mode=summary.get('mode') or folder.name.split('_')[1].rstrip('0123456789')
        stages[folder.name]={'summary':summary,'planned_records':planned,
            'records_with_errors':sum(bool(r.get('errors')) for r in rows),
            'job_error_counts':dict(Counter(error for r in rows for error in r.get('errors',[]))),
            'nonfinite_fields':dict(Counter(item['json_pointer'] for r in rows
                 for item in (r.get('numeric_diagnostics') or {}).get('nonfinite_values',[]))),
            'cohorts':{name:cohort_diagnosis(rows,name,mode) for name in ('raw_pool','selected_gt','generated')}}
    comparisons=[]
    for mode in ('native','spawn','physics'):
        lhs=stages.get('v2_'+mode);rhs=stages.get('v3_'+mode)
        if lhs and rhs:
            metric='at_least_one_joint' if mode in ('spawn','physics') else 'at_least_one_front_native'
            comparisons.append(dict(stage=mode,**paired_comparison(lhs['cohorts']['generated']['events'],
                                                                  rhs['cohorts']['generated']['events'],metric)))
            comparisons.append(dict(stage=mode,cohort='GT_repeatability',**paired_comparison(
                lhs['cohorts']['selected_gt']['events'],rhs['cohorts']['selected_gt']['events'],metric)))
    pair_reports={p.name:json.loads(p.read_text(encoding='utf-8')) for p in experiment.glob('*.audit.json')}
    context_comparisons=[]
    for model in ('v2','v3'):
        full=stages.get(model+'_spawn');sparse=stages.get(model+'_spawn_sparse')
        if full and sparse:
            context_comparisons.append({'model':model,'full_snapshot':full['cohorts']['generated']['scene_metrics'],
                'manifest_context':sparse['cohorts']['generated']['scene_metrics'],
                'meaning':'Same output/model input; physical non-target traffic differs. Manifest-context success is not online full-traffic success.'})
    physics_stages=[name for name in stages if name.endswith('_physics')]
    return {'full_evaluation_eligible':False,'evaluation_scope':'diagnostic subset, not final acceptance',
            'physics_evidence':{'stages_present':physics_stages,
                'status':'inspect_stage_results' if physics_stages else 'not_run',
                'meaning':'No physics evidence means speed execution, settling and dynamic collision remain unverified.'},
            'stages':stages,'v2_v3_paired_comparisons':comparisons,'pairing_scope':pair_reports,'warnings':warnings,
            'physical_context_ablation':context_comparisons,
            'limits':['No attention ranking or final green activation tested.',
                      'Sampled road polygon and finite uncontrolled physics probe are not exact mesh or episode replay.',
                      'Candidate coverage is diagnostic, not a requirement that a valid generated actor match GT lane.',
                      'Zero positives in a saved fleet does not prove no feasible solution exists.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-report',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=31100)
    p.add_argument('--timeout',type=float,default=300.)
    p.add_argument('--pool-records',type=int,default=100)
    p.add_argument('--native-records',type=int,default=100)
    p.add_argument('--probe-ticks',type=int,default=20)
    p.add_argument('--with-physics',action='store_true',help='Opt into dynamic tick probes; default runs snapshot/native/zero-tick spawn only')
    p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--resume-from',type=Path,help='Old experiment directory; preserve it and reuse verified completed stages/prefixes')
    a=p.parse_args();out=a.output_dir
    if a.analyze_only:
        report=analyze(out);path=out/('analysis_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');print('ANALYSIS='+str(path));return
    if not a.pair_report:p.error('--pair-report is required for experiments')
    if not 0<a.native_records<=a.pool_records or a.probe_ticks<=0 or a.timeout<=0:
        p.error('Require 0 < native-records <= pool-records, probe-ticks/timeout > 0')
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    state={'status':'running','stages':{},'port':a.port,'output_dir':str(out),
           'physics_requested':a.with_physics,
           'excluded_stages':[] if a.with_physics else ['pool_physics','v2_physics','v3_physics']}
    previous_status=json.loads((a.resume_from/'batch_status.json').read_text(encoding='utf-8')) if a.resume_from else {}
    state['resumed_from']=str(a.resume_from) if a.resume_from else None
    def save():
        (out/'batch_status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    def stage(name,script,args):
        command=[sys.executable,'-u',str(ROOT/script)]+[str(v) for v in args]
        old=a.resume_from/name if a.resume_from else None
        old_state=(previous_status.get('stages') or {}).get(name) or {}
        if old is not None and (old/'jobs.jsonl').is_file() and (old/'rows.jsonl').is_file():
            if stage_signature(old_state.get('command',[]))!=stage_signature(command):
                raise ValueError('Resume stage arguments/input SHA differ: '+name)
            if old_state.get('status')=='completed' and (old/'summary.json').is_file():
                summary=json.loads((old/'summary.json').read_text(encoding='utf-8'))
                if summary.get('config_sha256') and sha(summary['recognition_config'])!=summary['config_sha256']:
                    raise ValueError('Recognition config changed since completed stage: '+name)
                if len(read_jsonl(old/'rows.jsonl'))!=len(read_jsonl(old/'jobs.jsonl')):
                    raise ValueError('Completed stage has an incomplete ledger: '+name)
                target=out/name;target.mkdir(exist_ok=False)
                for filename in ('jobs.jsonl','rows.jsonl','summary.json'):
                    shutil.copy2(old/filename,target/filename)
                old_log=a.resume_from/'logs'/(name+'.log')
                if old_log.is_file():shutil.copy2(old_log,out/'logs'/(name+'.log'))
                state['stages'][name]={'status':'completed','command':command,'reused_from':str(old),
                                       'reused_rows':summary.get('planned_jobs')};save()
                print('REUSED '+name+' from '+str(old),flush=True)
                return True
            command+=['--resume-from',str(old)]
        state['stages'][name]={'status':'running','command':command};save()
        print('STAGE '+name,flush=True)
        with (out/'logs'/(name+'.log')).open('w',encoding='utf-8') as log:
            log.write(json.dumps(command)+'\n');log.flush()
            process=subprocess.Popen(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding='utf-8',errors='replace')
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=15.)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5.)
                state['stages'][name]['status']='interrupted';save()
                raise
        state['stages'][name].update(status='completed' if code==0 else 'failed',exit_code=code);save()
        print('STAGE %s exit=%d; log=%s'%(name,code,out/'logs'/(name+'.log')),flush=True)
        return code==0
    save();print('MAINLINE_RESULT_DIR='+str(out),flush=True)
    dataset=ROOT/'data/hv_init/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full/test.jsonl'
    v3=ROOT/'data/hv_init/eval/v2_v3_same_v3test_full_20261006/v3_raw_s1_generated.jsonl'
    config=ROOT/'RIFT/rift/cbv/recognition/config/rule.yaml'
    raw=ROOT/'data/hv_init/rift_labeled_phase7_v2_uncovered_routes_seed0_7'
    manifest=out/'v2_pairs.jsonl'
    audit='audit_hv_snapshot_experiment.py'
    common=['--host',a.host,'--port',a.port,'--timeout',a.timeout,'--load-maps',
            '--recognition-config',config,'--recognition-mode','eval','--sampling-resolution','1.0','--enrich-map',
            '--group-by-town']
    try:
        paired=stage('verified_subset','prepare_hv_diagnostic_subset.py',
                     ['--pair-report',a.pair_report,'--output-manifest',manifest])
        if paired:
            raw_source_manifest(manifest,out/'raw_sources.jsonl')
            pool_input=['--manifest',out/'raw_sources.jsonl']
        else:
            pool_input=['--input-dir',raw]
        stage('pool_snapshot',audit,pool_input+['--max-records',a.pool_records,
                                    '--mode','snapshot','--output-dir',out/'pool_snapshot'])
        stage('pool_native',audit,pool_input+['--max-records',a.pool_records,
                                  '--mode','native','--output-dir',out/'pool_native']+common)
        for mode in (('spawn','physics') if a.with_physics else ('spawn',)):
            stage('pool_'+mode,audit,pool_input+['--max-records',a.pool_records,'--mode',mode,
                  '--probe-ticks',a.probe_ticks,'--output-dir',out/('pool_'+mode)]+common)
        manifests={'v2':manifest} if paired else {}
        if paired:
            name='v3_same_sources';state['stages'][name]={'status':'running'};save()
            try:
                result=rebind_generated(manifest,dataset,v3,out/'v3_pairs.jsonl')
                manifests['v3']=out/'v3_pairs.jsonl'
                state['stages'][name].update(status='completed',result=result)
            except Exception as exc:
                state['stages'][name].update(status='failed',error='%s: %s'%(type(exc).__name__,exc))
            save()
        for model,path in manifests.items():
            stage(model+'_snapshot',audit,['--manifest',path,'--max-records',a.native_records,
                  '--mode','snapshot','--physical-context','full_snapshot','--output-dir',out/(model+'_snapshot')])
            for mode in (('native','spawn','physics') if a.with_physics else ('native','spawn')):
                stage(model+'_'+mode,audit,['--manifest',path,'--max-records',a.native_records,
                      '--mode',mode,'--physical-context','full_snapshot',
                      '--probe-ticks',a.probe_ticks,'--output-dir',out/(model+'_'+mode)]+common)
            for mode in ('snapshot','spawn'):
                stage(model+'_'+mode+'_sparse',audit,['--manifest',path,'--max-records',a.native_records,
                      '--mode',mode,'--physical-context','manifest',
                      '--output-dir',out/(model+'_'+mode+'_sparse')]+(common if mode=='spawn' else []))
    except BaseException as exc:
        state['error']='%s: %s'%(type(exc).__name__,exc)
        raise
    finally:
        state['status']='completed' if all(v['status']=='completed' for v in state['stages'].values()) else 'incomplete'
        try:
            report=analyze(out)
            (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        except Exception as exc:
            state['analysis_error']='%s: %s'%(type(exc).__name__,exc);state['status']='incomplete'
        save();print('MAINLINE RESULTS='+str(out),flush=True)
    if state['status']!='completed':raise SystemExit('Some stages failed; analysis and stage logs were preserved.')


if __name__=='__main__':main()
