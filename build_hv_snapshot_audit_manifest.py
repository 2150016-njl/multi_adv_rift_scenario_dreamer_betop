#!/usr/bin/env python3
"""Bind saved S1 records to original snapshots by state/route checks, not ID alone."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np

from audit_hv_snapshot_experiment import local_state, condition_from_snapshot, sha
from high_value_init.v5_candidates import _array
from hv_audit_inputs import snapshot_paths


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def select_records(path,max_records,selection='prefix',seed=0):
    if selection=='prefix' or max_records==0:
        result=[]
        for i,r in enumerate(jsonl(path)):
            if max_records>0 and len(result)>=max_records:break
            result.append((i,r))
        return result
    unique={}
    for i,r in enumerate(jsonl(path)):
        key=(r.get('town'),r.get('episode_key'),base_scene_id(r))
        priority=((r.get('context_sampling') or {}).get('sparse_context_copy_index') is not None,i)
        if key not in unique or priority<unique[key][0]:unique[key]=(priority,i,r)
    buckets=defaultdict(list)
    for _,i,r in unique.values():
        count=(r.get('rift_audit') or {}).get('observed_agent_count_before_context_sampling',
                                          len(r['condition'].get('observed_agent_states',[])))
        initial=base_scene_id(r).endswith('_frame0')
        buckets[(str(r.get('town')),initial,'sparse' if count<=4 else 'dense')].append((i,r))
    rng=random.Random(seed)
    for values in buckets.values():rng.shuffle(values)
    keys=sorted(buckets);rng.shuffle(keys);chosen=[]
    while keys and len(chosen)<max_records:
        for key in list(keys):
            if not buckets[key]:keys.remove(key);continue
            chosen.append(buckets[key].pop())
            if len(chosen)>=max_records:break
    return chosen  # Keep round-robin order so native prefixes retain strata.


def matches(a,b):
    return (len(b)>=6 and math.dist(a[:2],b[:2])<=.03 and
            abs(math.atan2(math.sin(a[2]-b[2]),math.cos(a[2]-b[2])))<=.01 and
            abs(a[3]-b[3])<=.02 and max(abs(a[k]-b[k]) for k in (4,5))<=.05)


def base_scene_id(record):
    scene_id=record['scene_id']
    index=(record.get('context_sampling') or {}).get('sparse_context_copy_index')
    suffix=':sparse%s' % index
    if index is not None and scene_id.endswith(suffix):
        return scene_id[:-len(suffix)]
    return scene_id


def replay_inputs(scene):
    """Exact replay inputs + saved labels; ignore IDs and unrelated diagnostics.

    Equivalent aliases are not proof of the historical source path. Keep every
    source SHA in the manifest, and only collapse exact physical/label equality.
    """
    keys=('type_id','x','y','z','yaw_deg','pitch_deg','roll_deg','vx','vy','vz',
          'speed_mps','length','width','height','bounding_box')
    def pose(p):return {k:p[k] for k in keys if k in p}
    actors=scene['agents'];labels={l['actor_id']:l for l in scene.get('rift_labels',[])}
    rows=[]
    for a in actors:
        label=dict(labels.get(a['actor_id'],{}))
        label.pop('actor_id',None);label.pop('local_id',None)
        rows.append({'pose':pose(a),'label':label})
    rows.sort(key=lambda r:json.dumps(r,sort_keys=True))
    debug=((scene.get('diagnostics') or {}).get('rift_labeler') or {}).get('rift_candidate_debug') or {}
    id_pose={a['actor_id']:pose(a) for a in actors+[scene['ego']]}
    capacity={k:sorted([id_pose.get(i,{'unresolved_id':i}) for i in debug.get(k,[])],
                       key=lambda x:json.dumps(x,sort_keys=True))
              for k in ('existing_cbv_ids','reach_goal_cbv_ids')}
    # Missing saved capacity and saved empty capacity are distinct evidence.
    value={'ego':pose(scene['ego']),'actors':rows,'ego_route':scene['ego_route'],
           'map_lanes':scene.get('map_lanes',[]),'town':scene['town'],
           'native_map_lanes':scene.get('native_map_lanes'),
           'native_map_sha256':(scene.get('native_map_metadata') or {}).get('opendrive_sha256'),
           'capacity':capacity,'capacity_saved':'existing_cbv_ids' in debug,
           'config':{k:debug[k] for k in ('search_radius','max_interaction_dis','max_interaction_fov',
                     'min_interaction_dis_diff','max_agent_num') if k in debug}}
    return value


def replay_fingerprint(scene):
    return hashlib.sha256(json.dumps(replay_inputs(scene),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def pair_job(path, source_bytes, record, sample):
    payload=json.loads(source_bytes);scene=payload.get('scene',payload)
    provenance=record.get('source_snapshot') or {}
    if provenance.get('sha256') and provenance['sha256']!=hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('Recorded source snapshot SHA differs')
    if scene.get('coordinate_frame')!='carla_world' or scene.get('scene_id')!=base_scene_id(record) or \
            scene.get('town')!=record.get('town') or sample['scene_id']!=record['scene_id']:
        raise ValueError('Scene/town/frame mismatch')
    condition=record['condition']
    if record.get('episode_key') and scene.get('route_id') is not None:
        episode='%s:%s:%s:%s' % (scene.get('town'),scene.get('route_id'),scene.get('route_index'),scene.get('repetition_index'))
        if episode!=record['episode_key']:raise ValueError('Episode identity differs')
    actual={p['actor_id']:local_state(scene['ego'],p) for p in scene['agents']}
    def identify(states):
        ids=[]
        for state in states:
            options=[i for i,s in actual.items() if matches(s,state)]
            if len(options)!=1:
                raise ValueError('Reference state has %d snapshot matches' % len(options))
            ids.append(options[0])
        if len(ids)!=len(set(ids)):raise ValueError('Repeated reference actor')
        return ids
    targets=record['target']['generated_agent_states']
    target_ids=identify(targets)
    context_ids=identify(condition.get('observed_agent_states',[]))
    if set(target_ids)&set(context_ids):raise ValueError('Target/context overlap')
    audit=record.get('rift_audit') or {}
    if audit:
        # Reproduce the original builder's FOV filtering, distance order and
        # sparse kept indices. Sparse subsets alone are weak source evidence.
        eligible=[p for p in scene['agents'] if
                  float(p.get('ego_distance',math.dist([p['x'],p['y']],[scene['ego']['x'],scene['ego']['y']])))<=80.
                  and max(abs(v) for v in actual[p['actor_id']][:2])<=32.]
        eligible.sort(key=lambda p:(float(p.get('ego_distance',math.dist([p['x'],p['y']],
                                                   [scene['ego']['x'],scene['ego']['y']]))),p['actor_id']))
        ranked=[p['actor_id'] for p in eligible]
        if 'removed_positive_local_ids' in audit:
            if [ranked.index(i) if i in ranked else -1 for i in target_ids]!=audit['removed_positive_local_ids']:
                raise ValueError('Builder target ranks differ')
        full_context=[i for i in ranked if i not in target_ids]
        if 'observed_agent_count_before_context_sampling' in audit and \
                len(full_context)!=audit['observed_agent_count_before_context_sampling']:
            raise ValueError('Builder original context count differs')
        indices=(record.get('context_sampling') or {}).get('kept_observed_indices')
        if indices is not None:
            if any(i<0 or i>=len(full_context) for i in indices) or [full_context[i] for i in indices]!=context_ids:
                raise ValueError('Builder sparse context indices differ')
        positive_ids={l['actor_id'] for l in scene.get('rift_labels',[]) if l.get('is_interactive')}
        if 'num_rift_positive_agents_original' in audit and \
                len(set(ranked)&positive_ids)!=audit['num_rift_positive_agents_original']:
            raise ValueError('Builder original positive count differs')
    if not matches(local_state(scene['ego'],scene['ego']),condition['ego_state']):
        raise ValueError('Ego state differs')
    raw_route=_array(condition_from_snapshot(scene)['ego_route']['polyline'])
    reference=_array(condition['ego_route']['polyline'])
    if raw_route is None or reference is None or len(reference)>len(raw_route) or \
            np.linalg.norm(raw_route[:len(reference)]-reference,axis=1).max()>.03:
        raise ValueError('Ego route differs')
    states=sample['generated']['generated_agent_states']
    if len(states)!=2 or len(target_ids)!=2:raise ValueError('Expected two targets and two S1 outputs')
    return {'snapshot_path':str(Path(path).resolve()),'town':scene['town'],
            'snapshot_sha256':hashlib.sha256(source_bytes).hexdigest(),
            'snapshot_replay_sha256':replay_fingerprint(scene),
            'dataset_scene_id':record['scene_id'],'raw_scene_id':scene['scene_id'],
            'scene_id':record['scene_id'],'episode_key':record.get('episode_key'),
            'target_actor_ids':target_ids,'context_actor_ids':context_ids,
            'reference_target_states':targets,'candidate_condition':condition,
            'generated_agent_states':states,'generated_blueprint':'vehicle.tesla.model3'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-jsonl',type=Path,required=True)
    p.add_argument('--generated-jsonl',type=Path,required=True)
    p.add_argument('--snapshot-dir',type=Path,required=True)
    p.add_argument('--output-manifest',type=Path,required=True)
    p.add_argument('--max-records',type=int,default=100,help='0=all; otherwise fixed dataset prefix')
    p.add_argument('--debug-snapshot-limit',type=int,default=20,help='Copy unresolved source candidates into evidence bundle; 0 disables')
    p.add_argument('--selection',choices=('prefix','stratified'),default='prefix')
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    report_path=Path(str(a.output_manifest)+'.audit.json')
    partial_path=Path(str(a.output_manifest)+'.partial.jsonl')
    if a.max_records<0:raise ValueError('--max-records must be >= 0')
    if a.output_manifest.exists() or report_path.exists() or partial_path.exists():raise FileExistsError('Refuse to overwrite manifest/evidence')
    paths=snapshot_paths(a.snapshot_dir)
    print('Snapshot directory:',a.snapshot_dir,'JSON files:',len(paths),flush=True)
    chosen=select_records(a.dataset_jsonl,a.max_records,a.selection,a.seed)
    indices=[i for i,r in chosen];records=[r for i,r in chosen]
    index_lookup={index:slot for slot,index in enumerate(indices)}
    if not records:raise ValueError('Empty dataset selection')
    samples={}
    for r in jsonl(a.generated_jsonl):
        index=r.get('record_idx',r.get('record_index'))
        if index not in index_lookup:continue
        index=index_lookup[index]
        if r.get('sample_idx',r.get('sample_index'))!=0 or index in samples:
            raise ValueError('Requires one unique S1 per selected record')
        samples[index]=r
    if len(samples)!=len(records):raise ValueError('Missing S1 outputs for selected records')
    wanted=defaultdict(list)
    for i,r in enumerate(records):wanted[(r.get('town'),base_scene_id(r))].append(i)
    found=defaultdict(dict);counts=Counter();exceptions=Counter();mismatch_examples=defaultdict(list)
    for path in paths:
        counts['files_scanned']+=1
        if counts['files_scanned']%1000==0:print('scanned',counts['files_scanned'],flush=True)
        try:
            data=path.read_bytes();payload=json.loads(data);scene=payload.get('scene',payload)
            key=(scene.get('town'),scene.get('scene_id'))
        except (OSError,ValueError,AttributeError):
            counts['invalid_or_non_snapshot_files']+=1;continue
        for index in wanted.get(key,[]):
            try:
                job=pair_job(path,data,records[index],samples[index])
                job['record_index']=indices[index]
                found[index][job['snapshot_sha256']]=job
            except (ValueError,KeyError,TypeError) as exc:
                exceptions[str(exc)]+=1
                if len(mismatch_examples[index])<3:
                    mismatch_examples[index].append({'snapshot_path':str(path),'reason':str(exc)})
    jobs=[];missing=[];ambiguous=[];ambiguity_details={};equivalent_duplicates={}
    for i in range(len(records)):
        values=list(found[i].values())
        groups=defaultdict(list)
        for j in values:groups[j['snapshot_replay_sha256']].append(j)
        if len(groups)==1:
            chosen=min(values,key=lambda j:j['snapshot_path'])
            chosen['source_match']='unique_content' if len(values)==1 else 'exact_replay_equivalent_sources'
            if len(values)>1:
                alternatives=[{k:j[k] for k in ('snapshot_path','snapshot_sha256','snapshot_replay_sha256')} for j in values]
                chosen['equivalent_source_files']=alternatives
                equivalent_duplicates[indices[i]]=alternatives
            jobs.append(chosen)
        elif not values:missing.append(indices[i])
        else:
            ambiguous.append(indices[i])
            ambiguity_details[indices[i]]=[{k:j[k] for k in ('snapshot_path','snapshot_sha256','snapshot_replay_sha256',
                                                  'target_actor_ids','context_actor_ids')} for j in values]
    a.output_manifest.parent.mkdir(parents=True,exist_ok=True)
    passed=not (missing or ambiguous)
    written_path=a.output_manifest if passed else partial_path
    written_path.write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs),encoding='utf-8')
    debug_files={}
    if ambiguous and a.debug_snapshot_limit>0:
        debug_dir=Path(str(a.output_manifest)+'.debug_snapshots')
        for i in ambiguous:
            for item in ambiguity_details[i][:3]:
                digest=item['snapshot_sha256']
                if digest in debug_files or len(debug_files)>=a.debug_snapshot_limit:continue
                debug_dir.mkdir(parents=True,exist_ok=True)
                saved=debug_dir/(digest+'.json')
                data=Path(item['snapshot_path']).read_bytes()
                if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Snapshot changed during pairing')
                saved.write_bytes(data);debug_files[digest]=str(saved)
    report={'requested_records':len(records),'written_jobs':len(jobs),
            'missing_record_indices':missing,'ambiguous_record_indices':ambiguous,
            'ambiguity_details':ambiguity_details,'exact_replay_equivalent_duplicates':equivalent_duplicates,
            'unresolved_reference_records':{i:{k:records[index_lookup[i]][k] for k in ('scene_id','episode_key','rift_audit',
                'context_sampling','target') if k in records[index_lookup[i]]} for i in (ambiguous+missing)[:20]},
            'mismatch_examples':{i:mismatch_examples[index_lookup[i]] for i in missing+ambiguous},
            'debug_snapshot_files':debug_files,
            'sparse_variant_records':sum(base_scene_id(r)!=r['scene_id'] for r in records),
            'counts':dict(counts),'candidate_mismatch_reasons':dict(exceptions),
            'dataset_sha256':sha(a.dataset_jsonl),'generated_sha256':sha(a.generated_jsonl),
            'manifest_sha256':sha(written_path) if passed else None,
            'manifest_path':str(a.output_manifest) if passed else None,
            'partial_manifest_path':str(partial_path) if not passed else None,
            'partial_manifest_sha256':sha(partial_path) if not passed else None,
            'snapshot_directory':str(a.snapshot_dir.resolve()),
            'selection':'all' if a.max_records==0 else a.selection,
            'selected_dataset_indices':indices,'selection_seed':a.seed,
            'passed':passed}
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('ambiguity_details',
        'exact_replay_equivalent_duplicates','unresolved_reference_records','mismatch_examples')},indent=2,ensure_ascii=False))
    print('Full pairing evidence:',report_path,flush=True)
    if not report['passed']:raise SystemExit('Pairing incomplete/ambiguous; do not report this subset as full evaluation')


if __name__=='__main__':main()
