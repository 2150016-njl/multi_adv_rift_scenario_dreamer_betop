#!/usr/bin/env python3
"""V7 round3: narrow recovery, provenance-preserving maps and rigorous audit gates.

Does not start or modify CARLA and never reads GT while building maps.
All outputs are new directories/files (exclusive creation); failures are fail-closed.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path


def read_json(p):
    with Path(p).open(encoding='utf-8') as f:return json.load(f)


def read_jsonl(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def write_json_new(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,msg):
    if not ok:raise ValueError(msg)


def jobs_index(plan):
    jobs=plan['jobs']; by={int(x['record_index']):x for x in jobs}
    require(len(by)==len(jobs),'Duplicate plan record indices')
    return by


def rows_index(root):
    rows=read_jsonl(Path(root)/'rows.jsonl'); out={int(x['record_index']):x for x in rows}
    require(len(out)==len(rows),'Duplicate calibration rows')
    return out


def plan_retry(a):
    plan=read_json(a.plan);jobs=jobs_index(plan)
    require(plan['run_mode']=='map_only','Retry requires original map_only plan')
    original=rows_index(a.original)
    indices=[int(s) for s in a.indices.split(',')]
    require(len(indices)==len(set(indices)),'Duplicate retry indices')
    require(set(indices).issubset(jobs),'Retry index absent from original plan')
    require(set(original)==set(jobs),'Original map-only rows do not cover the entire plan')
    for i in indices:
        require(original[i]['status']=='calibration_error','Refusing retry of successful record '+str(i))
        require(not (Path(a.original)/f'record_{i}_map.json').exists(),
                'Refusing to retry record with existing map '+str(i))
    out=dict(plan);out['jobs']=[jobs[i] for i in indices]
    out['retry_of_plan_sha256']=sha(a.plan)
    out['retry_reason']='fresh_owned_CARLA_server_to_diagnose_map_sha_cache_vs_live'
    write_json_new(a.output,out)
    print('RETRY_PLAN',a.output,'RECORDS',indices)


def normalize_map(payload):
    """Drop near-zero consecutive XY segments, preserving native station direction.

    This is a DERIVED (not raw) map. Original file hash is kept by the caller.
    No synthetic interpolation or target-based correction is allowed.
    """
    removed=0
    for lane in payload['lanes']:
        line=lane['centerline'];station=lane.get('s_samples_m');width=lane.get('width_m')
        yaw=lane.get('yaw_samples_deg')
        require(len(line)>=2 and station and len(station)==len(line)==len(width)==len(yaw),
                'Invalid sampled lane arrays '+str(lane['lane_uid']))
        sign=lane['station_travel_direction']
        require(sign in (-1,1) and all((b-a)*sign>1e-7 for a,b in zip(station,station[1:])),
                'Nonmonotone native station '+str(lane['lane_uid']))
        # Preserve first endpoint and the *latest* sample among near-duplicates.
        selected=[0]
        for i in range(1,len(line)):
            p,q=line[selected[-1]],line[i]
            if math.hypot(p[0]-q[0],p[1]-q[1])<1e-4:
                selected[-1]=i;removed+=1
            else:selected.append(i)
        require(len(selected)>=2,'Degenerate lane after dedup '+str(lane['lane_uid']))
        for key in ('centerline','s_samples_m','width_m','yaw_samples_deg'):
            lane[key]=[lane[key][i] for i in selected]
        gaps=[abs(b-a) for a,b in zip(lane['s_samples_m'],lane['s_samples_m'][1:])]
        lane['max_station_gap_m']=max(gaps)
        require(all(math.hypot(q[0]-p[0],q[1]-p[1])>=1e-4 for p,q in zip(lane['centerline'],lane['centerline'][1:])),
                'Duplicate XY survived normalization')
    payload['evidence']['normalization']={'algorithm':'drop_adjacent_xy_lt_0p0001m_keep_latest',
                                         'removed_vertices':removed,
                                         'derived_map':True,'synthetic_vertices':0}
    return payload,removed


def merge_maps(a):
    plan=read_json(a.plan);jobs=jobs_index(plan)
    require(plan['run_mode']=='map_only','Not a map_only plan')
    require(len(jobs)==95,'Expected exactly 95 historical records; refuse changed cohort')
    baseline=rows_index(a.original)
    retry={};retry_location={}
    for directory in a.retry:
        current=rows_index(directory)
        for idx,row in current.items():
            require(idx not in retry,'Duplicate retry record %s from %s' %(idx,directory))
            retry[idx]=row
            retry_location[idx]=Path(directory)
    require(set(baseline)==set(jobs),'Original rows do not match 95 planned jobs')
    required_retry={i for i,r in baseline.items() if r['status']!='map_exported'}
    require(required_retry=={342,597},'Unexpected original failure set: '+str(sorted(required_retry)))
    require(set(retry)==required_retry,'Retries must contain exactly failed 342 and 597')
    require(all(retry[i]['status']=='map_exported' and not retry[i].get('errors') for i in retry),
            'Retry results are not fully successful; STOP')
    source_map_sha={};removed_count={};maps=[]
    a.output.mkdir(parents=True,exist_ok=False)
    for idx,job in sorted(jobs.items()):
        src=(retry_location[idx] if idx in retry else Path(a.original))/f'record_{idx}_map.json'
        require(src.is_file(),'Missing source map '+str(src))
        raw=read_json(src)
        require(raw.get('record_index')==idx,'Map record index mismatch '+str(idx))
        for key,plan_key in [('town','town'),('opendrive_sha256','opendrive_sha256'),('snapshot_sha256','snapshot_sha256')]:
            require(raw.get(key)==job[plan_key],f'Map {idx} identity mismatch: {key}')
        require(raw['evidence'].get('export_schema_version')=='hv_v7_native_map_v2',
                'Map exporter schema mismatch: '+str(idx))
        require(raw['lanes'],'Empty map '+str(idx))
        original_sha=sha(src);raw['source_map_sha256']=original_sha
        raw['source_map_file']=str(src)
        clean,removed=normalize_map(raw)
        dest=a.output/f'record_{idx}_map.json'
        write_json_new(dest,clean)
        source_map_sha[str(idx)]=original_sha;removed_count[str(idx)]=removed
        maps.append({'record_index':idx,'town':job['town'],'derived_sha256':sha(dest),
                     'source_sha256':original_sha,'normalized_vertices_removed':removed,
                     'source_kind':'retry' if idx in retry else 'original'})
    write_json_new(a.output/'map_merge_manifest.json',{'plan_sha256':sha(a.plan),
        'original_rows_sha256':sha(Path(a.original)/'rows.jsonl'),
        'retry_rows_sha256_by_directory':{str(p):sha(Path(p)/'rows.jsonl') for p in a.retry},
        'original_calibration_dir':str(a.original),'retry_calibration_dirs':[str(x) for x in a.retry],
        'records':maps,'count':len(maps),'normalized_vertices_removed':sum(removed_count.values()),
        'map_sha256_by_record':source_map_sha})
    print('MERGED',len(maps),'NORMALIZED_REMOVED_VERTICES',sum(removed_count.values()),
          'OUT',a.output)


def full_plan(a):
    plan=read_json(a.plan);jobs=jobs_index(plan)
    indices=[2325,155,1532,160,40]
    require(all(i in jobs for i in indices),'Missing fixed P1 records')
    m=a.merged_maps
    for i in jobs:
        p=m/f'record_{i}_map.json';require(p.exists(),'Map missing before P1: '+str(i))
        raw=read_json(p)
        require(raw['snapshot_sha256']==jobs[i]['snapshot_sha256'] and
                raw['opendrive_sha256']==jobs[i]['opendrive_sha256'],
                'Map identity mismatch before P1: '+str(i))
    out=dict(plan);out['run_mode']='full';out['jobs']=[jobs[i] for i in indices]
    out['map_only_plan_sha256']=sha(a.plan)
    out['merged_manifest_sha256']=sha(m/'map_merge_manifest.json')
    out['scope']='five_independent_P1_native_model3_raw_spawn_no_physics'
    write_json_new(a.output,out)
    print('FULL_PLAN',a.output,'RECORDS',indices)


def gate(a):
    d=a.dataset
    summary=read_json(d/'summary.json')
    rows=read_jsonl(d/'rows.jsonl')
    conds=read_jsonl(d/'conditions.jsonl')
    oracles=read_jsonl(d/'oracles.jsonl')
    targets=read_jsonl(d/'targets.jsonl')
    alignment=read_jsonl(d/'map_alignment_rows.jsonl')
    declared=read_json(d/'protocol_sha256.json')
    inputs=read_json(d/'input_sha256.json')
    require(len(conds)==len(oracles)==len(targets)==len(alignment)==95,
            'Expected 95 matched triplets and 95 map alignment rows')
    require(set(declared)=={'conditions.jsonl','targets.jsonl','oracles.jsonl'},
            'Unrecognized protocol manifest entries')
    for name,hashvalue in declared.items():
        require(sha(d/name)==hashvalue,'Protocol SHA mismatch: '+name)
    require(all(c['record_id']==o['record_id']==t['record_id'] for c,o,t in zip(conds,oracles,targets)),
            'Condition/oracle/target identity mismatch')
    ids=[c['provenance']['record_index'] for c in conds]
    require(len(set(ids))==95,'Duplicate condition record indices')
    require(all(c['provenance'].get('map_input_kind')=='calibrated_carla_map' for c in conds),
            'HISTORICAL MAP FALLBACK DETECTED')
    require(all(c['provenance'].get('calibrated_map_sha256') for c in conds),
            'Missing per-record map SHA binding')
    m=a.merged_maps
    manifest=read_json(m/'map_merge_manifest.json')
    require(manifest['count']==95,'Merged map manifest is not a 95-map cohort')
    manifest_indices={r['record_index'] for r in manifest['records']}
    require(set(ids)==manifest_indices,'Dataset record indices differ from merged map set')
    for c in conds:
        idx=c['provenance']['record_index']
        src=m/f'record_{idx}_map.json'
        h=sha(src)
        require(h==c['provenance']['calibrated_map_sha256'],
                'Dataset map provenance SHA mismatch for %s'%idx)
        require(inputs['calibrated_map_%s'%idx]==h,
                'Dataset input SHA mismatch for map %s'%idx)
        raw=read_json(src)
        for ck,mk in [('source_sha256','snapshot_sha256'),('opendrive_sha256','opendrive_sha256'),('town','town')]:
            require(c['provenance'][ck]==raw[mk],
                    'Calibrated map vs condition mismatch: record %s field %s'%(idx,ck))
        require('station_audit' not in c['condition'],
                'Oracle station audit leaked into condition %s'%idx)
    from v7.schema import validate_triplets
    validate_triplets(conds,targets,oracles)
    front=[r for r in rows if r['cohort']=='selected_gt' and r['front']]
    raw=[r for r in rows if r['cohort']=='raw_front_positive']
    by_record=defaultdict(list)
    for r in front:by_record[r['record_index']].append(r)
    scene_count=len(by_record)
    doubles=sum(len(v)>=2 for v in by_record.values())
    require(len(front)==110 and scene_count==82 and doubles==28,
            'Fixed denominator contract changed: '+str((len(front),scene_count,doubles)))
    require(len(raw)==119,'Raw front-positive denominator changed')
    statuses=Counter(r.get('station_audit',{}).get('status','missing') for r in front)
    require(set(statuses).issubset({'supported','unknown'}),
            'Unexpected station audit status/missing actor results')
    preliminary=sum(bool(r.get('preliminary_directed_support')) for r in front)
    scene_supported=sum(any(r['station_audit']['status']=='supported' for r in v)
                        for v in by_record.values())
    both_supported=sum(all(r['station_audit']['status']=='supported' for r in v)
                       for v in by_record.values() if len(v)>=2)
    raw_status=Counter(r.get('station_audit',{}).get('status','missing') for r in raw)
    missing_rows=[{'record_index':r['record_index'],'actor_id':r['actor_id'],
                  'reason':r['station_audit'].get('reason')}
                 for r in front if r['station_audit']['status']!='supported']
    missing_route=[{'record_index':r['record_index'],
                    'route_points_without_lane_match':r['route_points_without_lane_match']}
                   for r in alignment if r['route_points_without_lane_match']]
    result={'status':'audit_only_not_P2_physics_gate',
            'p0_complete':False,'p1_complete':False,'p2_complete':False,
            'dataset':str(d),'dataset_summary_status':summary.get('status'),
            'map_provenance_confirmed':95,'map_input_kind':'calibrated_carla_map',
            'fixed_selected_front_denominator':110,
            'fixed_front_positive_scenes':82,'fixed_double_front_scenes':28,
            'preliminary_directed_support':preliminary,
            'strict_station_graph_statuses':dict(statuses),
            'strict_station_graph_supported':statuses['supported'],
            'scenes_with_at_least_one_station_supported':scene_supported,
            'double_scenes_both_station_supported':both_supported,
            'raw_front_station_graph_statuses':dict(raw_status),
            'route_points_without_lane_match_total':sum(r['route_points_without_lane_match'] for r in alignment),
            'route_unmatched_by_record':missing_route,
            'station_non_supported_rows':missing_rows,
            'limits':['Station graph uses GT solely in offline diagnostics, never in model conditions.',
                      'Station-monotone path is not continuous lane-chart, collision-free swept Model3, or dynamics certification.',
                      'Unknown is neither negative training supervision nor proof that k=0.',
                      'P1/full native and raw spawn must be independently audited.']}
    write_json_new(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in
           ('station_non_supported_rows','limits','route_unmatched_by_record')},
           indent=2,ensure_ascii=False))
    print('STATION_NOT_SUPPORTED',len(missing_rows),'report',a.output)


def _clean_destroy_evidence(evidence):
    return (isinstance(evidence,dict) and
            evidence.get('method')=='batch_destroy_actor_id_no_tick' and
            all(x.get('error') is None for x in evidence.get('responses',[])))


def p1_report(a):
    rows=rows_index(a.calibration)
    expected={2325,155,1532,160,40}
    require(set(rows)==expected,'Require exactly five fixed full-calibration records')
    run_config=read_json(Path(a.calibration)/'config.json')
    require(run_config.get('command')=='calibrate','Calibration command provenance missing')
    report=[]
    for i,row in sorted(rows.items()):
        cohorts=row.get('cohorts',{})
        gt=cohorts.get('historical_gt',{})
        sedan=cohorts.get('fixed_sedan_same_raw_pose',{})
        both_cohorts=bool(gt) and bool(sedan)
        comparisons=row.get('native_comparisons',[])
        native_clean=bool(comparisons) and all(not x['differences'] for x in comparisons)
        blueprint=row.get('fixed_blueprint_box',{})
        probe=blueprint.get('size_lwh_m')
        offset=blueprint.get('offset_xyz_m')
        geometry_measured=(blueprint.get('type_id')=='vehicle.tesla.model3' and
                           bool(probe) and len(probe)==3 and
                           all(isinstance(x,(int,float)) and math.isfinite(x) and 0.5<x<10 for x in probe) and
                           bool(offset) and len(offset)==3 and
                           all(isinstance(x,(int,float)) and math.isfinite(x) for x in offset))
        context_complete=both_cohorts and all(c.get('context_complete') is True for c in (gt,sedan))
        cleanup_clean=both_cohorts and all(
            not c.get('cleanup_errors') and
            c.get('world_ticks_requested')==0 and
            c.get('probe_ticks_requested')==0 and
            _clean_destroy_evidence(c.get('cleanup_evidence')) for c in (gt,sedan))
        cleanup_clean=bool(cleanup_clean and _clean_destroy_evidence(row.get('blueprint_probe_cleanup')))
        all_attempts=[x for c in (gt,sedan) for x in c.get('attempts',[])]
        raw_fidelity=both_cohorts and bool(all_attempts) and all(
            x.get('raw_request_fidelity') is True for x in all_attempts)
        full_scene=both_cohorts and gt.get('whole_scene_accepted') is True and sedan.get('whole_scene_accepted') is True
        geometry_errors=sum(len(c.get('geometry_errors',[])) for c in (gt,sedan))
        status_ok=row.get('status')=='calibration_observed' and not row.get('errors')
        pose_uncorrected=row.get('post_spawn_pose_corrections') is False
        no_physics_claim=row.get('physics')=='not_run'
        sedan_target_attempts=[x for x in sedan.get('attempts',[]) if x.get('role')=='target']
        report.append({'record_index':i,'run_status':row.get('status'),
          'errors':row.get('errors',[]),'native_replay_consistent':native_clean,
          'native_comparisons':len(comparisons),
          'fixed_blueprint_box_m':probe,'fixed_blueprint_offset_xyz_m':offset,
          'blueprint_geometry_measured':bool(geometry_measured),
          'context_complete':bool(context_complete),
          'cleanup_without_tick':bool(cleanup_clean),
          'raw_request_fidelity':bool(raw_fidelity),
          'both_whole_scenes_spawned':bool(full_scene),
          'model3_target_attempts':len(sedan_target_attempts),
          'model3_targets_spawned':sum(x.get('accepted') is True for x in sedan_target_attempts),
          'geometry_error_count':geometry_errors,
          'post_spawn_pose_corrections':row.get('post_spawn_pose_corrections'),
          'fixed_geometry_compatible_labels':row.get('fixed_sedan_labels',[]),
          'p1_strict_observed_complete':bool(all((status_ok,native_clean,geometry_measured,
             context_complete,cleanup_clean,raw_fidelity,full_scene,
             geometry_errors==0,pose_uncorrected,no_physics_claim)))})
    measured=[r['fixed_blueprint_box_m'] for r in report if r['blueprint_geometry_measured']]
    identical=bool(len(measured)==5 and all(
        all(abs(x-y)<1e-3 for x,y in zip(measured[0],v)) for v in measured[1:]))
    output={'status':'full_P1_evidence_review','records':report,
            'model3_box_consistent_across_five':identical,
            'all_five_strict_P1_pass':identical and all(r['p1_strict_observed_complete'] for r in report),
            'no_model_or_physics_claim':True,
            'notes':['Rejected raw spawn is a measured failure/unknown geometry; do not delete or reposition actors.',
                     'Full-scene acceptance is a strict audit gate, not a requirement to fabricate positive labels.',
                     'No ticks means dynamic/velocity realization remains untested.']}
    write_json_new(a.output,output)
    print('FULL_P1_STRICT_5',output['all_five_strict_P1_pass'],'report',a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('plan-retry');r.add_argument('--plan',type=Path,required=True)
    r.add_argument('--original',type=Path,required=True)
    r.add_argument('--indices',default='342,597');r.add_argument('--output',type=Path,required=True)
    m=sub.add_parser('merge-maps');m.add_argument('--plan',type=Path,required=True)
    m.add_argument('--original',type=Path,required=True);m.add_argument('--retry',type=Path,required=True,action='append')
    m.add_argument('--output',type=Path,required=True)
    f=sub.add_parser('plan-full');f.add_argument('--plan',type=Path,required=True)
    f.add_argument('--merged-maps',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    g=sub.add_parser('gate-p2');g.add_argument('--dataset',type=Path,required=True)
    g.add_argument('--merged-maps',type=Path,required=True)
    g.add_argument('--output',type=Path,required=True)
    h=sub.add_parser('gate-p1');h.add_argument('--calibration',type=Path,required=True)
    h.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    {'plan-retry':plan_retry,'merge-maps':merge_maps,'plan-full':full_plan,
     'gate-p2':gate,'gate-p1':p1_report}[args.command](args)


if __name__=='__main__':main()
