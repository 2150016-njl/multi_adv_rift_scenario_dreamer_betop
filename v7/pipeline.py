"""P0/P1 construction and preliminary P2 diagnostics; no training or chart."""
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
from .io import sha,read_jsonl,write_json,write_jsonl,initialize_run,resolve_source
from .coordinates import infer_frame_from_map,infer_frame_from_observed_actor,state_to_local,world_to_local,local_to_world,in_domain,wrap
from .condition_builder import build_condition
from .schema import validate_triplets
from .approach_builder import ConflictAnchorBuilder,ApproachBuilder
from .diagnostics import inspect_actor,legacy_reasons


def unique_index(records,name,index_key='record_index'):
    indexed={}
    for row in records:
        index=row.get(index_key,row.get('record_idx'))
        if index is None or index in indexed:raise ValueError('Duplicate/missing index in '+name)
        indexed[index]=row
    return indexed


def freeze_sources(root,experiment,dataset):
    eval_dir=Path(root)/'data/hv_init/eval/v2_v3_same_v3test_full_20261006'
    dataset_rows=read_jsonl(dataset);pair_index=unique_index(read_jsonl(experiment/'v2_pairs.jsonl'),'pairs')
    index=[];checkpoint_info={};outputs={}
    for model in ('v2','v3'):
        report=json.loads((eval_dir/(model+'_raw_s1.json')).read_text(encoding='utf-8'))
        source=resolve_source(report['checkpoint'],root)
        actual_sha=sha(source) if source else None
        checkpoint_info[model]={'reported_path':report['checkpoint'],'reported_sha256':report['checkpoint_sha256'],
            'actual_sha256':actual_sha,'status':'verified' if source and actual_sha==report['checkpoint_sha256'] else 'unverified_missing_weight' if source is None else 'hash_mismatch',
            'metadata_status':'not_loaded','train_config_status':'unverified','train_config_path':None}
        config_path=Path(report['checkpoint']).parent/'train_config.json'
        config=resolve_source(str(config_path),root)
        if config:
            checkpoint_info[model].update(train_config_path=str(config),train_config_sha256=sha(config),
                                          train_config_status='file_present_not_bound_to_weight_metadata')
        outputs[model]=unique_index(read_jsonl(eval_dir/(model+'_raw_s1_generated.jsonl')),model+' S1')
        if len(outputs[model])!=len(dataset_rows):raise ValueError('S1/dataset length mismatch')
        if report['jsonl_sha256']!=sha(dataset):raise ValueError('Evaluation dataset SHA mismatch')
    for i,row in enumerate(dataset_rows):
        records={m:outputs[m].get(i) for m in outputs}
        if any(r is None or r['scene_id']!=row['scene_id'] or r.get('sample_idx',r.get('sample_index'))!=0 for r in records.values()):
            raise ValueError('S1 identity/index mismatch at %d'%i)
        pair=pair_index.get(i)
        index.append({'record_index':i,'dataset_internal_idx':row.get('idx'),'scene_id':row['scene_id'],
                      'source_path':pair['snapshot_path'] if pair else None,
                      'source_sha256':pair['snapshot_sha256'] if pair else None,
                      'source_binding_status':'verified_historical_pair' if pair else 'unverified_not_paired',
                      'sample_seeds':{m:r.get('sample_seed') for m,r in records.items()}})
    return {'checkpoints':checkpoint_info,'full_s1_records':len(index),
            'historically_bound_sources':sum(r['source_path'] is not None for r in index),
            'p0_complete':False,'remaining_gate':'Unpaired full-test sources, actual weights and metadata require verification.'},index


def observation_for(root,pair,ledger,snapshot_root=None,allow_reconstruction=False):
    path=resolve_source(pair['snapshot_path'],root,snapshot_root)
    withheld=set(pair['target_actor_ids'])
    if path:
        if sha(path)!=pair['snapshot_sha256']:raise ValueError('Snapshot SHA mismatch: '+str(path))
        payload=json.loads(path.read_text(encoding='utf-8'));scene=payload.get('scene',payload)
        if scene.get('coordinate_frame')!='carla_world':raise ValueError('Snapshot must declare carla_world')
        observation={'ego':scene['ego'],'actors':[a for a in scene['agents'] if a['actor_id'] not in withheld],
                     'ego_route':scene['ego_route'],'existing_cbv_ids':scene.get('existing_cbv_ids'),
                     'observation_view':'withheld_real_actor_all_remaining_recorded_traffic'}
        # Actor identity/state fidelity is checked before reusing historical oracle.
        actors={a['actor_id']:a for a in scene['agents']}
        for actor in ledger['raw_pool']:
            source=actors.get(actor['actor_id'])
            if not source or any(abs(source[k]-actor['pose'][k])>1.e-5 for k in ('x','y','z','yaw_deg')):
                raise ValueError('Snapshot/ledger actor state mismatch')
        return observation,{'snapshot_status':'sha_verified','resolved_path':str(path),'frame_fit_residual_m':None,
                           'snapshot_internal_idx':scene.get('idx',payload.get('idx'))}
    if not allow_reconstruction:raise FileNotFoundError('Missing real snapshot '+pair['snapshot_path'])
    if ledger['audit_map_lanes']:
        ego,residual=infer_frame_from_map(ledger['audit_map_lanes'],ledger['v6_condition']['map']['lanes_v6'])
        frame_source='world_local_map_pairs'
    else:
        visible={str(a['actor_id']):a['pose'] for a in ledger['raw_pool'] if a['actor_id'] not in withheld}
        token=next((t for t in ledger['v6_condition']['actor_tokens'] if t['actor_token'] in visible),None)
        if token is None:raise ValueError('No non-target observation available to recover frame')
        ego=infer_frame_from_observed_actor(visible[token['actor_token']],token)
        residual=None;frame_source='visible_non_withheld_actor_pair'
    state=pair['candidate_condition']['ego_state']
    ego.update(speed_mps=state[3],length=state[4],width=state[5])
    return {'ego':ego,'actors':[a['pose'] for a in ledger['raw_pool'] if a['actor_id'] not in withheld],
            'ego_route_local_xy':ledger['v6_condition']['ego_route']['polyline'],
            'existing_cbv_ids':None,'observation_view':'withheld_real_actor_ledger_reconstruction'}, {
            'snapshot_status':'missing_explicit_ledger_reconstruction','resolved_path':None,'frame_fit_residual_m':residual,
            'frame_reconstruction_source':frame_source}


def target_record(record_id,actors):
    historical=[{'actor_id':a['actor_id'],'type_id':a['type_id'],'state_xy_yaw_speed':a['local_state'][:4],
                 'world_pose':a['pose'],'historical_actor_native_positive':a['native']['native_candidate'],
                 'fixed_sedan_geometry_compatible':None,'fixed_sedan_raw_spawn_accepted':None} for a in actors]
    front=[a for a in historical if in_domain(a['state_xy_yaw_speed'],True)]
    ordered=front[:1]+[a for a in historical if a not in front[:1] and in_domain(a['state_xy_yaw_speed'])]
    ordered=ordered[:2] if front else []
    return {'schema_version':'hv_v7_target_v1','record_id':record_id,
            'target_origin':'withheld_real_actor','historical_targets':historical,
            'num_targets':len(ordered) if ordered else None,
            'valid_mask':[i<len(ordered) for i in range(2)] if ordered else None,
            'target_states':[a['state_xy_yaw_speed'] for a in ordered]+[None]*(2-len(ordered)) if ordered else None,
            'target_actor_ids':[a['actor_id'] for a in ordered]+[None]*(2-len(ordered)) if ordered else None,
            'target_label_status':'historical_native_only_sedan_and_support_pending' if front else 'no_saved_front_target_NOT_zero_label',
            'training_eligible':False,'training_eligibility_reason':'P1/P2/sedan/chart gates incomplete'}


def build(root,experiment,dataset,output,config):
    root=Path(root);experiment=Path(experiment);dataset=Path(dataset);output=Path(output)
    inputs={'v2_manifest':experiment/'v2_pairs.jsonl','v3_manifest':experiment/'v3_pairs.jsonl',
            'native_v2':experiment/'v2_native/rows.jsonl','native_v3':experiment/'v3_native/rows.jsonl',
            'native_summary':experiment/'v2_native/summary.json',
            'dataset':dataset,'recognizer':root/'RIFT/rift/cbv/recognition/config/rule.yaml',
            'route_document':root/'1015技术路线.md'}
    eval_dir=root/'data/hv_init/eval/v2_v3_same_v3test_full_20261006'
    for model in ('v2','v3'):
        for suffix in ('_raw_s1.json','_raw_s1_generated.jsonl'):
            inputs[model+suffix]=eval_dir/(model+suffix)
    calibrated_dir=config.get('calibrated_map_dir')
    if calibrated_dir:
        calibrated_dir=Path(calibrated_dir)
        if not calibrated_dir.is_dir():raise FileNotFoundError('Calibrated map directory missing: '+str(calibrated_dir))
        # Require ALL selected indices, no silent fallback to the old audit graph.
        expected_indices=config.get('record_indices') or sorted(unique_index(read_jsonl(experiment/'v2_pairs.jsonl'),'pairs'))
        if config.get('max_records'):expected_indices=expected_indices[:config['max_records']]
        for idx in expected_indices:
            path=calibrated_dir/('record_%s_map.json'%idx)
            if not path.is_file():raise FileNotFoundError('Missing calibrated map for record %s: %s'%(idx,path))
            inputs['calibrated_map_%s'%idx]=path
    hashes=initialize_run(output,config,inputs,config.get('expected_input_sha256'))
    native_summary=json.loads(inputs['native_summary'].read_text(encoding='utf-8'))
    if native_summary['config_sha256']!=hashes['recognizer']:
        raise ValueError('Current recognition config differs from historical native ledger')
    freeze,index=freeze_sources(root,experiment,dataset)
    write_json(output/'source_manifest.json',freeze);write_jsonl(output/'full_s1_source_index.jsonl',index)
    data=read_jsonl(dataset);pairs=unique_index(read_jsonl(inputs['v2_manifest']),'v2 pairs')
    pairs3=unique_index(read_jsonl(inputs['v3_manifest']),'v3 pairs')
    native=unique_index(read_jsonl(inputs['native_v2']),'native v2')
    native3=unique_index(read_jsonl(inputs['native_v3']),'native v3')
    calibration_indices = (
        sorted(pairs)
        if config.get('calibration_all_records')
        else [i for i in (2325, 155, 1532, 160, 40) if i in pairs]
    )
    plan = {
        'schema_version': 'hv_v7_calibration_plan_v1',
        'carla_version': '0.9.15',
        'recognition_config': str(inputs['recognizer']),
        'recognition_config_sha256': hashes['recognizer'],
        'fixed_blueprint': 'vehicle.tesla.model3',

        'run_mode': (
            'map_only'
            if config.get('calibration_all_records')
            else 'full'
        ),

        'jobs': [
            {
                'record_index': i,
                'snapshot_path': pairs[i]['snapshot_path'],
                'snapshot_sha256': pairs[i]['snapshot_sha256'],
                'target_actor_ids': pairs[i]['target_actor_ids'],
                'town': native[i]['town'],
                'opendrive_sha256': native[i]['carla_opendrive_sha256'],
                'historical_gt': native[i]['selected_gt']
            }
            for i in calibration_indices
        ],

        'status': 'prepared_not_run',
        'model_evaluation': False
    }
    write_json(output/'calibration_plan.json',plan)
    selection=config.get('record_indices') or list(pairs)
    if len(set(selection))!=len(selection):raise ValueError('Duplicate selected record indices')
    if config.get('max_records',0):selection=selection[:config['max_records']]
    conditions=[];targets=[];oracles=[];map_rows=[];approach_rows=[];jobs=[]
    try:
        for sequence,idx in enumerate(selection,1):
            pair=pairs[idx];row=native[idx];other=native3[idx]
            for field in ('snapshot_path','snapshot_sha256','target_actor_ids','candidate_condition','reference_target_states'):
                if pair[field]!=pairs3[idx][field]:raise ValueError('V2/V3 condition/source mismatch: '+field)
            if row['source_sha256']!=pair['snapshot_sha256'] or row['selected_gt']!=other['selected_gt']:
                raise ValueError('GT repeatability/source mismatch')
            if row.get('errors'):raise ValueError('Native input has execution errors')
            observation,evidence=observation_for(root,pair,row,config.get('snapshot_root'),config.get('allow_ledger_reconstruction',False))
            record_id='v7:'+pair['snapshot_sha256'][:20]+':test'+str(idx)+':withheld-pair'
            provenance={'source_path':pair['snapshot_path'],'source_sha256':pair['snapshot_sha256'],
                'town':row['town'],'episode_key':pair['episode_key'],'frame_id':row['frame_id'],
                'split_group':pair['episode_key'],'record_index':idx,'dataset_internal_idx':data[idx].get('idx'),
                'snapshot_internal_idx':evidence.get('snapshot_internal_idx'),
                'opendrive_sha256':row['carla_opendrive_sha256'],'observation_evidence':evidence,
                'historical_ledger_sha256':hashes['native_v2']}
            map_lanes=row['audit_map_lanes'];map_evidence=None
            if calibrated_dir:
                payload=json.loads((calibrated_dir/('record_%s_map.json'%idx)).read_text(encoding='utf-8'))
                if payload.get('record_index')!=idx or payload.get('snapshot_sha256')!=pair['snapshot_sha256']:
                    raise ValueError('Calibrated map record/snapshot identity mismatch for record %s'%idx)
                if payload.get('opendrive_sha256')!=row['carla_opendrive_sha256'] or payload.get('town')!=row['town']:
                    raise ValueError('Calibrated map OpenDRIVE/Town mismatch for record %s'%idx)
                map_evidence=payload['evidence']
                if map_evidence.get('export_schema_version')!='hv_v7_native_map_v2':
                    raise ValueError('Unrecognized calibrated map schema for record %s'%idx)
                map_lanes=payload['lanes']
                provenance['map_input_kind']='calibrated_carla_map'
                provenance['calibrated_map_sha256']=hashes['calibrated_map_%s'%idx]
            else:
                provenance['map_input_kind']='historical_audit_map'
            condition=build_condition(record_id,observation,map_lanes,provenance,map_evidence=map_evidence)
            anchors=ConflictAnchorBuilder().build(condition)
            approaches=ApproachBuilder().build(condition,anchors)
            condition['condition'].update(conflict_anchors=anchors,approaches=approaches)
            target=target_record(record_id,row['selected_gt'])
            oracle={'schema_version':'hv_v7_oracle_v1','record_id':record_id,
                    'historical_selected_gt':row['selected_gt'],'historical_raw_pool':row['raw_pool'],
                    'old_generated_same_condition':{'v2':row['generated'],'v3':other['generated']},
                    'old_candidate_condition':pair['candidate_condition'],'legacy_v6_condition':row['v6_condition'],
                    'native_status':'historical_ledger_only_new_replay_not_run','native_config_sha256':hashes['recognizer'],
                    'new_spawn_status':'not_run','fixed_sedan_status':'not_run'}
            ego=observation['ego'];position_errors=[];yaw_errors=[]
            for actor in row['raw_pool']:
                xyz=[actor['pose'][k] for k in ('x','y','z')]
                position_errors.append(float(np.linalg.norm(np.asarray(local_to_world(ego,world_to_local(ego,xyz)))-xyz)))
                local=state_to_local(ego,actor['pose'])
                position_errors.append(float(np.linalg.norm(np.asarray(local[:2])-actor['local_state'][:2])))
                yaw_errors.append(abs(wrap(local[2]-actor['local_state'][2])))
            graph=condition['condition']['directed_map']
            map_rows.append({'record_id':record_id,'record_index':idx,'snapshot_status':evidence['snapshot_status'],
                'coordinate_max_error_m':max(position_errors,default=0.),'yaw_max_error_rad':max(yaw_errors,default=0.),
                'route_identity_complete':condition['condition']['ego_route']['identity_complete'],
                'route_points_without_lane_match':sum(not p['lane_matches'] for p in condition['condition']['ego_route']['waypoints']),
                'boundary_edges':graph['boundary_edges'],'reciprocity_mismatches':graph['reciprocity_mismatches'],
                'topology_complete':graph['topology_complete'],'station_identity_complete':graph['station_identity_complete'],
                'new_native_replay':'not_run','fixed_sedan_spawn':'not_run'})
            # Diagnose selected GT (including rear slot evidence) and all raw front positives separately.
            actors=[('selected_gt',a) for a in row['selected_gt']]+[('raw_front_positive',a) for a in row['raw_pool']
                         if a['forward_fov'] and a['native']['native_candidate']]
            for cohort,actor in actors:
                result=inspect_actor(condition,actor,anchors,approaches)
                result.update(record_id=record_id,record_index=idx,cohort=cohort,town=row['town'],
                              legacy=legacy_reasons(actor,pair['candidate_condition'],row['v6_condition']))
                approach_rows.append(result)
            conditions.append(condition);targets.append(target);oracles.append(oracle)
            jobs.append({'record_id':record_id,'record_index':idx,'source_path':pair['snapshot_path'],'source_sha256':pair['snapshot_sha256']})
            print('V7 built %d/%d record_index=%d snapshot=%s'%(sequence,len(selection),idx,evidence['snapshot_status']),flush=True)
        validate_triplets(conditions,targets,oracles)
        cases=output/'cases';cases.mkdir()
        for condition,target,oracle in zip(conditions,targets,oracles):
            idx=condition['provenance']['record_index']
            if idx not in (2325,155):continue
            for name,item in [('condition',condition),('target',target),('oracle',oracle),
                              ('approach_report',[r for r in approach_rows if r['record_index']==idx])]:
                write_json(cases/('record_%s_%s.json'%(idx,name)),item)
        write_jsonl(output/'conditions.jsonl',conditions);write_jsonl(output/'targets.jsonl',targets);write_jsonl(output/'oracles.jsonl',oracles)
        write_jsonl(output/'jobs.jsonl',jobs);write_jsonl(output/'map_alignment_rows.jsonl',map_rows)
        write_jsonl(output/'rows.jsonl',approach_rows)
        summary=summarize(map_rows,approach_rows)
        summary['map_input_kind_counts']=dict(Counter(c['provenance']['map_input_kind'] for c in conditions))
        summary['route_points_without_lane_match_total']=sum(r['route_points_without_lane_match'] for r in map_rows)
        summary['calibrated_map_files_sha_bound']=sum(bool(c['provenance'].get('calibrated_map_sha256')) for c in conditions)
        if calibrated_dir and (summary['map_input_kind_counts']!={'calibrated_carla_map':len(conditions)} or
                               summary['calibrated_map_files_sha_bound']!=len(conditions)):
            raise ValueError('Calibrated CARLA map provenance incomplete: no legacy fallback allowed')
        summary['p0']=freeze;summary['selected_records']=selection
        write_json(output/'summary.json',summary)
        write_json(output/'protocol_sha256.json',{name:sha(output/name) for name in ('conditions.jsonl','targets.jsonl','oracles.jsonl')})
    except BaseException as exc:
        with (output/'errors.log').open('a',encoding='utf-8') as stream:stream.write('%s: %s\n'%(type(exc).__name__,exc))
        write_json(output/'summary.json',{'status':'failed','error':str(exc),'completed_condition_records':len(conditions),
                                         'p0':freeze,'p1_complete':False,'p2_complete':False})
        raise
    return summary


def summarize(map_rows,rows):
    groups={}
    for name,predicate in [('selected_front',lambda r:r['cohort']=='selected_gt' and r['front']),
                           ('raw_front_positive',lambda r:r['cohort']=='raw_front_positive'),
                           ('selected_nonfront',lambda r:r['cohort']=='selected_gt' and not r['front'])]:
        group=[r for r in rows if predicate(r)]
        groups[name]={'denominator':len(group),'preliminary_directed_support':sum(r['preliminary_directed_support'] for r in group),
            'preliminary_topology_support':sum(r['preliminary_topology_support'] for r in group),
            'preview_lane_direction_opposed_rows':sum(r['preview_lane_direction_opposed'] for r in group),
            'support_status_counts':dict(Counter(r['support_status'] for r in group)),
            'missing_edge_kinds':dict(Counter(e['kind'] for r in group for e in r['native_route_missing_edges'])),
            'preview_lane_present':sum(r['preview_lane_in_graph'] for r in group),
            'native_route_missing_node_rows':sum(bool(r['native_route_missing_nodes']) for r in group),
            'native_route_missing_edge_rows':sum(bool(r['native_route_missing_edges']) for r in group),
            'failure_reasons':dict(Counter(reason for r in group for reason in r['failure_reasons'])),
            'v5_reported_true':sum(r['legacy']['v5_audit_geometry']['reported'] is True for r in group),
            'v6_reported_true':sum(r['legacy']['v6']['reported'] is True for r in group),
            'v5_miss_reasons':dict(Counter(r['legacy']['v5_audit_geometry']['reason'] for r in group if r['legacy']['v5_audit_geometry']['reported'] is not True)),
            'v6_miss_reasons':dict(Counter(r['legacy']['v6']['reason'] for r in group if r['legacy']['v6']['reported'] is not True)),
            'station_graph_status_counts':dict(Counter(r.get('station_audit',{}).get('status','missing') for r in group)),
            'station_graph_supported_count':sum(r.get('station_audit',{}).get('status')=='supported' for r in group),
            'station_graph_unknown_count':sum(r.get('station_audit',{}).get('status')=='unknown' for r in group),
            'legacy_v5_reproduction_mismatches':sum(r['legacy']['v5_audit_geometry']['reported'] is not r['legacy']['v5_audit_geometry']['recomputed'] for r in group),
            'legacy_v6_reproduction_mismatches':sum(r['legacy']['v6']['reported'] is not r['legacy']['v6']['recomputed'] for r in group)}
    front_by_record={}
    for r in rows:
        if r['cohort']=='selected_gt' and r['front']:
            front_by_record.setdefault(r['record_index'],[]).append(r)
    station_scene={'front_scenes':len(front_by_record),
        'front_scenes_at_least_one_station_graph_supported':sum(any(r.get('station_audit',{}).get('status')=='supported' for r in v) for v in front_by_record.values()),
        'double_front_scenes':sum(len(v)>=2 for v in front_by_record.values()),
        'double_front_scenes_both_station_graph_supported':sum(all(r.get('station_audit',{}).get('status')=='supported' for r in v) for v in front_by_record.values() if len(v)>=2)}
    return {'status':'diagnostic_complete','protocol':'hv_v7_three_file_v1','p1_complete':False,'p2_complete':False,
            'map_records':len(map_rows),'snapshot_sha_verified':sum(r['snapshot_status']=='sha_verified' for r in map_rows),
            'coordinate_max_error_m':max((r['coordinate_max_error_m'] for r in map_rows),default=None),
            'yaw_max_error_rad':max((r['yaw_max_error_rad'] for r in map_rows),default=None),
            'groups':groups,'station_graph_scene_counts':station_scene,
            'station_audit_scope':'oracle_side_station_monotone_graph_only_not_physical_kinematic_or_sedan',
            'native_rerun':'not_run','fixed_sedan_calibration':'not_run',
            'lane_chart':'not_implemented_pending_P2','training':'not_started',
            'limits':['Directed support is not lane-chart, sedan geometry, native rerun or spawn certification.',
                      'Station audit is oracle-only diagnostics, not online candidate/condition input.',
                      'Station progress-monotone connectivity is not a Model3 chart or dynamics certificate.',
                      'Unknown/failed map and spawn are not model negatives.']}


def audit_dataset(dataset,output,config):
    dataset=Path(dataset);output=Path(output)
    inputs={name:dataset/name for name in ('conditions.jsonl','targets.jsonl','oracles.jsonl')}
    expected_path=dataset/'protocol_sha256.json'
    declared=json.loads(expected_path.read_text(encoding='utf-8'))
    if any(sha(path)!=declared[name] for name,path in inputs.items()):raise ValueError('Protocol SHA mismatch')
    initialize_run(output,config,inputs,config.get('expected_input_sha256'))
    conditions,targets,oracles=[read_jsonl(inputs[name]) for name in inputs]
    validate_triplets(conditions,targets,oracles)
    if config.get('max_records'):
        conditions=conditions[:config['max_records']];oracles=oracles[:config['max_records']]
    rows=[]
    for condition,oracle in zip(conditions,oracles):
        anchors=ConflictAnchorBuilder().build(condition);approaches=ApproachBuilder().build(condition,anchors)
        for cohort,actors in [('selected_gt',oracle['historical_selected_gt']),('raw_front_positive',[
                a for a in oracle['historical_raw_pool'] if a['forward_fov'] and a['native']['native_candidate']])]:
            for actor in actors:
                row=inspect_actor(condition,actor,anchors,approaches)
                row.update(record_id=condition['record_id'],record_index=condition['provenance']['record_index'],cohort=cohort,
                           legacy=legacy_reasons(actor,oracle['old_candidate_condition'],oracle['legacy_v6_condition']))
                rows.append(row)
    write_jsonl(output/'rows.jsonl',rows);write_json(output/'summary.json',summarize([],rows))
