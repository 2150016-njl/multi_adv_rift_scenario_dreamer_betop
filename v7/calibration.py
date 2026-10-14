"""Independent CARLA GT/fixed-sedan calibration; no model success labels."""
import copy
import json
import math
from pathlib import Path
from .io import initialize_run,write_json,write_jsonl,sha,resolve_source
from .carla_map import export_directed_lanes


def compare_native(reference,actual,tolerance=1.e-4):
    from .directed_map import lane_uid
    problems=[]
    if reference['native_candidate']!=actual['native_candidate']:problems.append('native_boolean_changed')
    for key in ('interaction_waypoint',):
        a,b=reference.get(key),actual.get(key)
        if bool(a)!=bool(b) or a and lane_uid(a)!=lane_uid(b):problems.append(key+'_identity_changed')
        elif a and any(abs(a[k]-b[k])>tolerance for k in ('x','y','z','s')):problems.append(key+'_position_changed')
    a,b=reference.get('target_route_to_interaction',[]),actual.get('target_route_to_interaction',[])
    if len(a)!=len(b) or any(lane_uid(x)!=lane_uid(y) or any(abs(x[k]-y[k])>tolerance for k in ('x','y','z','s')) for x,y in zip(a,b)):
        problems.append('native_route_changed')
    for key in ('ego_distance','CBV_distance','distance_diff'):
        x=(reference.get('interaction_meta') or {}).get(key);y=(actual.get('interaction_meta') or {}).get(key)
        if (x is None)!=(y is None) or x is not None and abs(x-y)>tolerance:problems.append(key+'_changed')
    return problems


def calibrate(plan_path,output,host,port,timeout,dedicated,config):
    if not dedicated or host not in ('127.0.0.1','localhost') or 37998<=port<=38002:
        raise ValueError('Requires an explicitly dedicated local server outside reserved 38000 block')
    from high_value_init.carla_audit_environment import configure_carla_pythonapi
    configure_carla_pythonapi(validate=True)
    import carla
    from high_value_init.native_state_audit import NativeAudit,spawn_frozen_cohort,destroy_owned_without_tick
    from audit_hv_snapshot_experiment import json_compatible
    plan_path=Path(plan_path);output=Path(output)
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    root=Path(__file__).resolve().parents[1]
    recog=resolve_source(plan['recognition_config'],root)
    if not recog or sha(recog)!=plan['recognition_config_sha256']:raise ValueError('Recognizer SHA mismatch')
    initialize_run(output,config,{'plan':plan_path,'recognizer':recog},config.get('expected_input_sha256'))
    import yaml
    recognition=yaml.safe_load(recog.read_text(encoding='utf-8'))
    client=carla.Client(host,port);client.set_timeout(timeout)
    if client.get_server_version()!=plan['carla_version']:raise ValueError('CARLA version differs from plan')
    engine=NativeAudit(client,recognition,'eval',1.,True)
    jobs=plan['jobs'][:config['max_records']] if config.get('max_records') else plan['jobs']
    results=[];aborted=False
    with (output/'rows.jsonl').open('x',encoding='utf-8') as stream:
        for job in jobs:
            row={'record_index':job['record_index'],'model_evaluation':False,'errors':[]}
            def progress(phase,**details):
                write_json(output/'current_job.json',{'record_index':job['record_index'],'phase':phase,'details':details})
                print('V7 phase=%s record=%s %s'%(phase,job['record_index'],json.dumps(details)),flush=True)
            engine.progress=progress
            if aborted:
                row.update(status='not_run_after_cleanup_or_isolation_failure')
            else:
                settings=None
                try:
                    source=resolve_source(job['snapshot_path'],root)
                    if not source or sha(source)!=job['snapshot_sha256']:raise ValueError('Real snapshot missing/SHA mismatch')
                    scene=json.loads(source.read_text(encoding='utf-8'));scene=scene.get('scene',scene)
                    progress('map_prepare',town=job['town'])
                    client.set_timeout(max(300.,timeout))  # Map load has a distinct bounded budget.
                    engine.prepare(job['town']);world=engine.world
                    client.set_timeout(timeout)
                    if engine.map_sha256!=job['opendrive_sha256']:raise ValueError('OpenDRIVE fingerprint mismatch')
                    if list(world.get_actors().filter('vehicle.*')):raise ValueError('Dedicated world is not empty')
                    route,route_error=engine.resolve_route(scene['ego_route'])
                    progress('map_export')
                    lanes,map_evidence=export_directed_lanes(engine.world_map,scene['ego'],route,planner=engine.grp)
                    write_json(output/('record_%s_map.json'%job['record_index']),{'lanes':lanes,'evidence':map_evidence,
                               'opendrive_sha256':engine.map_sha256})
                    from .condition_builder import build_condition
                    from .approach_builder import ConflictAnchorBuilder,ApproachBuilder
                    from .diagnostics import inspect_actor
                    from .coordinates import state_to_local,in_domain
                    selected_ids=set(job['target_actor_ids'])
                    observation={'ego':scene['ego'],'ego_route':scene['ego_route'],
                        'actors':[a for a in scene['agents'] if a['actor_id'] not in selected_ids],
                        'existing_cbv_ids':None,'observation_view':'withheld_real_actor_all_remaining_recorded_traffic'}
                    map_condition=build_condition('calibration:'+str(job['record_index']),observation,lanes,
                        {'opendrive_sha256':engine.map_sha256,'source_sha256':job['snapshot_sha256']})
                    map_condition['condition']['directed_map']['lane_change_edges_separate_evidence']=map_evidence['lane_change_edges']
                    anchors=ConflictAnchorBuilder().build(map_condition)
                    approaches=ApproachBuilder().build(map_condition,anchors)
                    write_json(output/('record_%s_condition.json'%job['record_index']),map_condition)
                    row.update(map_evidence=map_evidence,route_resolution_max_error_m=route_error)
                    actors={a['actor_id']:a for a in scene['agents']}
                    row['native_comparisons']=[]
                    for reference in job['historical_gt']:
                        progress('native_replay',source_actor_id=reference['actor_id'])
                        pose=actors[reference['actor_id']]
                        actual=engine.actor(scene,pose,route)
                        geometry=engine.geometry(pose)
                        row['native_comparisons'].append({'actor_id':pose['actor_id'],'historical_type_id':pose['type_id'],
                            'historical_actor_native_positive':actual['native_candidate'],'native':actual,'road_geometry':geometry,
                            'differences':compare_native(reference['native'],actual)})
                        from .directed_map import lane_uid
                        previous=reference['road_geometry']['preview_waypoint'];current=geometry['preview_waypoint']
                        if bool(previous)!=bool(current) or previous and lane_uid(previous)!=lane_uid(current):
                            row['native_comparisons'][-1]['differences'].append('preview_lane_identity_changed')
                        state=state_to_local(scene['ego'],pose)
                        row['native_comparisons'][-1]['fresh_map_alignment']=inspect_actor(map_condition,
                            {'actor_id':pose['actor_id'],'type_id':pose['type_id'],'pose':pose,
                             'local_state':state,'forward_fov':in_domain(state,True),
                             'native':actual,'road_geometry':geometry},anchors,approaches)
                    settings=world.get_settings();old_sync=settings.synchronous_mode;old_delta=settings.fixed_delta_seconds
                    settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;world.apply_settings(settings)
                    frame=world.get_snapshot().frame
                    points=engine.world_map.get_spawn_points()
                    if not points:raise ValueError('No official geometry probe spawn point')
                    probe=world.try_spawn_actor(world.get_blueprint_library().find(plan['fixed_blueprint']),points[0])
                    if probe is None:raise ValueError('Single official blueprint geometry probe rejected; no retry')
                    try:
                        box=probe.bounding_box
                        sedan_size=[2*box.extent.x,2*box.extent.y,2*box.extent.z]
                        row['fixed_blueprint_box']={'type_id':probe.type_id,'size_lwh_m':sedan_size,
                            'offset_xyz_m':[box.location.x,box.location.y,box.location.z],
                            'probe_source':'official_map_spawn_point_0_geometry_only_not_GT_spawn'}
                    finally:
                        errors,ack=destroy_owned_without_tick(client,carla,[probe])
                        row['blueprint_probe_cleanup']=ack
                        if errors:aborted=True;raise RuntimeError(str(errors))
                    selected=job['target_actor_ids'];context=[p for p in scene['agents'] if p['actor_id'] not in selected]
                    gt=[actors[i] for i in selected]
                    sedan=[dict(p,type_id=plan['fixed_blueprint'],length=sedan_size[0],width=sedan_size[1],height=sedan_size[2]) for p in gt]
                    row['cohorts']={}
                    for name,targets in [('historical_gt',gt),('fixed_sedan_same_raw_pose',sedan)]:
                        progress('raw_spawn_cohort',cohort=name)
                        result=spawn_frozen_cohort(world,carla,scene['ego'],context,targets,engine=engine,scene=scene)
                        row['cohorts'][name]=result
                        if result['cleanup_errors']:aborted=True;raise RuntimeError('Owned cleanup failed')
                    row['fixed_sedan_labels']=[]
                    fixed=row['cohorts']['fixed_sedan_same_raw_pose']
                    for attempt in fixed['attempts']:
                        if attempt['role']!='target':continue
                        g=fixed['geometry_3d'].get(attempt['source_actor_id'],{})
                        compatible=None
                        if g and fixed['whole_scene_accepted']:
                            compatible=(g['whole_road']['whole_box_in_sampled_driving_polygon'] and not g['peer_3d_overlap_ids'] and
                                        not g['static_3d_box_overlaps'] and not g['static_query_errors'])
                        row['fixed_sedan_labels'].append({'actor_id':attempt['source_actor_id'],
                            'fixed_sedan_geometry_compatible':compatible,
                            'fixed_sedan_raw_spawn_accepted':attempt['accepted'],
                            'context_complete':fixed['context_complete']})
                    if world.get_snapshot().frame!=frame:aborted=True;raise RuntimeError('Unexpected frame advancement')
                    row.update(status='calibration_observed',physics='not_run',post_spawn_pose_corrections=False)
                except Exception as exc:
                    row.update(status='calibration_error');row['errors'].append('%s: %s'%(type(exc).__name__,exc))
                    with (output/'errors.log').open('a',encoding='utf-8') as error:error.write(str(row['errors'])+'\n')
                finally:
                    client.set_timeout(timeout)
                    if settings is not None:
                        try:
                            settings.synchronous_mode=old_sync;settings.fixed_delta_seconds=old_delta
                            world.apply_settings(settings)
                        except Exception as exc:aborted=True;row['errors'].append('Settings restore: '+str(exc))
            clean,numeric=json_compatible(row);clean['nonfinite_observations']=numeric
            stream.write(json.dumps(clean,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();results.append(clean)
            print('V7 calibration record=%s status=%s'%(job['record_index'],clean['status']),flush=True)
    disagreements=sum(bool(a['differences']) for r in results for a in r.get('native_comparisons',[]))
    write_json(output/'summary.json',{'planned_records':len(jobs),'written':len(results),
        'execution_error_records':sum(bool(r['errors']) for r in results),'model_failure_count':None,
        'native_comparison_disagreements':disagreements,
        'p1_complete':False,'physics':'not_run',
        'meaning':'GT/sedan calibration only; no model was evaluated; missing calibration is unknown.'})
    if any(r['errors'] for r in results) or aborted or disagreements:
        raise SystemExit('Calibration incomplete/inconsistent; evidence preserved')
