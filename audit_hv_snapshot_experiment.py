#!/usr/bin/env python3
"""Audit raw snapshot pools and matched GT/generated states in three stages.

snapshot: saved labels + V5 geometry coverage + 2D snapshot OBB occupancy.
native: same states through CARLA road queries and isolated original RIFT code.
spawn: native + frozen static fleet reconstruction on a dedicated empty server.

Inputs are raw world-coordinate snapshots/online dumps, not converted JSONL.
No model sampling, target rewriting, spawn retry or native CBV activation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import faulthandler

import numpy as np
from hv_audit_inputs import manifest_jobs, snapshot_paths

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scenario-dreamer'))
sys.path.insert(0,str(ROOT/'RIFT'))
from high_value_init.v5_candidates import make_candidates, closest
from high_value_init.native_state_audit import NativeAudit, spawn_frozen_cohort


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_compatible(value):
    """Keep strict JSON and retain the location/kind of every non-finite number."""
    nonfinite=[]
    def convert(item,path):
        if isinstance(item,np.ndarray):item=item.tolist()
        if isinstance(item,np.generic):item=item.item()
        if isinstance(item,float) and not math.isfinite(item):
            kind='NaN' if math.isnan(item) else ('Infinity' if item>0 else '-Infinity')
            nonfinite.append({'json_pointer':path,'original_value':kind})
            return None
        if isinstance(item,dict):
            return {k:convert(v,path+'/'+str(k).replace('~','~0').replace('/','~1'))
                    for k,v in item.items()}
        if isinstance(item,(list,tuple)):
            return [convert(v,path+'/'+str(i)) for i,v in enumerate(item)]
        return item
    return convert(value,''),nonfinite


def write_report(output,report):
    clean,nonfinite=json_compatible(report)
    clean['numeric_diagnostics']={'nonfinite_count':len(nonfinite),'nonfinite_values':nonfinite,
        'encoding':'null with explicit JSON pointer and original NaN/Infinity kind',
        'native_boolean_results_rewritten':False}
    # Encode the complete row before writing, so a serializer error cannot
    # leave a half-written JSONL record. Never enable nonstandard JSON NaN.
    output.write(json.dumps(clean,ensure_ascii=False,allow_nan=False)+'\n')
    output.flush()
    return clean


def resume_prefix(directory,jobs,mode,physical_context):
    previous=[json.loads(line) for line in (directory/'jobs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if previous!=jobs:raise ValueError('Resume planned jobs/order differs; refuse to mix experiments')
    log=directory.parent/'logs'/(directory.name+'.log')
    if not log.is_file():raise ValueError('Resume requires original stage command log')
    with log.open(encoding='utf-8') as stream:command=json.loads(stream.readline())
    def option(name,default=None):
        return command[command.index(name)+1] if name in command else default
    if option('--mode')!=mode or option('--physical-context','manifest')!=physical_context:
        raise ValueError('Resume mode/physical context differs')
    rows=[json.loads(line) for line in (directory/'rows.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows)>len(jobs):raise ValueError('Resume row ledger exceeds planned jobs')
    for i,row in enumerate(rows):
        if row.get('source_path')!=jobs[i]['snapshot_path']:
            raise ValueError('Resume row source/order differs at %d'%i)
        if row.get('source_sha256') and jobs[i].get('snapshot_sha256')!=row['source_sha256']:
            raise ValueError('Resume row source SHA differs at %d'%i)
    return rows


def local_state(ego, actor):
    dx,dy = float(actor['x'])-float(ego['x']), float(actor['y'])-float(ego['y'])
    a = math.radians(float(ego['yaw_deg']))
    h = math.radians(float(actor['yaw_deg'])-float(ego['yaw_deg']))
    return [math.cos(a)*dx+math.sin(a)*dy, -math.sin(a)*dx+math.cos(a)*dy,
            math.atan2(math.sin(h),math.cos(h)),float(actor.get('speed_mps',0.)),
            float(actor['length']),float(actor['width']),1.]


def condition_from_snapshot(scene):
    ego = scene['ego']
    def xy(p):
        # Only the XY transform is needed for map-only candidate conditions.
        q = dict(p,yaw_deg=ego['yaw_deg'],length=1.,width=1.)
        return local_state(ego,q)[:2]
    return {'ego_route':{'polyline':[xy(p) for p in scene.get('ego_route',[])]},
            'map':{'lane_polylines':[[xy(p if isinstance(p,dict) else {'x':p[0],'y':p[1]})
                                      for p in l.get('centerline',[])]
                                     for l in scene.get('map_lanes',[])]}}


def overlap(a,b):
    """2D OBB overlap at serialized actor centers; not a 3D/static-mesh test."""
    axes = []
    for p in (a,b):
        u = np.array([math.cos(p[2]),math.sin(p[2])])
        axes.extend([u,np.array([-u[1],u[0]])])
    delta = np.asarray(b[:2])-np.asarray(a[:2])
    for axis in axes:
        ra = a[4]/2*abs(axis@axes[0])+a[5]/2*abs(axis@axes[1])
        rb = b[4]/2*abs(axis@axes[2])+b[5]/2*abs(axis@axes[3])
        if abs(delta@axis) > ra+rb:
            return False
    return True


def actor_report(scene, pose, context, peers, candidates, label=None, fov=64., anchor=None,
                 candidate_inputs_known=True):
    state = local_state(scene['ego'],pose)
    options = []
    for i,c in enumerate(candidates):
        d,s,h,_ = closest(np.asarray(c['polyline']),state[:2])
        error = abs(math.atan2(math.sin(state[2]-h),math.cos(state[2]-h)))
        if d<=1.25 and error<=.38 and s<c['anchor_s']-.25:
            options.append(i)
    anchor_options = None if anchor is None else [i for i in options
        if math.dist(candidates[i]['anchor_xy'],anchor[:2])<=12.]
    ego_overlap = overlap(state,local_state(scene['ego'],scene['ego']))
    context_ids = [p['actor_id'] for p in context
                   if p['actor_id'] != pose['actor_id'] and overlap(state,local_state(scene['ego'],p))]
    peer_ids = [p['actor_id'] for p in peers
                if p['actor_id'] != pose['actor_id'] and overlap(state,local_state(scene['ego'],p))]
    return {'actor_id':pose['actor_id'],'type_id':pose['type_id'],'pose':pose,'local_state':state,
            'forward_fov':bool(0<state[0]<=fov/2 and abs(state[1])<=fov/2),
            'saved_label_available':label is not None,
            'saved_positive':bool(label.get('is_interactive')) if label is not None else None,
            'saved_score':label.get('score') if label is not None else None,
            'saved_interaction_waypoint':label.get('interaction_waypoint') if label is not None else None,
            'saved_target_route':label.get('target_route_to_interaction') if label is not None else None,
            'candidate_geometry_supported':bool(options) if candidate_inputs_known else None,'candidate_indices':options,
            'candidate_anchor_supported':None if anchor_options is None else bool(anchor_options),
            'candidate_anchor_indices':anchor_options,
            'occupancy_2d':{'ego_overlap':ego_overlap,'context_overlap_ids':context_ids,
                            'peer_overlap_ids':peer_ids,
                            'clear':not (ego_overlap or context_ids or peer_ids)},
            'native':None,'road_geometry':None}


def read_job(job, blueprint):
    path = Path(job['snapshot_path'])
    source_bytes = path.read_bytes()
    payload = json.loads(source_bytes)
    scene = payload.get('scene',payload)
    if scene.get('coordinate_frame') != 'carla_world':
        raise ValueError('Requires an explicit carla_world snapshot')
    for p in [scene['ego']]+scene['agents']:
        if not all(k in p for k in ('actor_id','type_id','x','y','z','yaw_deg','length','width')):
            raise ValueError('Snapshot actor lacks world pose/type/size')
        if not all(math.isfinite(float(p[k])) for k in ('x','y','z','yaw_deg','length','width')):
            raise ValueError('Non-finite snapshot pose/size')
        if min(float(p['length']),float(p['width']))<=0:
            raise ValueError('Nonpositive snapshot box')
    ids = [p['actor_id'] for p in [scene['ego']]+scene['agents']]
    if len(set(ids)) != len(ids):
        raise ValueError('Repeated actor IDs in snapshot')
    selected = job.get('target_actor_ids',[])
    if len(selected)!=len(set(selected)):
        raise ValueError('Repeated target IDs')
    if not set(selected).issubset({p['actor_id'] for p in scene['agents']}):
        raise ValueError('Manifest target actor absent from snapshot')
    if 'context_actor_ids' in job:
        context_ids = job['context_actor_ids']
        if len(context_ids)!=len(set(context_ids)) or set(context_ids)&set(selected) or \
                not set(context_ids).issubset({p['actor_id'] for p in scene['agents']}):
            raise ValueError('Manifest context IDs must be unique, present and disjoint from targets')
    if job.get('snapshot_sha256') and hashlib.sha256(source_bytes).hexdigest()!=job['snapshot_sha256']:
        raise ValueError('Snapshot content hash mismatch')
    if 'reference_target_states' in job:
        reference = job['reference_target_states']
        by_id = {p['actor_id']:p for p in scene['agents']}
        if len(reference)!=len(selected):
            raise ValueError('Reference target count differs from selected actor IDs')
        for i,actor_id in enumerate(selected):
            s=local_state(scene['ego'],by_id[actor_id]);gt=reference[i]
            if len(gt)<4 or math.dist(s[:2],gt[:2])>.03 or \
                    abs(math.atan2(math.sin(s[2]-gt[2]),math.cos(s[2]-gt[2])))>.01 or abs(s[3]-gt[3])>.02:
                raise ValueError('Reference target does not match snapshot actor: %s' % actor_id)
    generated = job.get('generated_agent_states')
    if generated is None:
        generated = (payload.get('proposal') or {}).get('generated_agent_states')
    if generated is not None:
        for s in generated:
            if len(s)<6 or not np.isfinite(s[:6]).all() or s[3]<0:
                raise ValueError('Invalid raw generated state')
        if len(generated)!=2:
            raise ValueError('This experiment expects exactly two generated states')
    # The manifest is the place to specify the exact deployed blueprint.
    blueprint = job.get('generated_blueprint',blueprint)
    if generated is not None and blueprint != 'vehicle.tesla.model3':
        raise ValueError('Current generated occupancy contract requires vehicle.tesla.model3')
    return payload,scene,selected,generated,blueprint,hashlib.sha256(source_bytes).hexdigest()


def audit_job(job, args, engine=None):
    payload,scene,selected,generated,blueprint,source_sha = read_job(job,args.blueprint)
    agents = scene['agents']
    context = [p for p in agents if p['actor_id'] not in selected]
    condition_context=context
    if 'context_actor_ids' in job:
        condition_context = [p for p in agents if p['actor_id'] in job['context_actor_ids']]
        if getattr(args,'physical_context','manifest')=='manifest':context=condition_context
    labels = {int(l['actor_id']):l for l in scene.get('rift_labels',[])}
    condition = job.get('candidate_condition') or condition_from_snapshot(scene)
    audit_lanes=None
    if engine:
        if getattr(engine,'progress',None):engine.progress('map_prepare',town=scene['town'])
        engine.prepare(scene['town'])
        if getattr(args,'enrich_map',False):
            if getattr(engine,'progress',None):engine.progress('map_export',town=scene['town'])
            audit_lanes=engine.enriched_lanes(scene)
            if not job.get('candidate_condition') and not condition['map']['lane_polylines']:
                condition=condition_from_snapshot(dict(scene,map_lanes=audit_lanes))
    candidate_inputs_known = (bool(condition['map']['lane_polylines']) and
                              len(condition['ego_route']['polyline'])>=2)
    candidates = make_candidates(condition,fov=args.fov)
    row = {'scene_id':scene['scene_id'],'town':scene['town'],'frame_id':scene.get('frame_id'),
           'source_path':str(Path(job['snapshot_path']).resolve()),'source_sha256':source_sha,
           'episode_key':job.get('episode_key'),'record_index':job.get('record_index'),
           'route_id':scene.get('route_id'),'route_index':scene.get('route_index'),
           'repetition_index':scene.get('repetition_index'),
           'snapshot_map_mode':(scene.get('diagnostics') or {}).get('map_coordinate_mode'),
           'candidate_count':len(candidates),'selected_target_ids':selected,
           'candidate_inputs_known':candidate_inputs_known,
           'candidate_condition_source':'manifest' if job.get('candidate_condition') else 'snapshot',
           'candidate_condition_sha256':hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest(),
           'context_actor_ids':[p['actor_id'] for p in context],
           'conditioning_context_actor_ids':[p['actor_id'] for p in condition_context],
           'physical_context_mode':getattr(args,'physical_context','manifest'),
           'comparison_context':'matched_target_replacement' if selected else 'additive_online',
           'raw_pool':[],'selected_gt':[],'generated':[],'spawn':{},'errors':[]}
    if audit_lanes is not None:
        from high_value_init.v6_conditions import make_condition
        row['audit_map_lanes']=audit_lanes
        row['v6_condition']=make_condition(scene,audit_lanes,condition_context)
        row['audit_map_source']='CARLA OpenDRIVE; %sm sampling; world XYZ/width/directed edges' % args.sampling_resolution
        if not job.get('candidate_condition') and not scene.get('map_lanes'):
            row['candidate_condition_source']='native_map_fallback_NOT_historical_model_input'
    for p in agents:
        label = labels.get(p['actor_id'])
        anchor = (label or {}).get('interaction_waypoint')
        if anchor is not None:
            anchor = local_state(scene['ego'],dict(anchor,length=1.,width=1.,
                                                   yaw_deg=scene['ego']['yaw_deg']))[:2]
        report = actor_report(scene,p,agents,[],candidates,label,args.fov,anchor,candidate_inputs_known)
        row['raw_pool'].append(report)
    agent_by_id = {p['actor_id']:p for p in agents}
    selected_poses = [agent_by_id[i] for i in selected]
    for p in selected_poses:
        label=labels.get(p['actor_id'])
        anchor=(label or {}).get('interaction_waypoint')
        if anchor is not None:
            anchor=local_state(scene['ego'],dict(anchor,length=1.,width=1.,yaw_deg=scene['ego']['yaw_deg']))[:2]
        row['selected_gt'].append(actor_report(scene,p,context,selected_poses,candidates,label,args.fov,
                                                anchor,candidate_inputs_known))
    route = None
    if engine:
        engine.prepare(scene['town'])
        row['carla_map_name'] = engine.world_map.name
        row['carla_opendrive_sha256'] = engine.map_sha256
        try:
            route,error = engine.resolve_route(scene['ego_route'])
            row['ego_route_resolution_max_error_m'] = error
        except Exception as exc:
            row['errors'].append('route_resolution: %s: %s' % (type(exc).__name__,exc))
    generated_poses = []
    for i,state in enumerate(generated or []):
        actor_id = -1-i
        if actor_id in labels or actor_id in {p['actor_id'] for p in agents+[scene['ego']]}:
            raise ValueError('Synthetic actor ID collides with snapshot')
        if engine:
            attempt = next((a for a in (payload.get('result') or {}).get('spawn_attempts',[])
                            if a.get('index')==i and a.get('generated_local_state')==state),None)
            p = engine.generated_pose(scene['ego'],state,actor_id,blueprint,
                                      None if attempt is None else attempt.get('spawn_transform'))
        else:
            e = scene['ego']; a=math.radians(float(e['yaw_deg']))
            p = {'actor_id':actor_id,'type_id':blueprint,
                 'x':e['x']+math.cos(a)*state[0]-math.sin(a)*state[1],
                 'y':e['y']+math.sin(a)*state[0]+math.cos(a)*state[1],
                 'z':e['z'],'yaw_deg':e['yaw_deg']+math.degrees(state[2]),
                 'length':4.624,'width':1.909,'speed_mps':state[3],
                 'z_source':'unknown_placeholder_NOT_for_native_or_spawn'}
        generated_poses.append(p)
    for i,p in enumerate(generated_poses):
        report = actor_report(scene,p,context,generated_poses,candidates,fov=args.fov,
                              candidate_inputs_known=candidate_inputs_known)
        report['raw_generated_local_state'] = generated[i]
        report['occupancy_box_source'] = 'Tesla Model3 size approximation; spawn records actual blueprint box'
        row['generated'].append(report)
    if engine:
        for report in row['raw_pool']+row['generated']:
            try:
                report['road_geometry'] = engine.geometry(report['pose'])
            except Exception as exc:
                report['road_error'] = '%s: %s' % (type(exc).__name__,exc)
                row['errors'].append('actor %s road_query: %s' % (report['actor_id'],report['road_error']))
            if route is not None:
                try:
                    report['native'] = engine.actor(scene,report['pose'],route)
                except Exception as exc:
                    report['native'] = {'evaluated':False,'error':'%s: %s' % (type(exc).__name__,exc)}
                    row['errors'].append('actor %s native: %s' % (report['actor_id'],report['native']['error']))
        native_by_id={r['actor_id']:r for r in row['raw_pool']}
        for report in row['selected_gt']:
            for key in ('native','road_geometry','road_error'):
                if key in native_by_id[report['actor_id']]:
                    report[key]=native_by_id[report['actor_id']][key]
    if row.get('v6_condition'):
        c=row['v6_condition'];lanes={l['lane_uid']:l for l in c['map']['lanes_v6']}
        for report in row['raw_pool']+row['selected_gt']+row['generated']:
            supported=[];state=report['local_state']
            for approach in c['approaches']:
                lane=lanes[approach['lane_uid']];line=np.asarray(lane['centerline'],dtype=float)[:,:2]
                d,_,h,_=closest(line,state[:2])
                k=int(np.argmin(np.linalg.norm(line-np.asarray(state[:2]),axis=1)))
                if d<=lane['width_m'][k]/2. and abs(math.atan2(math.sin(state[2]-h),math.cos(state[2]-h)))<=.38:
                    supported.append(lane['lane_uid'])
            report['v6_approach_geometry_supported']=bool(supported) if c['topology_route_roots_known'] else None
            report['v6_supporting_approach_ids']=supported
    if args.mode in ('spawn','physics'):
        if getattr(engine,'progress',None):engine.progress('spawn_prepare',town=scene['town'])
        world = engine.world
        if list(world.get_actors().filter('vehicle.*')):
            raise ValueError('Spawn replay requires an empty dedicated CARLA world; existing vehicles are not deleted')
        engine.ensure_geometry()  # Cache level boxes before any fleet is spawned.
        settings = world.get_settings()
        old_sync,old_delta = settings.synchronous_mode,settings.fixed_delta_seconds
        try:
            settings.synchronous_mode=True; settings.fixed_delta_seconds=.05
            world.apply_settings(settings)
            start = world.get_snapshot().frame
            if selected:
                groups=[('selected_gt',context,selected_poses)]
            else:
                groups=[('raw_fleet',[],agents)]
            if generated is not None:
                groups.append(('generated',context,generated_poses))
            for name,ctx,targets in groups:
                row['spawn'][name] = spawn_frozen_cohort(world,engine.carla,scene['ego'],ctx,targets,
                    engine=engine,probe_ticks=args.probe_ticks if args.mode=='physics' else 0,scene=scene)
                if row['spawn'][name]['cleanup_errors']:
                    row['errors'].append('Owned actor cleanup failed; stop replay')
                    row['abort_remaining_jobs']=True
                    break
                s=row['spawn'][name]
                row['errors'].extend('spawn geometry: '+e for e in s.get('geometry_errors',[]))
                if (s.get('dynamic_probe') or {}).get('errors'):
                    row['errors'].extend(s['dynamic_probe']['errors'])
                    row['abort_remaining_jobs']=True
                    break
            row['spawn_frames_advanced'] = world.get_snapshot().frame-start
            expected=sum((s.get('dynamic_probe') or {}).get('ticks_observed',0) for s in row['spawn'].values())
            row['spawn_expected_frame_advances']=expected
            if row['spawn_frames_advanced'] != expected:
                row['errors'].append('World advanced during frozen replay; external client or unsupported server behavior')
                row['abort_remaining_jobs']=True
        finally:
            settings.synchronous_mode=old_sync; settings.fixed_delta_seconds=old_delta
            world.apply_settings(settings)
    return row


def summarize(rows, mode):
    groups = defaultdict(Counter)
    towns = defaultdict(lambda:defaultdict(Counter))
    errors = []
    for row in rows:
        errors.extend(row.get('errors',[]))
        for name in ('raw_pool','selected_gt','generated'):
            actors = row.get(name,[])
            if not actors:
                continue
            counts=groups[name]; per_town=towns[str(row.get('town'))][name]
            for c in (counts,per_town):
                c['scenes']+=1; c['actors']+=len(actors)
                front=[a for a in actors if a['forward_fov']]
                c['forward_actors']+=len(front)
                positive=[a for a in front if a['saved_positive'] is True]
                c['saved_forward_positive_actors']+=len(positive)
                c['saved_forward_label_unknown_actors']+=sum(a['saved_positive'] is None for a in front)
                c['saved_positive_candidate_evaluated_actors']+=sum(a['candidate_geometry_supported'] is not None for a in positive)
                c['saved_positive_candidate_supported_actors']+=sum(a['candidate_geometry_supported'] is True for a in positive)
                c['scenes_with_saved_forward_positive']+=bool(positive)
                c['forward_occupancy_clear_actors']+=sum(a['occupancy_2d']['clear'] for a in front)
                evaluated=[a for a in front if (a.get('native') or {}).get('evaluated')]
                native=[a for a in evaluated if a['native']['native_candidate']]
                c['native_forward_evaluated_actors']+=len(evaluated)
                c['native_forward_unknown_actors']+=len(front)-len(evaluated)
                c['native_forward_positive_actors']+=len(native)
                c['native_forward_positive_occupancy_clear_actors']+=sum(a['occupancy_2d']['clear'] for a in native)
                c['native_positive_candidate_evaluated_actors']+=sum(a['candidate_geometry_supported'] is not None for a in native)
                c['native_positive_candidate_supported_actors']+=sum(a['candidate_geometry_supported'] is True for a in native)
                c['native_positive_v6_approach_evaluated_actors']+=sum(a.get('v6_approach_geometry_supported') is not None for a in native)
                c['native_positive_v6_approach_supported_actors']+=sum(a.get('v6_approach_geometry_supported') is True for a in native)
                c['scenes_with_native_forward_positive']+=bool(native)
                c['scenes_with_at_least_two_native_forward_positive']+=len(native)>=2
                for a in front:
                    geometry=a.get('road_geometry') or {}
                    c['center_road_query_known_actors']+='center_driving' in geometry
                    c['center_driving_actors']+=geometry.get('center_driving') is True
                    c['preview_road_query_known_actors']+='preview_driving' in geometry
                    c['preview_driving_actors']+=geometry.get('preview_driving') is True
                    c['preview_none_false_disagreements']+=geometry.get('preview_none_false_agree') is False
                    info=a.get('native') or {}
                    for reason,value in (info.get('candidate_debug') or {}).get('rejection_counts',{}).items():
                        c['native_reject_'+reason]+=value
                    c['saved_native_label_disagreements']+=bool(info.get('evaluated') and a['saved_positive'] is not None
                                                             and a['saved_positive']!=info['native_candidate'])
                    c['saved_config_mismatch_actors']+=bool(info.get('saved_config_mismatches'))
        for name,s in row.get('spawn',{}).items():
            c=groups['spawn_'+name];c['scenes']+=1
            for k in ('context_complete','targets_all_accepted','whole_scene_accepted'):
                c[k]+=bool(s[k])
            c['cleanup_errors']+=len(s['cleanup_errors'])
            c['geometry_errors']+=len(s.get('geometry_errors',[]))
            c['target_spawn_requests']+=sum(a['role']=='target' for a in s['attempts'])
            c['target_spawn_accepted']+=sum(a['role']=='target' and a['accepted'] for a in s['attempts'])
            target_ids={a['source_actor_id'] for a in s['attempts'] if a['role']=='target'}
            for source,g in s.get('geometry_3d',{}).items():
                if source not in target_ids:continue
                c['target_actual_box_evaluated']+=1
                c['target_whole_sampled_road_contained']+=bool(g['whole_road']['whole_box_in_sampled_driving_polygon'])
                c['target_peer_3d_clear']+=not g['peer_3d_overlap_ids']
                c['target_static_query_unknown']+=bool(g['static_query_errors'])
                c['target_static_box_clear']+=not g['static_query_errors'] and not g['static_3d_box_overlaps']
            d=s.get('dynamic_probe')
            if d:
                c['physics_probe_scenes']+=1;c['physics_probe_error_scenes']+=bool(d['errors'])
                for m in d['measurements']:
                    if m['source_actor_id'] not in target_ids:continue
                    c['target_physics_observed']+=bool(m['observed_frames'])
                    c['target_collision_events']+=m['collision_event_count']
                    c['target_first_tick_velocity_error_le_0p5']+=m['first_tick_velocity_error_mps'] is not None and m['first_tick_velocity_error_mps']<=.5
                    c['target_tail_z_range_le_0p1']+=m['tail_z_range_m'] is not None and m['tail_z_range_m']<=.1
            if name in ('selected_gt','generated'):
                outcomes=[]
                attempts={a['source_actor_id']:a for a in s['attempts'] if a['role']=='target'}
                for actor in row.get(name,[]):
                    aid=actor['actor_id'];g=s.get('geometry_3d',{}).get(aid)
                    native=actor.get('native') or {};attempt=attempts.get(aid)
                    if not s['context_complete']:
                        outcomes.append(None)
                    elif not actor['forward_fov'] or (attempt is not None and not attempt['accepted']) or \
                            (native.get('evaluated') and not native['native_candidate']):
                        outcomes.append(False)
                    elif g is None or not native.get('evaluated') or attempt is None:
                        outcomes.append(None)
                    else:
                        outcomes.append(g['whole_road']['whole_box_in_sampled_driving_polygon'] and
                                        not g['peer_3d_overlap_ids'])
                c['joint_front_native_polygon_peer_spawn_true_actors']+=sum(v is True for v in outcomes)
                c['joint_front_native_polygon_peer_spawn_unknown_actors']+=sum(v is None for v in outcomes)
                if any(v is True for v in outcomes):c['joint_at_least_one_true_scenes']+=1
                elif any(v is None for v in outcomes):c['joint_at_least_one_unknown_scenes']+=1
                if len(outcomes)==2 and all(v is True for v in outcomes):c['joint_both_true_scenes']+=1
                elif not any(v is False for v in outcomes) and any(v is None for v in outcomes):c['joint_both_unknown_scenes']+=1
    result={}
    for name,c in groups.items():
        r={'counts':dict(c),'planned_scene_count':len(rows),
           'missing_cohort_scene_count':len(rows)-c['scenes']}
        if not name.startswith('spawn_'):
            r['forward_actor_fraction']=c['forward_actors']/c['actors']
            r['at_least_one_front_native_scene_rate']=(
                c['scenes_with_native_forward_positive']/c['scenes']
                if mode!='snapshot' and c['native_forward_unknown_actors']==0 and c['scenes']==len(rows) else None)
            n=c['saved_positive_candidate_evaluated_actors']
            r['saved_positive_candidate_coverage']=c['saved_positive_candidate_supported_actors']/n if n else None
            n=c['native_forward_evaluated_actors']
            r['native_positive_rate_given_evaluated']=c['native_forward_positive_actors']/n if n else None
            n=c['native_positive_candidate_evaluated_actors']
            r['native_positive_candidate_coverage']=c['native_positive_candidate_supported_actors']/n if n else None
            n=c['native_positive_v6_approach_evaluated_actors']
            r['native_positive_v6_approach_coverage']=c['native_positive_v6_approach_supported_actors']/n if n else None
        elif name in ('spawn_selected_gt','spawn_generated'):
            r['joint_at_least_one_scene_rate']=(c['joint_at_least_one_true_scenes']/c['scenes']
                                               if not c['joint_at_least_one_unknown_scenes'] and c['scenes']==len(rows) else None)
            r['joint_both_scene_rate']=(c['joint_both_true_scenes']/c['scenes']
                                       if not c['joint_both_unknown_scenes'] and c['scenes']==len(rows) else None)
            r['joint_definition']='context complete + raw spawn accepted + forward FOV + initial native candidate + actual-box sampled road containment + no peer 3D overlap; static mesh excluded'
        result[name]=r
    numeric_jobs=sum(bool((r.get('numeric_diagnostics') or {}).get('nonfinite_count')) for r in rows)
    numeric_count=sum((r.get('numeric_diagnostics') or {}).get('nonfinite_count',0) for r in rows)
    numeric_scope=({'full_evaluation_eligible':False} if numeric_jobs else {})
    return {'mode':mode,'planned_jobs':len(rows),'job_errors':errors,'groups':result,
            **numeric_scope,
            'numeric_diagnostics':{'jobs_with_nonfinite_numbers':numeric_jobs,
                                   'nonfinite_serialized_value_count':numeric_count,
                                   'nonfinite_numbers_are_unknown_not_zero':True,
                                   'native_boolean_results_rewritten':False},
            'by_town':{t:{g:dict(c) for g,c in v.items()} for t,v in sorted(towns.items())},
            'initial_candidate_evidence':mode in ('native','spawn','physics'),
            'native_cbv_activation_evaluated':False,
            'whole_obb_road_containment_checked':any(s.get('geometry_3d') for r in rows for s in r.get('spawn',{}).values()),
            'road_containment_method':'actual blueprint footprint at raw requested pose against sampled OpenDRIVE lane ribbons',
            'exact_unreal_road_mesh_containment_checked':False,
            'actual_box_sampled_road_polygon_evaluated':any(s.get('geometry_3d') for r in rows for s in r.get('spawn',{}).values()),
            'physics_probe_evaluated':mode=='physics',
            'native_error_jobs':sum(bool(r.get('errors')) for r in rows),
            'map_coordinate_mode_counts':dict(Counter(str(r.get('snapshot_map_mode','unknown')) for r in rows)),
            'candidate_inputs_missing_jobs':sum(not r.get('candidate_inputs_known',False) for r in rows),
            'jobs_without_two_generated_states':sum(len(r.get('generated',[]))!=2 for r in rows),
            'limits':['Snapshot pools include serialized actors only, not unrecorded traffic.',
                      'Snapshot 2D OBB is approximate; spawn/physics add actual 3D boxes, sampled road ribbons and conservative level boxes.',
                      'Native uses unchanged base candidate rules; no attention ranking, live sensors or CBV activation.',
                      'Non-finite numeric observations are null with explicit field/kind evidence; affected output remains diagnostic, not full acceptance.',
                      'Spawn is frozen acceptance; physics is a separate uncontrolled finite probe, not historical episode replay.']}


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-dir',type=Path)
    inputs.add_argument('--manifest',type=Path,help='JSONL: snapshot_path, optional target_actor_ids/generated_agent_states/generated_blueprint')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume-from',type=Path,help='Copy a verified completed row prefix into a NEW output directory; old evidence unchanged')
    p.add_argument('--mode',choices=('snapshot','native','spawn','physics'),default='snapshot')
    p.add_argument('--probe-ticks',type=int,default=20,help='physics mode only; no control/pose correction')
    p.add_argument('--enrich-map',action='store_true',help='Export live-map directed lanes/width/XYZ; fill absent raw candidate map')
    p.add_argument('--max-records',type=int,default=0,help='0=all; otherwise deterministic input prefix')
    p.add_argument('--group-by-town',action='store_true',help='Reorder the already selected jobs to reduce map reloads; selection unchanged')
    p.add_argument('--physical-context',choices=('manifest','full_snapshot'),default='manifest',
                   help='Physical occupancy/replay context; model input condition remains unchanged')
    p.add_argument('--fov',type=float,default=64.)
    p.add_argument('--blueprint',choices=['vehicle.tesla.model3'],default='vehicle.tesla.model3')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=31000)
    p.add_argument('--timeout',type=float,default=120.)
    p.add_argument('--recognition-config',type=Path,default=ROOT/'RIFT/rift/cbv/recognition/config/attention.yaml')
    p.add_argument('--recognition-mode',choices=('train','eval'),default='eval')
    p.add_argument('--sampling-resolution',type=float,default=1.)
    p.add_argument('--load-maps',action='store_true',help='May replace world; use a dedicated audit server')
    return p.parse_args()


def main():
    a=arguments()
    if a.max_records<0 or a.probe_ticks<=0:
        raise ValueError('Require max-records >= 0 and probe-ticks > 0')
    if a.fov!=64.:
        raise ValueError('V5 candidate contract uses FOV64; other domains require separate experiment')
    if a.output_dir.exists():
        raise FileExistsError('Refuse to overwrite experiment directory')
    if a.input_dir:
        try:
            a.output_dir.resolve().relative_to(a.input_dir.resolve())
        except ValueError:
            pass
        else:
            raise ValueError('Output must be outside snapshot input tree')
        jobs=[{'snapshot_path':str(p.resolve())} for p in snapshot_paths(a.input_dir)]
    else:
        jobs=manifest_jobs(a.manifest)
        for j in jobs:
            path=Path(j['snapshot_path'])
            j['snapshot_path']=str((path if path.is_absolute() else a.manifest.parent/path).resolve())
    if a.max_records>0:jobs=jobs[:a.max_records]
    if not jobs:raise ValueError('No input snapshot jobs')
    if a.group_by_town:
        jobs=sorted(jobs,key=lambda j:(str(j.get('town','')),str(j.get('episode_key','')),j.get('record_index',-1)))
    previous=resume_prefix(a.resume_from,jobs,a.mode,a.physical_context) if a.resume_from else []
    engine=None;meta={'fov':a.fov,'generated_blueprint':a.blueprint,
                      'manifest_sha256':sha(a.manifest) if a.manifest else None,
                      'runner_source_sha256':sha(Path(__file__)),
                      'geometry_source_sha256':sha(ROOT/'scenario-dreamer/high_value_init/audit_physics_geometry.py'),
                      'v6_condition_source_sha256':sha(ROOT/'scenario-dreamer/high_value_init/v6_conditions.py'),
                      'adapter_source_sha256':sha(ROOT/'scenario-dreamer/high_value_init/native_state_audit.py')}
    meta['resumed_rows']=len(previous)
    meta['resumed_from']=str(a.resume_from) if a.resume_from else None
    if a.manifest:
        sidecar=Path(str(a.manifest)+'.audit.json')
        if sidecar.exists():
            evidence=json.loads(sidecar.read_text(encoding='utf-8'))
            meta['pairing_scope']={k:evidence[k] for k in ('evaluation_scope','full_evaluation_eligible',
                'original_requested_records','unique_pair_records','excluded_original_indices',
                'source_pair_report_sha256') if k in evidence}
    if a.mode!='snapshot':
        from high_value_init.carla_audit_environment import configure_carla_pythonapi
        meta['carla_pythonapi_environment']=configure_carla_pythonapi(ROOT,validate=True)
        import carla
        import yaml
        config=yaml.safe_load(a.recognition_config.read_text(encoding='utf-8'))
        client=carla.Client(a.host,a.port);client.set_timeout(a.timeout)
        engine=NativeAudit(client,config,a.recognition_mode,a.sampling_resolution,a.load_maps)
        meta.update({'recognition_config':str(a.recognition_config),'config_sha256':sha(a.recognition_config),
              'recognition_parameters':config[a.recognition_mode],
              'sampling_resolution':a.sampling_resolution,'carla_server_version':client.get_server_version(),
              'native_source_sha256':sha(ROOT/'RIFT/rift/cbv/recognition/base_cbv.py'),
              'grp_source_sha256':sha(ROOT/'RIFT/rift/scenario/tools/global_route_planner.py')})
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'jobs.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs),encoding='utf-8')
    reports=list(previous);aborted=any(r.get('abort_remaining_jobs',False) for r in previous)
    with (a.output_dir/'rows.jsonl').open('w',encoding='utf-8') as output:
        for report in previous:output.write(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n')
        output.flush()
        print('Verified completed prefix: %d/%d; new output=%s'%(len(previous),len(jobs),a.output_dir),flush=True)
        for i,job in enumerate(jobs[len(previous):],start=len(previous)):
            started=time.monotonic()
            def progress(phase,**details):
                current={'record_index':job.get('record_index'),'sequence':i+1,'total':len(jobs),
                         'mode':a.mode,'phase':phase,'elapsed_seconds':time.monotonic()-started,'details':details}
                (a.output_dir/'current_job.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
                print('Phase %s record=%s %s'%(phase,job.get('record_index'),json.dumps(details)),flush=True)
            if engine is not None:engine.progress=progress
            progress('audit_job_start')
            print('Processing %d/%d mode=%s record_index=%s' %
                  (i+1,len(jobs),a.mode,job.get('record_index')),flush=True)
            faulthandler.dump_traceback_later(60,repeat=True)
            try:
                report=(dict(source_path=job.get('snapshot_path'),errors=['Skipped after replay isolation/cleanup failure'])
                        if aborted else audit_job(job,a,engine))
            except Exception as exc:
                report={'source_path':job.get('snapshot_path'),'errors':['%s: %s' % (type(exc).__name__,exc)]}
            finally:
                faulthandler.cancel_dump_traceback_later()
            aborted=aborted or report.get('abort_remaining_jobs',False)
            report=write_report(output,report)
            reports.append(report)
            print('Finished %d/%d elapsed=%.1fs errors=%d' %
                  (i+1,len(jobs),time.monotonic()-started,len(report.get('errors',[]))),flush=True)
            numeric=report['numeric_diagnostics']
            if numeric['nonfinite_count']:
                print('Non-finite numeric observations in record %d: %s' %
                      (i,json.dumps(numeric['nonfinite_values'][:3])),flush=True)
            if (i+1)%50==0:print('processed %d/%d' % (i+1,len(jobs)),flush=True)
    summary=summarize(reports,a.mode)
    summary.update(meta)
    summary['input_selection']='all' if a.max_records==0 else 'deterministic_prefix'
    summary['execution_order']='grouped_by_town_after_selection' if a.group_by_town else 'input_order'
    summary['physical_context_mode']=a.physical_context
    if (meta.get('pairing_scope') or {}).get('full_evaluation_eligible') is False:
        summary['full_evaluation_eligible']=False
        summary['input_selection']='unambiguous_diagnostic_subset'
    summary['schema_version']='hv_snapshot_experiment_v2'
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('mode','planned_jobs','native_error_jobs',
                                           'candidate_inputs_missing_jobs','map_coordinate_mode_counts',
                                           'numeric_diagnostics')},indent=2))
    print('Full ledgers and summary:',a.output_dir)
    if summary['native_error_jobs']:
        raise SystemExit('Experiment has job errors/unknown native evidence; inspect rows.jsonl')


if __name__=='__main__':main()
