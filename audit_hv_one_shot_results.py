#!/usr/bin/env python3
"""Audit the strict single-shot HV contract from CARLA/RIFT dump files.

Does not change generated states, CARLA, RIFT, or run any simulator actions.
"""
import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path


def wrapped_angle_degrees(v):
    return ((v + 180.0) % 360.0) - 180.0


def carla_float32(value):
    """Round a Python float64 to CARLA/Unreal's float32 transform precision."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('non-finite CARLA transform component')
    return struct.unpack('<f', struct.pack('<f', value))[0]


def audit(dump_dir: Path, simulator_exit_code=None):
    errors, warnings, scenes = [], [], []
    dump_files = sorted(dump_dir.glob('*_online_generation.json'))
    if not dump_files:
        errors.append('No online generation dump files found: cannot verify a one-shot request')
    for path in dump_files:
        with path.open() as f:
            d = json.load(f)
        result = d.get('result') or {}
        prop = d.get('proposal')
        scene = d.get('scene') or {}
        sid = str(result.get('scene_id', path.name))
        attempted = bool(result.get('generator_called'))
        row = {'scene_id':sid,'dump_file':path.name,'attempted':attempted,
               'spawn_success':bool(result.get('success')),
               'failure_stage':result.get('failure_stage'),
               'skipped':bool(result.get('skipped')),
               'spawned_ids':[int(i) for i in result.get('spawned_actor_ids',[])],
               'green_ids':[]}
        if attempted:
            tries = result.get('generation_attempts', [])
            if result.get('generation_max_attempts') != 1 or len(tries) != 1:
                errors.append(f'{sid}: must report exactly one generation attempt')
            if result.get('rift_audit_status') != 'deferred_to_normal_cbvs_recog':
                errors.append(f'{sid}: unexpected eager RIFT audit')
            states = (prop or {}).get('generated_agent_states',[])
            if not isinstance(states,list) or len(states)!=2:
                if result.get('failure_stage') not in {'generator','proposal_validation'}:
                    errors.append(f'{sid}: proposal missing 2 states after spawn')
            attempts = result.get('spawn_attempts', [])
            if row['spawn_success']:
                if len(row['spawned_ids'])!=2 or len(attempts)!=2:
                    errors.append(f'{sid}: reported successful spawn without 2 raw actor spawn requests')
            if isinstance(states,list) and len(states)==2:
                ego = scene.get('ego') or {}
                if all(k in ego for k in ('x','y','yaw_deg')):
                    yaw=math.radians(float(ego['yaw_deg']))
                    for attempt in attempts:
                        i=int(attempt['index'])
                        if not 0 <= i < 2:
                            errors.append(f'{sid}: impossible spawn index {i}')
                            continue
                        state=states[i]
                        if len(state)<4:
                            errors.append(f'{sid}: insufficient raw state')
                            continue
                        expected_x=float(ego['x']) + math.cos(yaw)*float(state[0])-math.sin(yaw)*float(state[1])
                        expected_y=float(ego['y']) + math.sin(yaw)*float(state[0])+math.cos(yaw)*float(state[1])
                        expected_yaw=math.degrees(yaw+float(state[2]))
                        world=attempt.get('raw_world_transform') or {}
                        if not all(k in world for k in ('x','y','yaw_deg')):
                            errors.append(f'{sid}: missing raw_world_transform')
                            continue
                        dx=math.hypot(float(world['x'])-expected_x,float(world['y'])-expected_y)
                        dh=abs(wrapped_angle_degrees(float(world['yaw_deg'])-expected_yaw))
                        if dx>1e-5 or dh>1e-4:
                            errors.append(f'{sid}: generated state {i} was rewritten by {dx:.6g}m/{dh:.6g}deg')
                        if attempt.get('generated_local_state')!=state:
                            errors.append(f'{sid}: generated state {i} differs from spawn request')
                        # The CARLA spawn command must contain the raw world
                        # x/y/yaw, not just a correct-looking audit field.
                        spawn=attempt.get('spawn_transform') or {}
                        if not all(k in spawn for k in ('x','y','yaw_deg')):
                            errors.append(f'{sid}: missing actual spawn request transform {i}')
                        else:
                            # CARLA Python bindings store Location/Rotation in
                            # float32, while the raw local->world computation
                            # is Python float64.  Compare the *actual command*
                            # against float32(raw), not raw float64 itself.
                            # Otherwise an unchanged 2 km world position can
                            # cause a false alarm of about 0.1 mm or more.
                            try:
                                expected_cmd_x=carla_float32(world['x'])
                                expected_cmd_y=carla_float32(world['y'])
                                expected_cmd_yaw=carla_float32(world['yaw_deg'])
                                command_dxy=math.hypot(float(spawn['x'])-expected_cmd_x,
                                                       float(spawn['y'])-expected_cmd_y)
                                command_dyaw=abs(wrapped_angle_degrees(
                                    float(spawn['yaw_deg'])-expected_cmd_yaw))
                                # These tolerances cover JSON serialization
                                # only, not model-output postprocessing.
                                if not math.isfinite(command_dxy) or not math.isfinite(command_dyaw) or \
                                        command_dxy>2e-6 or command_dyaw>1e-5:
                                    errors.append(f'{sid}: spawn command {i} differs from float32(raw) '
                                                  f'by {command_dxy:.6g}m/{command_dyaw:.6g}deg')
                            except (OverflowError, ValueError, TypeError, KeyError) as exc:
                                errors.append(f'{sid}: invalid spawn transform {i}: {exc}')
                        # CARLA may return a stale get_transform() before its
                        # first natural simulation tick.  These readings are
                        # not verified acceptance states; the actual strict
                        # contract is whether the spawn request was unchanged.
                        if float(attempt.get('spawn_acceptance_xy_error_m',0.))>0.25 or \
                                float(attempt.get('spawn_acceptance_yaw_error_deg',0.))>2.0:
                            warnings.append(f'{sid}: pre-tick CARLA pose unconfirmed for actor index {i} '
                                            '(do not interpret as state rewriting)')
        scenes.append(row)
    green_events=0
    events_by_actor=Counter()
    lifecycle_path=dump_dir/'lifecycle.jsonl'
    if lifecycle_path.exists():
        for lineno, line in enumerate(lifecycle_path.read_text().splitlines(),1):
            if not line.strip():continue
            try:e=json.loads(line)
            except Exception as ex:
                errors.append(f'lifecycle.jsonl:{lineno}: {ex}')
                continue
            if e.get('event')=='rift_cbv_activated':
                key=(str(e.get('scene_id')), int(e['actor_id']))
                events_by_actor[key]+=1
    for row in scenes:
        row['green_ids']=[x for x in row['spawned_ids']
                          if (row['scene_id'],x) in events_by_actor]
        row['at_least_one_green']=bool(row['green_ids'])
        green_events+=int(row['at_least_one_green'] and row['attempted'])
    attempted_rows=[r for r in scenes if r['attempted']]
    eligible_rows=[r for r in scenes if not r['skipped']]
    spawn_rows=[r for r in eligible_rows if r['spawn_success']]
    completed = simulator_exit_code == 0 if simulator_exit_code is not None else None
    return {
        'schema_version':'hv_strict_one_shot_audit_v2',
        'simulator_exit_code':simulator_exit_code,
        'simulator_completed':completed,
        'dump_dir':str(dump_dir),
        'num_dump_files':len(dump_files),
        'num_eligible_generation_requests':len(eligible_rows),
        'num_generator_called':len(attempted_rows),
        'num_skipped_pre_generation':sum(r['skipped'] for r in scenes),
        'num_both_spawned':len(spawn_rows),
        'num_at_least_one_native_green':green_events,
        # Primary end-to-end denominator also counts import/model-load failures.
        # A crash before normal CBVs_recog() makes green-rate unknown,
        # not zero.  Only a completed evaluation has a meaningful rate.
        'native_green_rate_per_eligible_request':green_events/len(eligible_rows) if completed and eligible_rows else None,
        'native_green_rate_per_generator_call':green_events/len(attempted_rows) if completed and attempted_rows else None,
        'native_green_rate_per_both_spawned':green_events/len(spawn_rows) if completed and spawn_rows else None,
        'failure_stages':dict(Counter(r['failure_stage'] for r in eligible_rows if not r['spawn_success'])),
        'contract_errors':errors,
        'diagnostic_warnings':warnings,
        'scenes':scenes,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dump_dir',type=Path,required=True)
    p.add_argument('--output_json',type=Path)
    p.add_argument('--min_native_green_rate',type=float,default=None)
    p.add_argument('--simulator-exit-code',type=int,default=None,
                   help='Exit code from the actual CARLA runner; required to report a green rate.')
    args=p.parse_args()
    result=audit(args.dump_dir, args.simulator_exit_code)
    summary={k:v for k,v in result.items() if k!='scenes'}
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    if result['contract_errors']:
        raise SystemExit('FAIL: strict single-shot contract violated')
    if args.min_native_green_rate is not None:
        rate=result['native_green_rate_per_eligible_request']
        if rate is None or rate < args.min_native_green_rate:
            raise SystemExit(f'FAIL: native green rate {rate} < {args.min_native_green_rate}')

if __name__=='__main__':main()
