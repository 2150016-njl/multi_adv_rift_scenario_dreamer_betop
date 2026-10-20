#!/usr/bin/env python3
"""P1 real-CARLA supervisor: unchanged frozen auditor, one record per process.

A science failure marks its own record FAIL and does not suppress later records.
Safety/transport failures stop all further operations. All 95 source rows stay fixed.
Python 3.8+ stdlib; no synthetic PASS, no retries/pose changes.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

FROZEN_SHA='b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb'
MANIFEST_SHA='92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
MAP_SHA='9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'
CAPSULE='p1_frozen_capsule_world_transition_depcomplete_v2_20261019'

class EvidenceError(RuntimeError): pass

def need(cond, why):
    if not cond: raise EvidenceError(why)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def readj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def writex(p,obj):
    p=Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')

def input_contract(root):
    root=Path(root).resolve();base=root/'data/hv_init/v7'
    manifest=base/'manifests/p1_calibration95.jsonl'
    mmap=base/'manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    cap=base/CAPSULE
    need(sha(manifest)==MANIFEST_SHA,'P1 manifest hash changed')
    need(sha(mmap)==MAP_SHA,'map manifest hash changed')
    need(sha(cap/'hv_v8/p1_native_audit.py')==FROZEN_SHA,'frozen auditor hash changed')
    prov=readj(cap/'p1_source_provenance.json')
    need(prov.get('dependency_complete_capsule') is True,'capsule not dependency complete')
    for fn,key in [('p1_native_audit.py','dependency_auditor_sha256'),('p1_world_guard.py','dependency_guard_sha256'),('common.py','dependency_common_sha256')]:
        need(sha(cap/'hv_v8'/fn)==prov[key],'dependency hash changed: '+fn)
    need((cap/'hv_v8/__init__.py').is_file(),'missing package init')
    cfg=root/'RIFT/rift/cbv/recognition/config/attention.yaml'
    need(cfg.is_file(),'missing original RIFT attention config')
    egg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
    need(egg.is_file(),'pinned CARLA 0.9.15 py3.7 egg missing')
    rows=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    need(len(rows)==95,'expected exactly fixed 95 input records')
    indices=set();paths=set();scenes=[]
    for ordinal,row in enumerate(rows):
        idx=int(row['record_index']);p=Path(row['snapshot_path']).resolve()
        need(idx not in indices,'duplicate record_index')
        need(str(p) not in paths,'duplicate snapshot path')
        indices.add(idx);paths.add(str(p))
        need(p.is_file(),'missing snapshot '+str(p))
        snapsha=sha(p)
        if row.get('snapshot_sha256'):need(row['snapshot_sha256']==snapsha,'snapshot hash mismatch '+str(idx))
        scene=readj(p)
        need(isinstance(scene.get('ego_route'),list) and len(scene['ego_route'])>0,'empty route '+str(idx))
        town=scene.get('town')
        need(town in ['Town04','Town11','Town12','Town13','Town15'],'unknown town '+str(town))
        scenes.append({'ordinal':ordinal,'record_index':idx,'town':town,'scene_id_manifest':row.get('scene_id'),
                       'scene_id_snapshot':scene.get('scene_id'),'snapshot_sha256':snapsha,'snapshot_path':str(p),
                       'scene_id_equal':row.get('scene_id')==scene.get('scene_id')})
    return {'root':root,'base':base,'cap':cap,'cfg':cfg,'egg':egg,'manifest':manifest,'map_manifest':mmap,'rows':rows,'scenes':scenes}

def process_env(c):
    original=os.environ.get('PYTHONPATH','').split(os.pathsep)
    original=[x for x in original if x and 'carla-0.9.15-' not in x and 'p1_frozen_capsule_' not in x]
    paths=[str(c['cap']),str(c['egg']),str(c['root']/'carla/PythonAPI/carla'),str(c['root']/'RIFT'),str(c['root'])]+original
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(dict.fromkeys(paths))
    return env

def probe_modules(py,env,c):
    script=('import carla,importlib,pathlib,sys;'
        'from importlib import import_module;'
        'm=import_module("hv_v8.p1_native_audit");'
        'assert pathlib.Path(m.__file__).resolve()==pathlib.Path(sys.argv[1]).resolve();'
        'assert pathlib.Path(carla.__file__).resolve().as_posix().startswith(sys.argv[2]+"/");'
        'print("P1_PY="+sys.version.split()[0]);print("P1_AUDITOR="+str(m.__file__));print("P1_CARLA="+str(carla.__file__))')
    p=subprocess.run([py,'-c',script,str(c['cap']/'hv_v8/p1_native_audit.py'),str(c['egg'])],
      cwd=str(c['cap']),env=env,capture_output=True,text=True,timeout=120)
    need(p.returncode==0,'frozen module import failed: '+(p.stderr+p.stdout)[-2000:])
    h=subprocess.run([py,'-m','hv_v8.p1_native_audit','--help'],cwd=str(c['cap']),env=env,capture_output=True,text=True,timeout=120)
    need(h.returncode==0 and '--manifest' in h.stdout,'real module entrypoint unavailable: '+h.stderr[-2000:])
    return p.stdout.strip()

_ACTOR_PREFLIGHT='''import json,carla,sys
c=carla.Client(sys.argv[1],int(sys.argv[2]));c.set_timeout(25)
w=c.get_world();a=w.get_actors()
print(json.dumps({'server':c.get_server_version(),'client':c.get_client_version(),'map':w.get_map().name,'vehicles':len(a.filter('vehicle.*')),'walkers':len(a.filter('walker.pedestrian.*'))}))
'''

def server_safe(py,env,c,host,port):
    p=subprocess.run([py,'-c',_ACTOR_PREFLIGHT,host,str(port)],cwd=str(c['cap']),env=env,
                     capture_output=True,text=True,timeout=75)
    need(p.returncode==0,'CARLA RPC preflight failed: '+p.stderr[-2000:])
    data=json.loads(p.stdout.strip().splitlines()[-1])
    need(data['server']=='0.9.15' and data['client']=='0.9.15','CARLA version changed')
    need(data['vehicles']==0 and data['walkers']==0,'nonempty CARLA world: '+json.dumps(data))
    return data

def failure_class(log):
    if re.search(r'ego_route\[\d+\] exact lane identity mismatch',log):return 'route_exact_lane_identity_mismatch'
    if re.search(r'ego_route\[\d+\] exact get_waypoint returned None',log):return 'route_exact_lookup_missing'
    if 'MAP_IDENTITY_FAIL' in log or 'WORLD_' in log or 'episode' in log.lower() and 'FAIL' in log:return 'world_or_map_identity_failure'
    if 'get_waypoint_xodr returned None' in log:return 'xodr_boundary_or_source_failure'
    if 'P1_FAIL' in log:return 'other_scientific_or_contract_failure'
    if 'Traceback' in log:return 'unhandled_runtime_failure'
    return 'nonzero_process_without_recognized_reason'

def summarize(run,c):
    records=[]
    for s in c['scenes']:
        p=run/'cases'/('record_%04d'%s['record_index'])/'result.json'
        obj=readj(p) if p.exists() else dict(s,execution_status='not_run',scientific_status='not_evaluated')
        records.append(obj)
    counts=dict(sorted(collections.Counter(r['execution_status'] for r in records).items()))
    native_subgates=sum(r.get('native_subgate_pass') is True for r in records)
    failures=[{'ordinal':r['ordinal'],'record_index':r['record_index'],'town':r['town'],
               'status':r['execution_status'],'reason':r.get('failure_class'),'error':r.get('last_error')}
              for r in records if r['execution_status']!='record_written']
    # Never imply native legal Model3 from p1_pass.
    names=collections.Counter(s['scene_id_manifest'] for s in c['scenes'])
    provenance_flags=sum(not s['scene_id_equal'] or names[s['scene_id_manifest']]>1 for s in c['scenes'])
    out={'schema':'p1_isolated95_ledger_v1','source_sha256':FROZEN_SHA,'records_expected':95,
         'records_written':counts.get('record_written',0),'execution_counts':counts,
         'native_subgates_passed_among_written':native_subgates,
         'complete_execution':all(r['execution_status']=='record_written' for r in records),
         'strict_original_frozen_audit_passed_95':all(r['execution_status']=='record_written' and r.get('native_subgate_pass') is True for r in records),
         'source_provenance_verified':False if provenance_flags else None,
         'source_identity_flagged_record_count':provenance_flags,
         'overall_P1_scientific_pass':False,
         'model3_full_scene_physical_training_ready':False,
         'failed_records':failures,'records':records}
    return out

def run(args):
    c=input_contract(args.root);need(args.ack_dedicated,'dedicated CARLA server acknowledgement required')
    py=str(Path(args.python).resolve());need(Path(py).is_file(),'Python missing')
    env=process_env(c)
    print(probe_modules(py,env,c),flush=True)
    print('P1_CARLA_BEFORE='+json.dumps(server_safe(py,env,c,args.host,args.port)),flush=True)
    target=Path(args.output_dir).resolve();need(not target.exists(),'output directory exists; refuse overwrite')
    target.mkdir(parents=True)
    writex(target/'input_contract.json',{'frozen_sha256':FROZEN_SHA,'manifest_sha256':MANIFEST_SHA,
      'map_manifest_sha256':MAP_SHA,'scenes':c['scenes'],'single_record_process_isolation':True})
    stop_reason=None
    try:
        for s,row in zip(c['scenes'],c['rows']):
            idx=s['record_index'];d=target/'cases'/('record_%04d'%idx);d.mkdir(parents=True,exist_ok=False)
            with (d/'manifest_one.jsonl').open('x',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            out=d/'audit'
            cmd=[py,'-m','hv_v8.p1_native_audit','--manifest',str(d/'manifest_one.jsonl'),
                 '--rift-root',str(c['root']/'RIFT'),'--rift-recognition-config',str(c['cfg']),
                 '--carla-pythonapi',str(c['root']/'carla/PythonAPI/carla'),
                 '--map-identity-manifest',str(c['map_manifest']),
                 '--host',args.host,'--port',str(args.port),'--output',str(out),
                 '--startup-timeout','90','--rpc-timeout','120']
            # Confirm no leftovers BEFORE each record. A safety error aborts remaining runs.
            try: before=server_safe(py,env,c,args.host,args.port)
            except (EvidenceError,subprocess.TimeoutExpired) as e:
                stop_reason='pre_record_safety_abort: '+str(e)
                writex(d/'result.json',dict(s,execution_status='safety_abort',scientific_status='not_evaluated',failure_class='unsafe_world',last_error=stop_reason))
                break
            writex(d/'execution.json',{'command':cmd,'world_before':before,'frozen_sha256':FROZEN_SHA})
            t0=time.monotonic()
            try:
                with (d/'process.log').open('x',encoding='utf-8') as f:
                    res=subprocess.run(cmd,env=env,cwd=str(c['cap']),stdout=f,stderr=subprocess.STDOUT,
                                       timeout=args.timeout_seconds,check=False)
                rc=res.returncode;timed_out=False
            except subprocess.TimeoutExpired:
                rc=124;timed_out=True
            duration=round(time.monotonic()-t0,2)
            log=(d/'process.log').read_text(encoding='utf-8',errors='replace') if (d/'process.log').exists() else ''
            record_path=out/('record_%04d.json'%idx)
            found=record_path.is_file()
            # A written record is the actual frozen auditor's result; do not conflate with a PASS summary.
            if found:
                r=readj(record_path)
                identity=(r.get('record_index')==idx and r.get('scene_id')==row['scene_id'])
                mp=r.get('map_identity',{})
                native=r.get('native_rows',[])
                route=r.get('route_checks',[])
                gt_ids=[str(x['actor_id']) for x in readj(s['snapshot_path']).get('rift_labels',[]) if x.get('is_interactive') is True]
                sel_ids=[str(x) for x in (r.get('target_selection') or {}).get('selected_target_actor_ids',[])]
                def coord_ok(n):
                    v=n.get('coordinate_roundtrip') or {}
                    return all(isinstance(v.get(k),(int,float)) and 0<=v[k]<=tol for k,tol in
                               (('roundtrip_position_error_m',1e-6),('roundtrip_z_error_m',1e-6),('roundtrip_yaw_error_rad',1e-9)))
                native_pass=bool(identity and mp.get('town_pass') is True and mp.get('opendrive_pass') is True and
                      route and all(x.get('pass') is True for x in route) and
                      (r.get('target_selection') or {}).get('status')=='pass' and
                      gt_ids and len(gt_ids)==len(set(gt_ids)) and sorted(sel_ids)==sorted(gt_ids) and
                      (r.get('fixed_model3_blueprint') or {}).get('status')=='pass' and
                      r.get('post_spawn_pose_correction') is False and r.get('forbidden_fallbacks_used')==[] and
                      len(native)==len(gt_ids) and
                      all(x.get('rift_reference_parity_status')=='pass' and x.get('native_positive_status')=='pass'
                          and x.get('native_positive') is True and coord_ok(x) for x in native))
                status='record_written' if identity else 'record_identity_invalid'
                reason=None if identity else 'manifest_record_output_mismatch'
                physical=(r.get('fixed_model3_scene_occupancy') or {}).get('status','unknown')
                raw_counts=dict(collections.Counter(x.get('status','unknown') for x in r.get('fixed_model3_raw_pose_spawn',[])))
            else:
                status='timeout' if timed_out else 'record_failed_without_result'
                native_pass=False;physical='not_run';raw_counts={}
                reason=failure_class(log)
            # Capture the real XODR/XYZ disagreement before the next Town is loaded.
            # Snapshot, map SHA and VERIFIED episode ID must match the failing record.
            witness_status=None
            mismatch=re.search(r'ego_route\[(\d+)\] (?:exact lane identity mismatch|exact get_waypoint returned None)',log)
            trans=out/('map_transition_record_%04d.json'%idx)
            if mismatch and not found and trans.is_file():
                witness_cmd=[py,str(Path(__file__).with_name('p1_route_failure_witness.py')),
                    '--snapshot',s['snapshot_path'],'--map-manifest',str(c['map_manifest']),
                    '--transition',str(trans),'--record-index',str(idx),'--route-index',mismatch.group(1),
                    '--host',args.host,'--port',str(args.port),'--output',str(d/'route_failure_witness.json')]
                try:
                    wproc=subprocess.run(witness_cmd,env=env,cwd=str(c['cap']),capture_output=True,text=True,timeout=120)
                    with (d/'route_failure_witness.log').open('x',encoding='utf-8') as f:
                        f.write(wproc.stdout+wproc.stderr)
                    witness_status='recorded' if wproc.returncode==0 else 'witness_unavailable'
                except (OSError,subprocess.TimeoutExpired) as exc:
                    witness_status='witness_unavailable_'+type(exc).__name__
            result=dict(s,execution_status=status,scientific_status='native_subgate_pass' if native_pass else 'fail_or_unknown',
              failure_witness_status=witness_status,
              native_subgate_pass=native_pass,raw_model3_spawn_status_counts=raw_counts,
              occupancy_status=physical,process_exit_code=rc,process_seconds=duration,
              process_log_sha256=sha(d/'process.log') if (d/'process.log').exists() else None,
              record_sha256=sha(record_path) if found else None,
              failure_class=reason,last_error=next((line for line in reversed(log.splitlines()) if line.startswith('P1_FAIL')),log[-500:] if rc else None))
            writex(d/'result.json',result)
            print('P1_REAL_CASE='+json.dumps({'ordinal':s['ordinal'],'idx':idx,'town':s['town'],'status':status,
                 'native_pass':native_pass,'rc':rc,'reason':reason},ensure_ascii=False),flush=True)
            # Crash/timeout/unsafe process: no presumption the server is clean; do not continue.
            if timed_out or (not found and reason in ('unhandled_runtime_failure','nonzero_process_without_recognized_reason')):
                stop_reason='unsafe_or_unclassified_failure at record '+str(idx)
                break
            try:server_safe(py,env,c,args.host,args.port)
            except (EvidenceError,subprocess.TimeoutExpired) as e:
                stop_reason='post_record_safety_abort at '+str(idx)+': '+str(e)
                break
    finally:
        ledger=summarize(target,c)
        ledger['abort_reason']=stop_reason
        writex(target/'ledger.json',ledger)
        print('P1_REAL_LEDGER='+str(target/'ledger.json'),flush=True)
        print('P1_REAL_WRITTEN='+str(ledger['records_written'])+'/95',flush=True)
        print('P1_REAL_COUNTS='+json.dumps(ledger['execution_counts']),flush=True)
    return 4 if stop_reason else (0 if ledger['strict_original_frozen_audit_passed_95'] else 3)

def report(args):
    c=input_contract(args.root);run_dir=Path(args.run_dir).resolve()
    need((run_dir/'input_contract.json').is_file(),'missing run input contract')
    ledger=summarize(run_dir,c)
    print('P1_REAL_COUNTS='+json.dumps(ledger['execution_counts']))
    print('P1_NATIVE_VERIFIED='+str(ledger['native_subgates_passed_among_written'])+'/95')
    print('P1_FULL95_ORIGINAL_CONTRACT_PASS='+str(ledger['strict_original_frozen_audit_passed_95']))
    print('P1_MODEL3_FULL_SCENE_TRAINING_READY=False')
    print('P1_FAILED_RECORDS='+json.dumps(ledger['failed_records'],ensure_ascii=False))
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('run');a.add_argument('--root',required=True);a.add_argument('--python',required=True)
    a.add_argument('--host',default='127.0.0.1');a.add_argument('--port',type=int,default=31120)
    a.add_argument('--output-dir',required=True);a.add_argument('--ack-dedicated',action='store_true');a.add_argument('--timeout-seconds',type=int,default=900)
    b=sub.add_parser('report');b.add_argument('--root',required=True);b.add_argument('--run-dir',required=True)
    args=p.parse_args()
    try:return run(args) if args.cmd=='run' else report(args)
    except (EvidenceError,ValueError,OSError,subprocess.TimeoutExpired) as e:
        print('P1_SUPERVISOR_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
