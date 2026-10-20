#!/usr/bin/env python3
"""HV-V8 P1 road140 / ego_route[7] evidence acquisition (NOT an audit patch).

Python 3.8+. Offline phase reads frozen source, 95 manifest, all snapshots,
map provenance and the ORIGINAL 95-failure log. Geometry phase additionally
connects to existing CARLA world and requires it ALREADY be the correct Town,
with exact OpenDRIVE SHA confirmed BEFORE any waypoint query. NEVER loads a
world, spawns/destroys actors, ticks world, changes P1 thresholds or inputs.
Reports are NEW separate JSON evidence files, created exclusively with 'x'.

Examples:
 python p1_route140_readonly_forensics.py --phase offline
 python p1_route140_readonly_forensics.py --phase geometry --town Town15
 python p1_route140_readonly_forensics.py --self-test
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT_DEFAULT = '/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop'
FROZEN_SHA = '34a80666b1030434c528d0b91f7bb7f2994eb65bba21b5b4c8540fa972898d2c'
MAP_SHA = {
    'Town04':'4dddd1560ab7812b0e89cd4a523d8b6f2b93947eafb496d9eeb83cc772116c85',
    'Town11':'52ffa8412425b87394afa6a37bf0b688c0584a24df6ae88fea53866acb91e222',
    'Town12':'26fa53d4902ac3397d35f948969f0c77c1bf023755ede2c67edf00bef817151f',
    'Town13':'5354326f2fe4d26f0dcb5c82acf462257e638278363ab7adecb97ec3aa1f426f',
    'Town15':'8e9c69cfb3e750ace3e5d6b057e46786952a073c3924f319bbabb1a166dc2c17',
}
MAP_COUNTS = {'Town04':12,'Town11':5,'Town12':28,'Town13':28,'Town15':22}
SIGNATURE = {'route_index':7,'expected_road_id':140,'expected_section_id':0,'expected_lane_id':-3,
             's_abs_error_m':0.05094718933105469,'position_error_m':0.05077684135740183,
             'yaw_error_deg':0.00669097900390625}
REQUIRED_ROUTE = ('x','y','z','yaw_deg','road_id','section_id','lane_id','s')

class EvidenceError(RuntimeError): pass

def require(ok, msg):
    if not ok: raise EvidenceError(msg)

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(path): return sha_bytes(path.read_bytes())
def load_json(path): return json.loads(path.read_text(encoding='utf-8'))
def town_of(raw): return str(raw or '').replace('\\','/').rstrip('/').split('/')[-1]
def small_float(v):
    if isinstance(v, bool): raise ValueError('boolean is not numeric')
    out=float(v)
    if not math.isfinite(out): raise ValueError('nonfinite number')
    return out

def route_point(r):
    require(isinstance(r,dict) and all(k in r for k in REQUIRED_ROUTE),
            'missing one of original route fields {}'.format(REQUIRED_ROUTE))
    v={k:(int(r[k]) if k in ('road_id','section_id','lane_id') else small_float(r[k]))
       for k in REQUIRED_ROUTE}
    return v

def resolve_snapshot(root, manifest, raw):
    require(isinstance(raw,str) and bool(raw.strip()), 'empty snapshot_path')
    p=Path(raw).expanduser()
    choices=[p] if p.is_absolute() else [root/p,manifest.parent/p]
    found=sorted(set(x.resolve() for x in choices if x.is_file()))
    require(len(found)==1, 'snapshot_path ambiguous/missing: {!r}; candidates={}'.format(raw,found))
    return found[0]

def parse_log(text):
    match=re.search(r'P1_FAIL\s+ego_route\[(\d+)\]\s+CARLA identity mismatch:\s*\{',text)
    require(match is not None,'failure signature missing from original log')
    start=match.end()-1
    # Reconstruct the printed Python dict without executing untrusted content.
    level=0; end=None
    for i in range(start,len(text)):
        if text[i]=='{': level+=1
        if text[i]=='}':
            level-=1
            if not level: end=i+1;break
    require(end is not None,'incomplete failure dictionary in log')
    values=ast.literal_eval(text[start:end])
    require(isinstance(values,dict),'failure details not a dictionary')
    require(int(match.group(1))==7,'log concerns a different route index')
    for k in ('expected_road_id','expected_section_id','expected_lane_id','s_abs_error_m',
              'position_error_m','yaw_error_deg'):
        require(k in values,'log missing {}'.format(k))
    for k in ('expected_road_id','expected_section_id','expected_lane_id'):
        require(int(values[k])==SIGNATURE[k],'different failure identity {}'.format(k))
    require(abs(float(values['s_abs_error_m'])-SIGNATURE['s_abs_error_m'])<1e-9,
            'different s-error failure, do not mix run evidence')
    return {'route_index':int(match.group(1)),'details':values,
            'process_exit_2_reported':bool(re.search('P1_FULL95_PROCESS_EXIT=2',text)),
            'strict_acceptance_fail_reported':('P1_FULL95_STRICT_ACCEPTANCE_FAIL' in text)}

def inputs(root):
    base=root/'data/hv_init/v7'
    return {
        'manifest':base/'manifests/p1_calibration95.jsonl',
        'map_manifest':base/'manifests/p1_map_identity_manifest_v2_20261017_154737.json',
        'frozen':base/'p1_frozen_capsule_20261018_033841_9535/hv_v8/p1_native_audit.py',
        'provenance':base/'p1_frozen_capsule_20261018_033841_9535/p1_source_provenance.json',
        'log':base/'p1_full95_resume_20261018_041350_746.log',
        'failed_output':base/'p1_real_frozen_full95_resume_20261018_041350_746',
        'smoke_summary':base/'p1_real_frozen_smoke_20261018_033841_9535/summary.json',
        'smoke_record':base/'p1_real_frozen_smoke_20261018_033841_9535/record_0000.json',
    }

def collect_offline(root):
    paths=inputs(root)
    for key,p in paths.items():
        if key != 'failed_output': require(p.is_file(),'missing {}: {}'.format(key,p))
    require(sha_file(paths['frozen'])==FROZEN_SHA,
            'frozen source is NOT known-PASS SHA: {}'.format(sha_file(paths['frozen'])))
    prov=load_json(paths['provenance'])
    require(prov.get('frozen_sha256')==FROZEN_SHA,'frozen provenance SHA mismatch')
    require(prov.get('patches_applied_to')=='FROZEN_CAPSULE_ONLY',
            'provenance does not identify isolated frozen source')
    hashes={key:sha_file(paths[key]) for key in ('manifest','map_manifest','frozen','provenance','log','smoke_summary','smoke_record')}
    for key,label in [('manifest','p1_manifest'),('map_manifest','map_identity_manifest')]:
        payload=(prov.get('inputs') or {}).get(label) or {}
        require(payload.get('sha256')==hashes[key],
                '{} changed relative to frozen source provenance'.format(key))
    rc=(prov.get('inputs') or {}).get('rift_config') or {}
    riftpath=Path(rc.get('path',''))
    require(riftpath.is_file() and sha_file(riftpath)==rc.get('sha256'),
            'RIFT config hash does not match frozen provenance')
    hashes['rift_config']=sha_file(riftpath)
    mp=load_json(paths['map_manifest'])
    require(mp.get('schema_version')=='p1_map_identity_manifest_v1','map identity schema changed')
    maps=mp.get('maps') or []
    require(len(maps)==5,'expected exactly five independently signed maps')
    actual_map={town_of(x.get('town')):str(x.get('opendrive_sha256','')).lower() for x in maps}
    require(actual_map==MAP_SHA,'independent map identity manifest differs from historical calibration')
    manifest=paths['manifest']
    lines=manifest.read_text(encoding='utf-8').splitlines()
    rows=[json.loads(line) for line in lines if line.strip()]
    require(len(rows)==95,'manifest has {} rows not 95'.format(len(rows)))
    scanned=[];counts=Counter();ids=set();scene_id_mismatches=[]
    for ordinal,row in enumerate(rows):
        require(isinstance(row,dict),'manifest row {} invalid'.format(ordinal))
        rid=int(row['record_index'])
        require(rid not in ids,'duplicate record_index {}'.format(rid));ids.add(rid)
        path=resolve_snapshot(root,manifest,row.get('snapshot_path'))
        raw=path.read_bytes();scene=json.loads(raw.decode('utf-8'))
        scene_id=scene.get('scene_id')
        if row.get('scene_id') != scene_id:
            scene_id_mismatches.append({
                'manifest_ordinal': ordinal, 'record_index': rid,
                'manifest_scene_id': row.get('scene_id'),
                'snapshot_scene_id': scene_id,
                'verification_status': 'UNRESOLVED_IDENTITY_DIFFERENCE',
            })
        dig=sha_bytes(raw)
        if row.get('snapshot_sha256') is not None:
            require(str(row['snapshot_sha256']).lower()==dig,
                    'snapshot SHA drift ordinal={}'.format(ordinal))
        town=town_of(scene.get('town') or scene.get('map_name'))
        require(town in MAP_SHA,'unknown snapshot Town row={}: {!r}'.format(ordinal,town))
        if row.get('town') is not None:
            require(town_of(row['town'])==town,'row/scene Town mismatch ordinal={}'.format(ordinal))
        counts[town]+=1
        route=scene.get('ego_route')
        require(isinstance(route,list) and bool(route),'missing route ordinal={}'.format(ordinal))
        points=[route_point(p) for p in route[:50]]
        meta={'manifest_ordinal':ordinal,'record_index':rid,'scene_id':scene_id,
              'town':town,'snapshot_path':str(path),'snapshot_sha256':dig,
              'route_length':len(route),'route_points':points,
              'snapshot_ego_route_s_provenance':'UNVERIFIED; needs producer-code and input-lineage evidence'}
        if len(points)>7:
            p=points[7]
            meta['route7_matches_road_identity']=(p['road_id']==140 and p['section_id']==0 and p['lane_id']==-3)
        else: meta['route7_matches_road_identity']=False
        scanned.append(meta)
    require(dict(counts)==MAP_COUNTS,'Town distribution mismatch {}'.format(dict(counts)))
    producer_path=root/'RIFT/rift/cbv/initialization/hv_scene_snapshot.py'
    serializer_path=root/'RIFT/rift/cbv/initialization/rift_interaction_labeler.py'
    source_chain={'version_used_for_historical_snapshots_confirmed':False,
                  'caveat':'Current source text is not proof of the version used to collect historical snapshots'}
    for label,file in [('snapshot_producer',producer_path),('waypoint_serializer',serializer_path)]:
        source_chain[label]={'path':str(file),'file_exists':file.is_file()}
        if file.is_file():
            txt=file.read_text(encoding='utf-8')
            source_chain[label]['sha256']=sha_bytes(txt.encode('utf-8'))
            source_chain[label]['static_contract_match']=(
                'waypoint_to_dict(wp)' in txt and 'ego_rest_route_waypoints' in txt
                if label=='snapshot_producer' else
                '"s": float(waypoint.s)' in txt and 'transform.location.x' in txt)
    report={'schema_version':'hv_v8_p1_road140_evidence_v1','phase':'offline',
            'scientific_P1_pass_claim':False,'remote_files_read':True,
            'all_inputs_read_only':True,'world_changed':False,
            'input_sha256':hashes,'trusted_opendrive_sha256':actual_map,
            'snapshot_route_s_source_code':source_chain,
            'log_failure':parse_log(paths['log'].read_text(encoding='utf-8',errors='replace')),
            'manifest_counts':dict(counts),'manifest_rows':len(rows),
            'scene_id_mismatches':scene_id_mismatches,
            'scene_id_identity_all_verified':not bool(scene_id_mismatches),
            'scan':scanned}
    summary=load_json(paths['smoke_summary'])
    report['prior_smoke']={'records':summary.get('records'),'p1_pass':summary.get('p1_pass'),
                           'rift_reference_parity_pass_count':summary.get('rift_reference_parity_pass_count')}
    require(summary.get('records')==1 and summary.get('p1_pass') is True and
            summary.get('rift_reference_parity_pass_count')==4,
            'prior smoke does not match reported actual one-record success')
    failed=paths['failed_output']
    done=[];unknown=[];corrupted=[]
    if failed.is_dir():
        for f in sorted(failed.glob('record_*.json')):
            try:
                d=load_json(f); i=int(d['record_index'])
                require(f.name=='record_{:04d}.json'.format(i),'record filename/index inconsistent {}'.format(f.name))
                if i in ids: done.append({'record_index':i,'filename':f.name,'sha256':sha_file(f)})
                else: unknown.append(f.name)
            except Exception as e: corrupted.append({'path':str(f),'reason':repr(e)})
    report['failed_output']={'path':str(failed),'directory_exists':failed.is_dir(),
                             'summary_exists':(failed/'summary.json').exists(),
                             'completed_record_files':done,'unknown_record_files':unknown,'corrupt_record_files':corrupted}
    require(not report['failed_output']['summary_exists'],'unexpected full95 summary exists; inspect run before proceeding')
    require(not unknown and not corrupted,'failed output contains unknown/corrupt records')
    ordered=[r['record_index'] for r in scanned]
    doneids={x['record_index'] for x in done}
    prefix_len=0
    for i in ordered:
        if i in doneids: prefix_len+=1
        else: break
    report['completed_prefix_length']=prefix_len
    report['first_not_completed_record']=(scanned[prefix_len]['record_index'] if prefix_len<95 else None)
    report['completed_files_form_prefix']=(doneids==set(ordered[:prefix_len]))
    require(report['completed_files_form_prefix'],'completed records not a strict prefix; cannot infer sequence')
    candidates=[r for r in scanned if r['route7_matches_road_identity']]
    report['candidates']=[dict([(k,v) for k,v in r.items() if k!='route_points'] + [('route7',r['route_points'][7])]) for r in candidates]
    report['candidate_count']=len(candidates)
    # This sequence intersection is only a clue until real geometry reproduces errors.
    report['first_not_completed_is_candidate']=(
        prefix_len<95 and scanned[prefix_len]['route7_matches_road_identity'])
    report['record_identification_status']='CANDIDATES_ONLY_UNTIL_GEOMETRY_SIGNATURE_UNIQUE'
    return report

def angle_delta(a,b): return abs((float(a)-float(b)+180)%360-180)

def xyzdistance(a,b):
    return math.sqrt(sum((a[k]-b[k])**2 for k in ('x','y','z')))

def way_dict(w):
    if w is None: return {'found':False}
    tf=w.transform
    return {'found':True,'road_id':int(w.road_id),'section_id':int(w.section_id),
            'lane_id':int(w.lane_id),'s':float(w.s),
            'x':float(tf.location.x),'y':float(tf.location.y),'z':float(tf.location.z),
            'yaw_deg':float(tf.rotation.yaw)}

def compare(w,r):
    if not w.get('found'): return {'status':'NOT_EVALUABLE'}
    return {'status':'EVALUABLE',
            'identity_match':all(w[k]==r[k] for k in ('road_id','section_id','lane_id')),
            's_signed_delta_m':w['s']-r['s'],
            's_abs_error_m':abs(w['s']-r['s']),
            'position_xy_error_m':math.hypot(w['x']-r['x'],w['y']-r['y']),
            'position_xyz_error_m':xyzdistance(w,r),
            'yaw_error_deg':angle_delta(w['yaw_deg'],r['yaw_deg'])}

def current_wp(map_,carla,r,method):
    loc=carla.Location(x=r['x'],y=r['y'],z=r['z'])
    if method=='xodr': return map_.get_waypoint_xodr(r['road_id'],r['lane_id'],r['s'])
    if method=='exact':return map_.get_waypoint(loc,project_to_road=False,lane_type=carla.LaneType.Driving)
    if method=='projected':return map_.get_waypoint(loc,project_to_road=True,lane_type=carla.LaneType.Driving)
    raise ValueError(method)

def observe(map_,carla,r):
    out={'snapshot':r}
    for method in ('xodr','exact','projected'):
        try: w=way_dict(current_wp(map_,carla,r,method))
        except Exception as e:w={'found':False,'error_type':type(e).__name__,'error':str(e)}
        out[method]={'waypoint':w,'vs_snapshot':compare(w,r)}
    for a,b in (('xodr','exact'),('xodr','projected'),('exact','projected')):
        wa=out[a]['waypoint'];wb=out[b]['waypoint']
        out[a+'_vs_'+b]=compare(wa,wb) if wb.get('found') else {'status':'NOT_EVALUABLE'}
    # No numeric misses become a PASS.
    ex=out['exact']['vs_snapshot']
    out['original_P1_exact_s_gate_pass']=(ex.get('status')=='EVALUABLE' and ex.get('identity_match')
                                          and ex.get('s_abs_error_m',float('inf'))<=0.05)
    return out

def signature_match(o,log):
    ex=o['exact']['vs_snapshot'];r=o['snapshot'];v=log['details']
    if ex.get('status')!='EVALUABLE' or not ex.get('identity_match'):return False
    if (r['road_id'],r['section_id'],r['lane_id']) != (140,0,-3):return False
    # Extremely narrow comparisons, NOT acceptance thresholds. Identify THIS specific failure only.
    return (abs(ex['s_abs_error_m']-float(v['s_abs_error_m']))<=1e-6
            and abs(ex['position_xyz_error_m']-float(v['position_error_m']))<=1e-5
            and abs(ex['yaw_error_deg']-float(v['yaw_error_deg']))<=1e-4
            and ex['s_abs_error_m']>0.05)

def quantile(v,q):
    if not v:return None
    a=sorted(v);pos=(len(a)-1)*q;lo=int(pos);hi=min(lo+1,len(a)-1)
    return a[lo]*(1-(pos-lo))+a[hi]*(pos-lo)

def stats(vals):
    return {'n':len(vals),'min':min(vals) if vals else None,
            'p50':quantile(vals,.5),'p90':quantile(vals,.9),'p99':quantile(vals,.99),
            'max':max(vals) if vals else None}

def collect_geometry(root,offline,town,host,port,rpc_timeout,only_candidates=False):
    require(town in MAP_SHA,'--town must be one of trusted P1 Towns')
    # Read-only: this script has deliberately NO load_world, spawn_actor,
    # destroy, tick or wait_for_tick invocation anywhere.
    distdir=root/'carla/PythonAPI/carla/dist'
    for egg in sorted(distdir.glob('carla-0.9.15-*.egg')):
        sys.path.insert(0,str(egg))
    sys.path.insert(0,str(root/'carla/PythonAPI/carla'))
    import carla
    client=carla.Client(host,int(port));client.set_timeout(float(rpc_timeout))
    require(str(client.get_server_version())=='0.9.15','CARLA server not 0.9.15')
    world=client.get_world();map_=world.get_map();current=town_of(map_.name)
    require(current==town,'CARLA currently {} but requested {}; no world change made'.format(map_.name,town))
    expected=offline['trusted_opendrive_sha256'][town]
    real=sha_bytes(map_.to_opendrive().encode('utf-8'))
    require(real==expected,'OpenDRIVE SHA mismatch! expected={} actual={}'.format(expected,real))
    records=[];observations=[];matches=[];metric={k:[] for k in ('exact','xodr','projected')}
    missing=Counter();lane_mismatches=Counter();bins=Counter()
    candidate_ids={c['record_index'] for c in offline['candidates']}
    for r in offline['scan']:
        if r['town']!=town:continue
        if only_candidates and r['record_index'] not in candidate_ids: continue
        record={'manifest_ordinal':r['manifest_ordinal'],'record_index':r['record_index'],
                'scene_id':r['scene_id'],'snapshot_path':r['snapshot_path'],
                'snapshot_sha256':r['snapshot_sha256'],'town':town,
                'points':[]}
        for idx,p in enumerate(r['route_points']):
            o=observe(map_,carla,p)
            o['route_index']=idx
            record['points'].append(o);observations.append(o)
            for method in ('xodr','exact','projected'):
                val=o[method]['vs_snapshot']
                if val['status']!='EVALUABLE':missing[method]+=1
                elif not val['identity_match']:lane_mismatches[method]+=1
                else: metric[method].append(val['s_abs_error_m'])
            e=o['exact']['vs_snapshot']
            if e.get('status')=='EVALUABLE' and e.get('identity_match'):
                d=e['s_abs_error_m']
                boundary=('<=1e-4' if d<=1e-4 else '<=0.01' if d<=.01 else
                          '<=0.05' if d<=.05 else '<=0.10' if d<=.1 else '>0.10')
                bins[boundary]+=1
            if idx==7 and signature_match(o,offline['log_failure']):
                matches.append({'manifest_ordinal':r['manifest_ordinal'],
                                'record_index':r['record_index'],'scene_id':r['scene_id'],
                                'town':town,'snapshot_path':r['snapshot_path'],
                                'snapshot_sha256':r['snapshot_sha256'],
                                'route7_observation':o})
        records.append(record)
    # Re-check map AFTER all observations; a concurrent map switch invalidates evidence.
    world_after=client.get_world()
    map_after=world_after.get_map()
    require(town_of(map_after.name)==town and
            sha_bytes(map_after.to_opendrive().encode('utf-8'))==expected,
            'world map changed during diagnostic; invalid observations')
    observed_candidates=[x for x in offline['candidates'] if x['town']==town]
    identification=('UNIQUE_IN_SCANNED_TOWN' if len(matches)==1 else
                    'NO_SIGNATURE_MATCH' if not matches else 'AMBIGUOUS_MULTIPLE_SIGNATURE_MATCHES')
    # To certify global uniqueness, scan each Town with a candidate; do not infer
    # from a single Town unless all offline candidates are in that Town.
    if (len(matches)==1 and set(x['town'] for x in offline['candidates'])=={town}
            and offline['completed_files_form_prefix']
            and matches[0]['record_index']==offline['first_not_completed_record']):
        identification='UNIQUE_SIGNATURE_AND_COMPLETED_PREFIX_CONSISTENT'
    return {'schema_version':'hv_v8_p1_road140_geometry_v1','phase':'geometry',
            'scientific_P1_pass_claim':False,'CARLA_geometry_observed':True,
            'world_unchanged_by_this_tool':True,'town':town,'CARLA_map_name':map_.name,
            'opendrive_sha256_expected':expected,'opendrive_sha256_actual':real,
            'full_town_scanned':not only_candidates,
            'records_scanned':len(records),'points_scanned':len(observations),
            'candidate_records_in_town':len(observed_candidates),
            'historical_failure_signature':offline['log_failure'],
            'exact_s_error_bins_m':dict(bins),
            'missing_lookup_counts':dict(missing),'lane_mismatch_counts':dict(lane_mismatches),
            's_abs_error_statistics_m':{k:stats(v) for k,v in metric.items()},
            'signature_matches':matches,'identification_status':identification,
            'snapshot_s_provenance':'UNVERIFIED: examine producer extraction and original trace',
            'records':records}

def write_report(root,phase,report,output):
    from datetime import datetime
    name='p1_road140_'+phase+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'
    path=Path(output).expanduser() if output else root/'data/hv_init/v7'/name
    require(path.parent.is_dir(),'report parent directory must already exist: {}'.format(path.parent))
    with path.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('EVIDENCE_REPORT={}'.format(path))
    print('EVIDENCE_SHA256={}'.format(sha_file(path)))
    return path

def self_test():
    r={'x':0.,'y':0.,'z':0.,'yaw_deg':0.,'road_id':140,'section_id':0,'lane_id':-3,'s':10.}
    w=dict(r);w['found']=True;w['s']+=SIGNATURE['s_abs_error_m'];w['x']+=SIGNATURE['position_error_m'];w['yaw_deg']+=SIGNATURE['yaw_error_deg']
    d=compare(w,r)
    assert d['identity_match'] and d['s_abs_error_m']>.05
    o={'snapshot':r,'exact':{'vs_snapshot':d}}
    log={'details':dict(SIGNATURE)}
    assert signature_match(o,log)
    bad=dict(r);bad['lane_id']=-2
    assert not signature_match({'snapshot':bad,'exact':{'vs_snapshot':d}},log)
    wbad=dict(w);wbad['s']=10.10
    assert not signature_match({'snapshot':r,'exact':{'vs_snapshot':compare(wbad,r)}},log)
    e={'snapshot':r,'exact':{'vs_snapshot':{'status':'NOT_EVALUABLE'}}}
    assert not signature_match(e,log)
    assert angle_delta(179.,-179.)==2.
    class Point:
        def __init__(self,x=0.,y=0.,z=0.):self.x,self.y,self.z=x,y,z
    class Rotation:
        def __init__(self,yaw=0.):self.yaw=yaw
    class Transform:
        def __init__(self,x,s,yaw=0.):self.location=Point(x,0.,0.);self.rotation=Rotation(yaw)
    class Waypoint:
        def __init__(self,x,s,yaw=0.,lane=-3):
            self.road_id,self.section_id,self.lane_id=140,0,lane
            self.s=s;self.transform=Transform(x,s,yaw)
    class FakeCarla:
        Location=Point
        class LaneType:Driving=1
    class FakeMap:
        def get_waypoint_xodr(self,road,lane,s):return Waypoint(0.,s)
        def get_waypoint(self,location,project_to_road=False,lane_type=None):
            return Waypoint(SIGNATURE['position_error_m'],
                            10.+SIGNATURE['s_abs_error_m'],SIGNATURE['yaw_error_deg'])
    obs=observe(FakeMap(),FakeCarla,r)
    assert obs['xodr']['vs_snapshot']['s_abs_error_m']==0.
    assert not obs['original_P1_exact_s_gate_pass']
    assert signature_match(obs,log)
    class WrongLaneMap(FakeMap):
        def get_waypoint(self,*args,**kwargs):return Waypoint(0.,10.,lane=-2)
    assert not signature_match(observe(WrongLaneMap(),FakeCarla,r),log)
    print('LOCAL_SELF_TEST_PASS (mock 3-way map, math, positive-negative examples only; NOT CARLA / P1 PASS)')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['offline','geometry'],default='offline')
    p.add_argument('--root',default=ROOT_DEFAULT)
    p.add_argument('--town',choices=sorted(MAP_SHA))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=31120)
    p.add_argument('--rpc-timeout',type=float,default=120.)
    p.add_argument('--only-candidates',action='store_true',help='compare only candidates (less complete distribution)')
    p.add_argument('--output',help='new JSON report path, parent must exist; never overwrite')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test();return 0
    root=Path(args.root).expanduser().resolve()
    require(root.is_dir(),'ROOT not accessible: {}'.format(root))
    offline=collect_offline(root)
    if args.phase=='offline':
        report=offline
        print('SCENE_ID_MISMATCH_COUNT={}'.format(len(offline['scene_id_mismatches'])))
        for mismatch in offline['scene_id_mismatches']:
            print('UNRESOLVED_SCENE_ID={}'.format(json.dumps(mismatch,ensure_ascii=False,sort_keys=True)))
        print('FIRST_NOT_COMPLETED_RECORD={}'.format(offline['first_not_completed_record']))
        print('FIRST_NOT_COMPLETED_IS_CANDIDATE={}'.format(offline['first_not_completed_is_candidate']))
        print('CANDIDATE_COUNT={}'.format(offline['candidate_count']))
        print('COMPLETED_PREFIX_LENGTH={}'.format(offline['completed_prefix_length']))
        for c in offline['candidates']:
            print('CANDIDATE={}'.format(json.dumps(c,ensure_ascii=False,sort_keys=True)))
    else:
        require(args.town is not None,'--phase geometry requires --town (map never auto-switched)')
        report=collect_geometry(root,offline,args.town,args.host,args.port,args.rpc_timeout,args.only_candidates)
        print('REAL_CARLA_GEOMETRY_TOWN={} SHA_MATCH={}'.format(args.town,report['opendrive_sha256_actual']))
        print('SIGNATURE_MATCH_COUNT={}'.format(len(report['signature_matches'])))
        print('RECORD_IDENTIFICATION_STATUS={}'.format(report['identification_status']))
        for x in report['signature_matches']:
            print('MATCHED_RECORD={}'.format(json.dumps({k:v for k,v in x.items() if k!='route7_observation'},ensure_ascii=False)))
    write_report(root,args.phase,report,args.output)
    print('DIAGNOSTIC_ONLY_NO_P1_PASS_CLAIM')
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as e:
        print('FORENSICS_FAIL_CLOSED {}: {}'.format(type(e).__name__,e),file=sys.stderr)
        sys.exit(2)
