#!/usr/bin/env python3
"""Offline, read-only, fail-closed one-shot P1/P2 graph inputs extraction.

No CARLA connection, no RIFT, no spawning, no GT in candidate creation.
Input contract: existing P1 frozen 95 manifest + immutable snapshot hashes;
P2 pinned Town15 graph and GT-free candidates.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import shutil
import tarfile
import sys

MANIFEST_SHA = '92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
MAP_MANIFEST_SHA = '9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'
GRAPH = 'p2_topology_town15_20261019_041323_1291.json'
APP_DIR = 'p2_town15_full_lane_v2_20261019_122546_5560'
REVIEW_DIR = 'p2_station_screen_20261019_134327_5929'

def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def need(cond,msg):
    if not cond:raise ValueError(msg)

def load(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def jlines(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def lane(v):
    if isinstance(v,(list,tuple)):
        need(len(v)==3,'invalid lane list')
        return tuple(int(n) for n in v)
    return (int(v['road_id']),int(v['section_id']),int(v['lane_id']))

def pt(row):
    keys=('road_id','section_id','lane_id','s','x','y','z','yaw_deg','is_junction')
    need(all(k in row for k in keys),'incomplete original ego_route waypoint')
    z={k:row[k] for k in keys}
    need(all(math.isfinite(float(z[k])) for k in ('s','x','y','z','yaw_deg')),'nonfinite original route coordinates')
    return z

def p1_extract(root,base):
    mf=base/'manifests/p1_calibration95.jsonl'
    mm=base/'manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    need(digest(mf)==MANIFEST_SHA,'P1 95 immutable manifest SHA mismatch')
    need(digest(mm)==MAP_MANIFEST_SHA,'P1 fixed map manifest SHA mismatch')
    records=jlines(mf)
    need(len(records)==95 and len(set(int(r['record_index']) for r in records))==95,'P1 not exact frozen 95 unique records')
    occurrences=[]; clusters=collections.defaultdict(list); issues=[]; totals=collections.Counter()
    for ordinal,r in enumerate(records):
        path=pathlib.Path(r['snapshot_path']).resolve()
        need(path.is_file(),'missing original P1 snapshot: '+str(path))
        need(digest(path)==r['snapshot_sha256'],'immutable snapshot SHA mismatch: '+str(path))
        snap=load(path);route=snap['ego_route'];town=snap['town'];idx=int(r['record_index'])
        need(isinstance(route,list) and len(route)>=2,'short ego route '+str(idx))
        totals[town]+=1
        for j in range(1,len(route)):
            a,b=pt(route[j-1]),pt(route[j])
            ka,kb=lane(a),lane(b)
            dist=math.sqrt(sum((float(a[v])-float(b[v]))**2 for v in ('x','y','z')))
            if ka==kb:
                directional=(float(b['s'])-float(a['s'])) * (-1 if ka[2]>0 else 1)
                if directional < -0.01:
                    issues.append({'record_index':idx,'town':town,'route_index':j,
                                   'lane':list(ka),'delta_travel_s_m':directional,'distance_xyz_m':dist})
                continue
            occurrence={'ordinal':ordinal,'record_index':idx,'scene_id':r['scene_id'],
                        'town':town,'route_index':j,'source':a,'target':b,'xyz_gap_m':dist,
                        'episode_key':r.get('episode_key'),'snapshot_sha256':r['snapshot_sha256']}
            occurrences.append(occurrence)
            key=(town,ka,kb)
            clusters[key].append({'record_index':idx,'route_index':j,'xyz_gap_m':dist,
                                  'source_s':a['s'],'target_s':b['s']})
    grouped=[{'town':town,'source_lane':list(a),'target_lane':list(b),'occurrences':len(v),
              'records':sorted(set(o['record_index'] for o in v)),
              'positions':v} for (town,a,b),v in clusters.items()]
    grouped.sort(key=lambda v:(v['town'],v['source_lane'],v['target_lane']))
    return {'schema':'p1_exact_snapshot_transition_catalog_v1','gt_touched':False,
            'frozen_manifest_sha256':MANIFEST_SHA,'map_manifest_sha256':MAP_MANIFEST_SHA,
            'town_counts':dict(totals),'records_checked':len(records),
            'all_key_change_occurrences':occurrences,'unique_directed_key_changes':grouped,
            'same_lane_reverse_station_warnings':issues}

def p2_extract(base):
    graphpath=base/GRAPH
    appropath=base/APP_DIR/'approaches.json'
    graph=load(graphpath); app=load(appropath)
    need(graph.get('town')=='Town15' and app.get('town')=='Town15','P2 input Town mismatch')
    need(graph.get('map_sha256')=='8e9c69cfb3e750ace3e5d6b057e46786952a073c3924f319bbabb1a166dc2c17','Town15 XODR SHA mismatch')
    need(app.get('GT_used_in_make') is False,'GT contamination in candidate generation')
    frozen={(lane(e['from']),lane(e['to'])) for e in graph['directed_edges']}
    edge_users=collections.defaultdict(list); counts=collections.Counter(); paths=0; records=0
    for record in app['records']:
        idx=int(record['record_index']);records+=1
        for k,p in enumerate(record['approaches']):
            paths+=1
            keys=[lane(k0) for k0 in p['lane_path_to_anchor']]
            types=p['path_edge_types'];need(len(keys)==len(types)+1,'lane path edge count mismatches')
            for i,t in enumerate(types):
                if t!='longitudinal':continue
                edge=(keys[i],keys[i+1]);need(edge in frozen,'P2 candidate longitudinal edge absent from pinned graph')
                counts[edge]+=1
                edge_users[edge].append([idx,k,i])
    contactpath=base/REVIEW_DIR/'directed_contact_samples.json'
    contacts=None
    if contactpath.is_file():
        d=load(contactpath)
        need(d.get('town')=='Town15' and d.get('map_sha256')==graph['map_sha256'],'contact map SHA mismatch')
        need(d.get('topology_sha256')==digest(graphpath),'contact topology SHA mismatch')
        contacts=d
    have=collections.defaultdict(list)
    if contacts is not None:
        for c in contacts['samples']:
            edge=(lane(c['from_lane']),lane(c['to_lane']))
            if edge in frozen:have[edge].append({'source_s':c['source']['s'],'target_s':c['target']['s']})
    edgeinfo=[{'from_lane':list(e[0]),'to_lane':list(e[1]),'path_references':n,
               'reference_sample':[{'record_index':t[0],'path_ordinal':t[1],'edge_ordinal':t[2]} for t in edge_users[e][:5]],
               'observed_CARLA_contact_samples':have.get(e,[]),
               'has_contact_sample':bool(have.get(e))}
              for e,n in counts.most_common()]
    return {'schema':'p2_actual_candidate_edges_demand_v1','GT_used_in_make':False,
            'town':'Town15','map_sha256':graph['map_sha256'],
            'input_sha256':{'topology':digest(graphpath),'approaches':digest(appropath),
                            'contact':digest(contactpath) if contacts is not None else None},
            'candidate_records':records,'candidate_paths':paths,'unique_longitudinal_edges_required':len(edgeinfo),
            'longitudinal_edge_references':sum(counts.values()),
            'unique_required_edges_with_local_contact':sum(bool(v) for v in (have.get(e) for e in counts)),
            'unique_required_edges_without_local_contact':sum(not have.get(e) for e in counts),
            'actual_required_directed_edges':edgeinfo}

def write(path,data):
    pathlib.Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def collect(root,out):
    root=pathlib.Path(root).resolve();base=root/'data/hv_init/v7';out=pathlib.Path(out).resolve()
    need(not out.exists(),'Refuse overwrite output directory: '+str(out))
    p1=p1_extract(root,base);p2=p2_extract(base)
    out.mkdir(parents=True)
    write(out/'p1_all95_route_transitions.json',p1)
    write(out/'p2_candidate_124_edge_demand.json',p2)
    raw=[base/'manifests/p1_calibration95.jsonl',
         base/'manifests/p1_map_identity_manifest_v2_20261017_154737.json',
         base/GRAPH,base/APP_DIR/'approaches.json',
         base/APP_DIR/'geometry.json',base/APP_DIR/'lateral.json',
         base/REVIEW_DIR/'directed_contact_samples.json']
    copied=[]; missing=[]
    for f in raw:
        if f.is_file():
            dst=out/'source'/f.relative_to(base)
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(str(f),str(dst))
            copied.append({'name':str(f.relative_to(base)),'sha256':digest(f)})
        else: missing.append(str(f))
    write(out/'manifest.json',{'P1_checked_records':p1['records_checked'],
           'P1_unique_key_changes':len(p1['unique_directed_key_changes']),
           'P1_transition_occurrences':len(p1['all_key_change_occurrences']),
           'P1_reverse_station_warnings':len(p1['same_lane_reverse_station_warnings']),
           'P2_paths':p2['candidate_paths'], 'P2_unique_edges':p2['unique_longitudinal_edges_required'],
           'P2_contact_covered_edges':p2['unique_required_edges_with_local_contact'],
           'copied_sources':copied,'missing_optional':missing})
    target=out.with_suffix('.tar.gz')
    need(not target.exists(),'Refuse overwrite archive')
    with tarfile.open(str(target),'w:gz') as tf:tf.add(str(out),arcname=out.name)
    print('COLLECTION_COMPLETE=1')
    print('P1_UNIQUE_TRANSITIONS='+str(len(p1['unique_directed_key_changes'])))
    print('P1_TOTAL_TRANSITION_OCCURRENCES='+str(len(p1['all_key_change_occurrences'])))
    print('P2_UNIQUE_REQUIRED_EDGES='+str(p2['unique_longitudinal_edges_required']))
    print('P2_SAMPLED_REQUIRED_EDGES='+str(p2['unique_required_edges_with_local_contact']))
    print('UPLOAD_FILE='+str(target))
    print('UPLOAD_SHA256='+digest(target))
    print('UPLOAD_BYTES='+str(target.stat().st_size))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    try:collect(args.root,args.out)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:
        print('INPUT_COLLECTION_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(2)
