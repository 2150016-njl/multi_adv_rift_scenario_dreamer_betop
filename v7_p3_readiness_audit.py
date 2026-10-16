#!/usr/bin/env python3
"""Conservative READ-ONLY P0/P1/P2 pre-P3 gate on frozen V7 Round3 artifacts.

This is an audit, not a relabeling step: unknown != negative, spawn accepted !=
collision-free, and map graph station connectivity != Model3 physical feasibility.
Returns 0 only when the declared strict evidence contract is met; otherwise 2.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def readl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def audit(dataset, calibration, maps):
    dataset=Path(dataset);calibration=Path(calibration);maps=Path(maps)
    d=read(dataset/'summary.json');inputs=read(dataset/'protocol_sha256.json')
    protocol={name:sha(dataset/name)==expected for name,expected in inputs.items()}
    conditions=readl(dataset/'conditions.jsonl');targets=readl(dataset/'targets.jsonl');oracles=readl(dataset/'oracles.jsonl')
    aligned=len(conditions)==len(targets)==len(oracles)==95 and all(
        c['record_id']==t['record_id']==o['record_id'] for c,t,o in zip(conditions,targets,oracles))
    provenance=all(c['provenance'].get('map_input_kind')=='calibrated_carla_map' for c in conditions)
    distinct_ids=len({c['record_id'] for c in conditions})==95
    manifest=read(maps/'map_merge_manifest.json')
    # Manifest content differs across revisions; dataset code is authoritative
    # for binding maps, and we only test paths that actually exist.
    map_count=sum(p.is_file() for p in maps.glob('record_*_map.json'))
    by_manifest={int(row['record_index']):row for row in manifest.get('records',[])}
    map_bindings=[]
    for c in conditions:
        prov=c['provenance']; idx=int(prov['record_index'])
        fp=maps/('record_%s_map.json'%idx)
        source=read(fp) if fp.is_file() else {}
        digest=sha(fp) if fp.is_file() else None
        record=by_manifest.get(idx,{})
        ok=bool(digest and digest==prov.get('calibrated_map_sha256')==record.get('derived_sha256')
                and source.get('record_index')==idx
                and source.get('town')==prov.get('town')
                and source.get('snapshot_sha256')==prov.get('source_sha256')
                and source.get('opendrive_sha256')==prov.get('opendrive_sha256')
                and prov.get('observation_evidence',{}).get('snapshot_status')=='sha_verified')
        map_bindings.append({'record_index':idx,'sha_and_identity_valid':ok})
    rows=readl(dataset/'rows.jsonl')
    front=[r for r in rows if r.get('cohort')=='selected_gt' and r.get('front') is True]
    raw=[r for r in rows if r.get('cohort')=='raw_front_positive' and r.get('front') is True]
    states=Counter(r.get('station_audit',{}).get('status','missing') for r in front)
    by_record=defaultdict(list)
    for r in front:by_record[int(r['record_index'])].append(r)
    native_missing=[{'record_index':r['record_index'],'actor_id':r['actor_id'],
                     'edges':r.get('native_route_missing_edges')}
                    for r in front if r.get('native_route_missing_edges')]
    unknown=[{'record_index':r['record_index'],'actor_id':r['actor_id'],
              'reason':r.get('station_audit',{}).get('reason')}
             for r in front if r.get('station_audit',{}).get('status')!='supported']
    # Inspect route coverage against the actual 100 m CARLA export extent.
    route_gap_detail=[]
    route_gap_totals={'inside_99m':0,'boundary_99_to_101m':0,'outside_101m':0}
    for condition in conditions:
        count=Counter()
        for waypoint in condition['condition']['ego_route']['waypoints']:
            if waypoint.get('lane_matches'):
                continue
            xyz=waypoint['local_xyz'];radius=math.hypot(float(xyz[0]),float(xyz[1]))
            key=('inside_99m' if radius<=99. else
                 'outside_101m' if radius>101. else 'boundary_99_to_101m')
            count[key]+=1
            route_gap_totals[key]+=1
        if count:
            route_gap_detail.append({'record_index':condition['provenance']['record_index'],
                                     'counts':dict(count)})
    alignment=readl(dataset/'map_alignment_rows.jsonl')
    unpaired=[{'record_index':r['record_index'],'unmatched_points':r['route_points_without_lane_match']}
              for r in alignment if r.get('route_points_without_lane_match')]
    outrows=readl(calibration/'rows.jsonl')
    ix={int(r['record_index']):r for r in outrows}
    required={2325,155,1532,160,40}
    p1=[]
    for rec in sorted(required):
        r=ix.get(rec,{})
        cohorts=r.get('cohorts') or {}
        gt=cohorts.get('historical_gt') or {}
        sedan=cohorts.get('fixed_sedan_same_raw_pose') or {}
        targets_s=[a for a in sedan.get('attempts',[]) if a.get('role')=='target']
        background_gt=[a for a in gt.get('attempts',[]) if a.get('role')!='target']
        background_s=[a for a in sedan.get('attempts',[]) if a.get('role')!='target']
        compar=r.get('native_comparisons') or []
        measure=r.get('fixed_blueprint_box') or {}
        probe=r.get('map_identity_probe') or {}
        geom=(sedan.get('geometry_3d') or {})
        static_overlap=[]
        for t in targets_s:
            g=geom.get(str(t['source_actor_id']),geom.get(t['source_actor_id'],{}))
            if g.get('static_3d_box_overlaps'):
                static_overlap.append({'actor_id':t['source_actor_id'],
                                       'static_box_overlaps':g['static_3d_box_overlaps'],
                                       'mesh_collision_checked':g.get('static_mesh_collision_checked')})
        p1.append({'record_index':rec,'status':r.get('status','missing'),
                   'map_identity_classification':probe.get('classification'),
                   'errors':r.get('errors',[]),
                   'native_comparison_count':len(compar),
                   'native_disagreements':sum(bool(a.get('differences')) for a in compar),
                   'model3_measured_box':measure.get('size_lwh_m'),
                   'gt_spawn':{'accepted':sum(a.get('accepted') is True for a in gt.get('attempts',[])),
                               'total':len(gt.get('attempts',[]))},
                   'sedan_spawn':{'accepted':sum(a.get('accepted') is True for a in sedan.get('attempts',[])),
                                  'total':len(sedan.get('attempts',[]))},
                   'background_gt_rejected':[a['source_actor_id'] for a in background_gt if not a.get('accepted')],
                   'background_sedan_rejected':[a['source_actor_id'] for a in background_s if not a.get('accepted')],
                   'target_model3_rejected':[a['source_actor_id'] for a in targets_s if not a.get('accepted')],
                   'static_box_overlap_warnings':static_overlap,
                   'cleanup_errors':gt.get('cleanup_errors',[])+sedan.get('cleanup_errors',[]),
                   'raw_pose_unchanged':bool(gt and sedan and all(a.get('raw_request_fidelity') is True
                                  for c in (gt,sedan) for a in c.get('attempts',[]))),
                   'cohorts_complete':gt.get('whole_scene_accepted') is True and
                                      sedan.get('whole_scene_accepted') is True})
    p0_scoped=bool(all(protocol.values()) and aligned and distinct_ids and provenance and map_count==95
                   and len(map_bindings)==95 and all(x['sha_and_identity_valid'] for x in map_bindings))
    # No physical/lane-chart proof is supplied by this artifact set. Never infer it
    # from graph reachability or a lack of collision sensor observations.
    p1_full=bool(len(ix)==5 and all(x['status']=='calibration_observed' and
        x['map_identity_classification']=='verified_same_live_map_and_expected_sha' and
        x['native_comparison_count']>0 and x['native_disagreements']==0 and
        x['model3_measured_box'] is not None and x['cohorts_complete'] and
        x['raw_pose_unchanged'] and not x['cleanup_errors'] for x in p1))
    p2_graph=bool(len(front)==110 and states.get('supported',0)==110 and len(by_record)==82 and
                  sum(len(v)>=2 and all(x['station_audit']['status']=='supported' for x in v)
                      for v in by_record.values())==28)
    p2_fidelity=not native_missing and route_gap_totals['inside_99m']==0 and route_gap_totals['boundary_99_to_101m']==0
    gates={'p0_frozen_95_provenance':p0_scoped,
           'p0_full_s1_and_model_metadata':False,  # requires external provenance evidence
           'p1_full_native_and_spawn':p1_full,
           'p1_model3_collision_geometry':False, # requires collision/road certificate
           'p2_station_graph_110_of_110':p2_graph,
           'p2_trace_fidelity_and_route_coverage':p2_fidelity,
           'p2_continuous_model3_lane_chart':False} # P3 prerequisite not yet implemented
    report={'status':'blocked' if not all(gates.values()) else 'ready',
            'source':'read_only_round3_evidence','dataset':str(dataset),'calibration':str(calibration),
            'maps':str(maps),'gates':gates,'protocol_sha256_valid':protocol,
            'record_identity_aligned':aligned,'map_count':map_count,'map_bindings_verified':sum(x['sha_and_identity_valid'] for x in map_bindings),
            'map_binding_mismatches':[x for x in map_bindings if not x['sha_and_identity_valid']],
            'manifest_entries_available':bool(manifest),
            'selected_front_denominator':len(front),'selected_front_station_status':dict(states),
            'front_positive_scenes':len(by_record),'raw_front_count':len(raw),
            'selected_unknown':unknown,'native_route_edges_missing':native_missing,
            'ego_route_unmatched':unpaired,'route_unmatched_distance_buckets':route_gap_totals,
            'route_unmatched_distance_by_record':route_gap_detail,'p1_records':p1,
            'critical_caveats':['No GT/actor states are changed by this audit.',
               'Spawn rejection reason cannot be established from try_spawn_actor returning None.',
               'A static bbox overlap is not proof of mesh collision.',
               'Finite station-graph reachability is not a drivable, collision-free trajectory.',
               'The P0 full-test weight+metadata gate requires original sources not in this archive.']}
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,type=Path)
    ap.add_argument('--calibration',required=True,type=Path)
    ap.add_argument('--maps',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    result=audit(a.dataset,a.calibration,a.maps)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print('STATUS',result['status'])
    for k,v in result['gates'].items():print('GATE',k,'PASS' if v else 'BLOCK')
    print('REPORT',a.output)
    raise SystemExit(0 if result['status']=='ready' else 2)

if __name__=='__main__':main()
