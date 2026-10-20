#!/usr/bin/env python3
"""Summarize immutable P1 endpoint probes and P2 station-screen; no CARLA/GT reads beyond supplied files."""
import argparse,collections,csv,json,sys
from pathlib import Path

def main():
    a=argparse.ArgumentParser();a.add_argument('--p2-screen',required=True)
    a.add_argument('--p2-gt26',required=True);a.add_argument('--p1-town12')
    a.add_argument('--p1-town13');a.add_argument('--out-dir',required=True)
    args=a.parse_args();o=Path(args.out_dir)
    if o.exists():raise SystemExit('REFUSE_OVERWRITE_OUTPUT_DIR')
    s=json.loads(Path(args.p2_screen).read_text());g=json.loads(Path(args.p2_gt26).read_text())
    if s.get('schema')!='p2_gtfree_station_necessary_screen_v2' or g.get('schema')!='p2_posthoc_station_screen_diagnostic_v2':
        raise SystemExit('INCORRECT_P2_SCHEMA')
    if s['town']!=g['town']:raise SystemExit('TOWN_MISMATCH')
    p2=[]
    for rec in s['records']:
        for c in rec['candidate_station_checks']:
            r=c['station_check']
            p2.append({'record_index':rec['record_index'],'origin_lane':str(c['lane']),
                       'anchor_lane':str(c['anchor_lane']),
                       'anchor_route_index':None if c['ego_anchor'] is None else c['ego_anchor']['ego_route_index'],
                       'anchor_s':None if c['ego_anchor'] is None else c['ego_anchor']['ego_s'],
                       'station_status':r['status'],'screen_possible':r['station_screen_possible'],
                       'lateral_count':r.get('lateral_change_count',0),
                       'unverified_longitudinal':r.get('longitudinal_station_contact_proven') is False})
    p1=[]
    for town,path in [('Town12',args.p1_town12),('Town13',args.p1_town13)]:
        if not path:continue
        d=json.loads(Path(path).read_text());
        if d.get('schema')!='p1_endpoint_xml_link_diagnostic_v2' or d.get('town')!=town:
            raise SystemExit('INCORRECT_P1_EVIDENCE_SCHEMA')
        p1.append({'town':town,'record_index':d['record_index'],
                   'old_geometry_statuses':d['old_census_source_statuses'],
                   'original_anchor_source_validated':d['each_anchor_meets_old_geometry_classification'],
                   'direct_junction_xml_candidates':len(d['xml_link_evidence_without_short_distance_filter']['direct_junction_connections']),
                   'original_od_guard_status':d['old_guard_link_result']['status'],
                   'complete_directed_connection_proven':False})
    counts=dict(collections.Counter(v['station_status'] for v in p2))
    result={'schema':'p1p2_station_candidate_review_v1',
            'P2_lane_candidate_count':len(p2),'P2_station_check_counts':counts,
            'P2_GT26_necessary_screen':g['station_screen_counts'],
            'P2_GT26_actor_record_observations':g['GT_observations'],
            'P2_formal_110_recall_proven':False,
            'P2_exact_longitudinal_contacts_missing':sum(v['unverified_longitudinal'] for v in p2),
            'P1_live_probes':p1,'P1_95_scientific_PASS':False,
            'Model3_physical_training_READY':False}
    o.mkdir(parents=True)
    (o/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (o/'p2_station_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(p2[0]) if p2 else ['record_index']);w.writeheader();w.writerows(p2)
    with (o/'p2_gt26_station.csv').open('w',newline='') as f:
        rows=g['cases'];w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['record_index']);w.writeheader();w.writerows(rows)
    (o/'P1_P2_conclusions.md').write_text(
        '# P1/P2 科学证据报告\n\n'
        'P1 新端点证据: '+str(len(p1))+' 条；未证实的连接保持 UNKNOWN。\n\n'
        'P2 检验的是 sampled station-order 的必要条件，不是完整连续 approach。\n\n'
        'P2 GT26 筛查: '+str(g['station_screen_counts'])+'，正式 110 GT 召回未证明。\n\n'
        '特别注意: 纵向 contact s、lane-change 连续轨迹、actor 初始 s/航向/速度和碰撞/Model3 物理未验证。\n',
        encoding='utf-8')
    print('P1P2_ANALYSIS='+str(o/'summary.json'))
    print('P1P2_COUNTS='+json.dumps(result,ensure_ascii=False))
    return 0
if __name__=='__main__':sys.exit(main())
