#!/usr/bin/env python3
"""Offline source-complete P2 lane-candidate explanation; GT only read after candidates."""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path

def lane(v):
    return tuple(int(x) for x in v)
def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--topology',required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--lateral',required=True);p.add_argument('--approaches',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    g,geo,lat,can=map(read,[a.topology,a.geometry,a.lateral,a.approaches])
    if Path(a.out).exists():raise SystemExit('REFUSE_OVERWRITE')
    if not(g['town']==geo['town']==lat['town']==can['town']):raise SystemExit('TOWN_MISMATCH')
    if not (g['map_sha256']==geo['map_sha256']==lat['opendrive_sha256']):raise SystemExit('MAP_SHA_MISMATCH')
    ranges={lane(x['lane']) for x in geo['sampled_lane_station_ranges']}
    nodes={lane(x) for x in g['node_keys']}
    edges=collections.defaultdict(set)
    for e in g['directed_edges']:edges[lane(e['from'])].add(lane(e['to']))
    candidate_by={int(r['record_index']):r for r in can['records']}
    records=[];cnt=collections.Counter()
    for r in g['records']:
        idx=int(r['record_index']);c=candidate_by.get(idx)
        if c is None:continue
        route={lane(z['lane']) for z in c['anchors']}
        generated={lane(z['lane']) for z in c['approaches']}
        for gt in r.get('gt_targets',[]):
            k=gt.get('actor_lane')
            if k is None:
                reason='gt_lane_unverified';key=None
            else:
                key=lane(k)
                if key in generated:reason='candidate_lane_hit'
                elif key not in nodes:reason='actor_lane_missing_from_graph'
                elif key not in ranges:reason='actor_lane_missing_from_geometry_sampler'
                elif not route:reason='no_anchor_on_graph'
                else:reason='directed_or_lateral_path_missing'
            cnt[reason]+=1
            records.append({'record_index':idx,'actor_id':gt['actor_id'],
                'actor_lane':list(key) if key else None,'reason':reason,
                'sampled_actor_lane':key in ranges if key else None,
                'candidate_count':len(generated),
                'anchor_count':len(route),
                'anchor_lanes_missing_samples': [list(x) for x in sorted(route-ranges)]})
    result={'town':g['town'],'total_gt_diagnostic':len(records),'reason_counts':dict(sorted(cnt.items())),
       'graph_lanes':len(nodes),'sampled_station_lanes':len(ranges),
       'graph_lanes_without_station_samples':len(nodes-ranges),
       'lateral_intervals':len(lat['intervals']),
       'crossing_anchor_candidates':sum(len(x['crossing_anchor_candidates']) for x in geo['records']),
       'cases':records,'not_formal_gt110_recall':True}
    Path(a.out).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('P2_DIAGNOSTIC_GT='+str(len(records)))
    print('P2_DIAGNOSTIC_REASONS='+json.dumps(result['reason_counts'],ensure_ascii=False))
    print('P2_GRAPH_LANES_WITHOUT_S_SAMPLES='+str(len(nodes-ranges)))
    print('P2_DIAGNOSTIC_OUTPUT='+str(a.out))
if __name__=='__main__':main()
