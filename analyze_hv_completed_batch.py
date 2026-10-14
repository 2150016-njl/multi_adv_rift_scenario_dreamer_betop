"""Additional offline native funnel statistics; never contacts CARLA."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def positive(actor):
    return actor['forward_fov'] and actor['native']['native_candidate']


def compute(experiment):
    result = {'input_sha256': {}, 'models': {}}
    all_rows = {}
    for model in ('v2', 'v3'):
        path = experiment/(model+'_native')/'rows.jsonl'
        result['input_sha256'][model] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = read(path)
        all_rows[model] = {r['record_index']: r for r in rows}
        cohorts = {}
        for name in ('raw_pool', 'selected_gt', 'generated'):
            actors = [a for r in rows for a in r[name]]
            front = [a for a in actors if a['forward_fov']]
            eligible = [a for a in front if a['native']['native_candidate']]
            cohorts[name] = {
                'actors': len(actors), 'front': len(front),
                'x_nonpositive': sum(a['local_state'][0]<=0 for a in actors),
                'front_center_driving': sum(a['road_geometry']['center_driving'] for a in front),
                'front_preview_driving': sum(a['road_geometry']['preview_driving'] for a in front),
                'front_native': len(eligible),
                'front_native_occupancy_clear': sum(a['occupancy_2d']['clear'] for a in eligible),
                'front_native_candidate_covered': sum(a['candidate_geometry_supported'] is True for a in eligible),
                'front_native_approach_covered': sum(a['v6_approach_geometry_supported'] is True for a in eligible),
                'front_native_approach_unknown': sum(a['v6_approach_geometry_supported'] is None for a in eligible),
                'funnel': dict(Counter('preview_false' if not a['road_geometry']['preview_driving'] else
                    'native_false' if not a['native']['native_candidate'] else 'native_true' for a in front)),
                'scenes_at_least_one': sum(any(positive(a) for a in r[name]) for r in rows),
                'scenes_at_least_two': sum(sum(positive(a) for a in r[name])>=2 for r in rows),
                'front_center_preview_native_cross': dict(Counter(
                    '%d%d%d'%(a['road_geometry']['center_driving'],a['road_geometry']['preview_driving'],
                               a['native']['native_candidate']) for a in front))}
        towns = {}
        for town in sorted({r['town'] for r in rows}):
            group = [r for r in rows if r['town']==town]
            towns[town] = {'scenes':len(group), **{name:sum(any(positive(a) for a in r[name]) for r in group)
                          for name in ('raw_pool','selected_gt','generated')}}
        raw_has = [r for r in rows if any(positive(a) for a in r['raw_pool'])]
        raw_empty = [r for r in rows if not any(positive(a) for a in r['raw_pool'])]
        rejections=Counter();interaction_failures=Counter()
        for r in rows:
            for actor in r['generated']:
                if not (actor['forward_fov'] and actor['road_geometry']['preview_driving'] and
                        not actor['native']['native_candidate']):continue
                debug=actor['native']['candidate_debug']
                rejections.update({k:v for k,v in debug['rejection_counts'].items() if v})
                for detail in debug['interaction_debug_by_actor'].values():
                    best=detail['best_distance_diff']
                    label=('no_targets_ahead' if detail['num_targets_ahead']==0 else
                           'best_unknown' if best is None else 'best_diff_gt20' if best>20 else 'best_diff_le20')
                    interaction_failures[label]+=1
        result['models'][model] = {'scenes':len(rows),'cohorts':cohorts,'towns':towns,
            'front_preview_native_rejections':dict(rejections),
            'interaction_failure_details':dict(interaction_failures),
            'front_center_preview_native_occupancy_scenes':sum(any(positive(a) and
                a['road_geometry']['center_driving'] and a['road_geometry']['preview_driving'] and
                a['occupancy_2d']['clear'] for a in r['generated']) for r in rows),
            'generated_success_given_saved_front_positive': [sum(any(positive(a) for a in r['generated']) for r in raw_has),len(raw_has)],
            'generated_success_without_saved_front_positive': [sum(any(positive(a) for a in r['generated']) for r in raw_empty),len(raw_empty)],
            'candidate_condition_sources':dict(Counter(r['candidate_condition_source'] for r in rows)),
            'known_graph_roots':sum(r['v6_condition']['topology_route_roots_known'] for r in rows),
            'scenes_with_approaches':sum(bool(r['v6_condition']['approaches']) for r in rows)}
    result['selected_gt_identical_between_models'] = (
        set(all_rows['v2'])==set(all_rows['v3']) and all(
        all_rows['v2'][idx]['selected_gt']==all_rows['v3'][idx]['selected_gt'] for idx in all_rows['v2']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compute(args.experiment),indent=2,allow_nan=False),encoding='utf-8')
    print('REPORT='+str(args.output))
