#!/usr/bin/env python3
"""Read-only forensic accounting of P2 Town15 directed-contact sampling.

Compares frozen candidate lane paths, CARLA local waypoint.next() contact samples,
and the optional station-screen result. Never calls CARLA, never reads GT for
candidate generation, and never claims full lane-path or physical feasibility.
Python 3.8+; standard library.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def die(msg):
    raise SystemExit('P2_CONTACT_REVIEW_FAIL_CLOSED=' + msg)


def read(path):
    p = Path(path)
    if not p.is_file(): die('MISSING_INPUT=' + str(p))
    with p.open(encoding='utf-8') as f: return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def lkey(seq):
    if not isinstance(seq, (list, tuple)) or len(seq) != 3:
        die('INVALID_LANE_KEY=' + str(seq))
    return tuple(int(x) for x in seq)


def edge_key(a, b):
    return lkey(a), lkey(b)


def aslabel(e):
    return '%d:%d:%d -> %d:%d:%d' % (e[0] + e[1])


def check_paths(approaches):
    if approaches.get('GT_used_in_make') is not False:
        die('APPROACHES_GT_ISOLATION_NOT_CONFIRMED')
    unique = set()
    refs = collections.Counter()
    paths = []
    for record in approaches.get('records', []):
        idx = int(record['record_index'])
        for j, candidate in enumerate(record.get('approaches', [])):
            lane_path = candidate['lane_path_to_anchor']
            kinds = candidate['path_edge_types']
            if len(kinds) != len(lane_path) - 1:
                die('PATH_LENGTH_MISMATCH_RECORD_%d' % idx)
            longitudinal = []
            for k, t in enumerate(kinds):
                if t == 'longitudinal':
                    e = edge_key(lane_path[k], lane_path[k+1])
                    longitudinal.append(e)
                    unique.add(e)
                    refs[e] += 1
            paths.append({'record_index': idx, 'path_ordinal': j,
                          'origin_lane': lkey(candidate['lane']),
                          'anchor_lane': lkey(candidate['anchor_lane']),
                          'long_edges': longitudinal})
    return unique, refs, paths


def parse_screen(screen, approaches, contacts_path):
    if screen is None: return None
    if screen.get('schema') != 'p2_gtfree_station_necessary_screen_v2':
        die('WRONG_SCREEN_SCHEMA')
    if not screen.get('CARLA_contact_samples_used'):
        die('SCREEN_WAS_NOT_BUILT_WITH_CONTACTS')
    if screen.get('contact_sha256') != sha256(contacts_path):
        die('SCREEN_CONTACT_SHA_MISMATCH')
    approach_rows = {int(r['record_index']): r['approaches']
                     for r in approaches['records']}
    lines = []
    for record in screen.get('records', []):
        rid = int(record['record_index'])
        checks = record['candidate_station_checks']
        if rid not in approach_rows or len(checks) != len(approach_rows[rid]):
            die('SCREEN_CANDIDATE_ORDER_COUNT_MISMATCH_%d' % rid)
        for j, row in enumerate(checks):
            original = approach_rows[rid][j]
            if row.get('lane_path_to_anchor') != original.get('lane_path_to_anchor') or row.get('path_edge_types') != original.get('path_edge_types'):
                die('SCREEN_PATH_ORDER_MISMATCH_%d_%d' % (rid, j))
            ck = row['station_check']
            edges = ck.get('edges', [])
            longitudinal = [e for e in edges if e.get('type') == 'longitudinal']
            num_positive = sum(int(e.get('local_CARLA_next_contact_samples_compatible',0)) > 0 for e in longitudinal)
            lines.append((rid,j,ck.get('status'),len(longitudinal),num_positive))
    return lines


def run(args):
    contacts = read(args.contacts)
    approaches = read(args.approaches)
    topology = read(args.topology)
    screen = read(args.screen) if args.screen else None
    if contacts.get('schema') != 'p2_CARLA_local_directed_contact_samples_v1':
        die('WRONG_CONTACT_SCHEMA')
    if contacts.get('GT_used') is not False:
        die('CONTACT_SOURCE_NOT_GT_FREE')
    if not (contacts.get('town') == approaches.get('town') == topology.get('town')):
        die('CONTACT_APPROACH_TOPOLOGY_TOWN_MISMATCH')
    if contacts.get('map_sha256') != topology.get('map_sha256'):
        die('CONTACT_TOPOLOGY_MAP_SHA_MISMATCH')
    if contacts.get('topology_sha256') != sha256(args.topology):
        die('CONTACT_TOPOLOGY_FILE_SHA_MISMATCH')
    if approaches.get('source_sha256',{}).get('topology') != sha256(args.topology):
        die('APPROACHES_TOPOLOGY_FILE_SHA_MISMATCH')
    if screen is not None and screen.get('input_sha256',{}).get('topology') != sha256(args.topology):
        die('SCREEN_TOPOLOGY_FILE_SHA_MISMATCH')
    if screen is not None and screen.get('town') != contacts.get('town'):
        die('CONTACT_SCREEN_TOWN_MISMATCH')
    unique, refs, paths = check_paths(approaches)
    contact_by = collections.defaultdict(list)
    for row in contacts['samples']:
        e = edge_key(row['from_lane'],row['to_lane'])
        contact_by[e].append(row)
    covered = unique & set(contact_by)
    missing = unique - set(contact_by)
    rows = []
    for e in sorted(unique):
        rows.append({'from_lane': str(list(e[0])), 'to_lane': str(list(e[1])),
                     'path_references': refs[e], 'has_CARLA_local_contact_sample': bool(contact_by.get(e)),
                     'sample_count': len(contact_by.get(e, []))})
    screen_rows = parse_screen(screen, approaches, args.contacts)
    paths_with_long = [x for x in paths if x['long_edges']]
    no_contact = sum(all(e not in contact_by for e in p['long_edges']) for p in paths_with_long)
    any_contact = sum(any(e in contact_by for e in p['long_edges']) for p in paths_with_long)
    all_contact = sum(all(e in contact_by for e in p['long_edges']) for p in paths_with_long)
    summary = {'schema':'p2_contact_effectiveness_readonly_v1',
               'town':contacts['town'],
               'map_sha256':contacts.get('map_sha256'),
               'candidate_generation_GT_free':True,
               'candidate_paths':len(paths),
               'candidate_paths_with_longitudinal_edges':len(paths_with_long),
               'candidate_unique_longitudinal_edges':len(unique),
               'candidate_total_longitudinal_edge_references':sum(refs.values()),
               'CARLA_census_unique_contact_edges':len(contact_by),
               'CARLA_census_sample_rows':len(contacts['samples']),
               'candidate_unique_edges_with_any_contact':len(covered),
               'candidate_unique_edges_without_contact':len(missing),
               'candidate_paths_with_any_contact_edge':any_contact,
               'candidate_paths_with_all_long_edges_locally_sampled':all_contact,
               'candidate_paths_with_no_contact_edges':no_contact,
               'screen_status_is_not_equivalent_to_contact_coverage':True,
               'actual_station_compatibility_requires_screen':True,
               'continuous_connectivity_proven':False,
               'fixed_110_GT_recall_proven':False,
               'not_physical_spawn_proven':True,
               'input_sha256':{'contacts':sha256(args.contacts),'approaches':sha256(args.approaches),'topology':sha256(args.topology)}}
    if screen_rows is not None:
        if len(screen_rows) != len(paths):die('SCREEN_CANDIDATE_TOTAL_MISMATCH')
        summary['screen_paths_with_at_least_one_compatible_station_contact'] = sum(x[4]>0 for x in screen_rows)
        summary['screen_paths_with_all_longitudinal_edges_compatible'] = sum(x[3]>0 and x[3]==x[4] for x in screen_rows)
        summary['screen_paths_with_long_edges_but_no_compatible_station_contacts'] = sum(x[3]>0 and x[4]==0 for x in screen_rows)
        summary['screen_status_counts'] = dict(collections.Counter(x[2] for x in screen_rows))
        summary['input_sha256']['screen'] = sha256(args.screen)
    out = Path(args.output_dir)
    if out.exists():die('REFUSE_OVERWRITE_OUTPUT_DIR='+str(out))
    out.mkdir(parents=True)
    with (out/'summary.json').open('x',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,sort_keys=True,ensure_ascii=False)
        f.write('\n')
    with (out/'edge_coverage.csv').open('x',newline='',encoding='utf-8') as f:
        dw = csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['from_lane'])
        dw.writeheader();dw.writerows(rows)
    with (out/'missing_edges.txt').open('x',encoding='utf-8') as f:
        for e in sorted(missing,key=lambda e:-refs[e]):
            f.write('%s\tpath_references=%d\n'%(aslabel(e),refs[e]))
    print('P2_CONTACT_REVIEW_SUMMARY='+str(out/'summary.json'))
    print('P2_CONTACT_REVIEW_COUNTS='+json.dumps({k:v for k,v in summary.items() if k not in ('input_sha256','map_sha256')},sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('topology','contacts','approaches','output_dir'):
        ap.add_argument('--'+key.replace('_','-'),required=True)
    ap.add_argument('--screen',help='optional GT-free station_with_contacts.json')
    args=ap.parse_args()
    return run(args)

if __name__=='__main__':
    raise SystemExit(main())
