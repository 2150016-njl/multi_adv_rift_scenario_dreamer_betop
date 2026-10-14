#!/usr/bin/env python3
"""Read-only attribution of map -> candidate -> chart support, before training.

Supports original V3/V4C records and the already-filtered V5 records. Never
creates candidates from targets, changes states, samples a generator, or claims
CARLA/native RIFT validity. Report both at-least-one and two-target coverage.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from high_value_init.v5_candidates import make_candidates, get_lanes, closest, _array
from high_value_init.v5_model import CARLAJointRawGenerator, candidate_tensors


STAGES = ('raw_map', 'candidate_geometry', 'candidate_upstream',
          'candidate_anchor', 'chart_roundtrip')


def analyze_record(record, model=None):
    condition = record.get('condition') or {}
    target = record.get('target') or {}
    states = target.get('generated_agent_states') or record.get('targets') or []
    labels = target.get('route_interaction_supervision') or []
    if len(states) != 2 or any(len(s) < 4 or not np.isfinite(s[:4]).all() for s in states):
        raise ValueError('Expected exactly two finite [x,y,yaw,speed] targets')
    candidates = make_candidates(condition)
    model = model if model is not None else CARLAJointRawGenerator(hidden=16)
    tensors = candidate_tensors(candidates) if candidates else None
    polylines = [p for p in (_array(l) for l in get_lanes(condition)) if p is not None]
    actors = []
    for slot, state in enumerate(states):
        flags = dict.fromkeys(STAGES, False)
        yaw = float(state[2])
        for poly in polylines:
            d, _, angle, _ = closest(poly, state[:2])
            error = abs(math.atan2(math.sin(yaw-angle), math.cos(yaw-angle)))
            if d <= 1.25 and error <= .38:
                flags['raw_map'] = True
                break
        label = labels[slot] if slot < len(labels) else {}
        anchor = (label or {}).get('conflict_point_local_xy')
        for index, c in enumerate(candidates):
            d, s, angle, _ = closest(np.asarray(c['polyline']), state[:2])
            error = abs(math.atan2(math.sin(yaw-angle), math.cos(yaw-angle)))
            if d > 1.25 or error > .38:
                continue
            flags['candidate_geometry'] = True
            if s >= c['anchor_s']-.25:
                continue
            flags['candidate_upstream'] = True
            if anchor is not None and math.dist(c['anchor_xy'], anchor[:2]) > 12.:
                continue
            flags['candidate_anchor'] = True
            # Test every feasible chart, including alternatives to the greedy
            # V5 assignment. This is an offline representability oracle only.
            try:
                z = model.target_latents(candidates, [state, state], [index, index])
                decoded = model.decode(z, [index, index], tensors).detach().numpy()[0]
            except (ValueError, RuntimeError):
                continue
            pos = np.abs(decoded[:2]-np.asarray(state[:2])).max()
            dh = abs(math.atan2(math.sin(decoded[2]-yaw), math.cos(decoded[2]-yaw)))
            if pos <= .03 and dh <= .01 and abs(decoded[3]-state[3]) <= .02:
                flags['chart_roundtrip'] = True
        flags['first_failure'] = next((k for k in STAGES if not flags[k]), None)
        actors.append(flags)
    return {'actors': actors, 'candidate_count': len(candidates),
            'native_anchor_available': len(labels) == 2 and all(
                (l or {}).get('conflict_point_local_xy') is not None for l in labels),
            'already_filtered_v5': 'targets' in record,
            'scene_id': record.get('scene_id'), 'town': record.get('town')}


def run(path, max_records=0):
    counts = Counter()
    by_town = {}
    examples = []
    digest = hashlib.sha256()
    model = CARLAJointRawGenerator(hidden=16)
    # Hash the processed bytes only. A prefix must not masquerade as a hash
    # or random sample of the full dataset.
    with Path(path).open('rb') as source:
        for line in source:
            if max_records > 0 and counts['records'] >= max_records:
                break
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            row = analyze_record(record, model)
            counts['records'] += 1
            counts['already_filtered_v5_records'] += int(row['already_filtered_v5'])
            counts['records_with_native_anchor'] += int(row['native_anchor_available'])
            town = by_town.setdefault(str(row['town']), Counter())
            town['records'] += 1
            for stage in STAGES:
                n = sum(a[stage] for a in row['actors'])
                counts[stage+'_actors'] += n
                counts[stage+'_at_least_one'] += int(n >= 1)
                counts[stage+'_both'] += int(n == 2)
                town[stage+'_both'] += int(n == 2)
            for a in row['actors']:
                counts['first_failure_'+str(a['first_failure'] or 'none')] += 1
            if not all(a['chart_roundtrip'] for a in row['actors']) and len(examples) < 12:
                examples.append(row)
    n = counts['records']
    if n == 0:
        raise ValueError('No records to audit')
    return {'schema_version': 'hv_support_chain_audit_v1',
            'source': str(path), 'processed_bytes_sha256': digest.hexdigest(),
            'selection': 'file_prefix' if max_records > 0 else 'all_records',
            'max_records': max_records, 'counts': dict(counts),
            'rates_per_record': {k: counts[k]/n for stage in STAGES
                                 for k in (stage+'_at_least_one', stage+'_both')},
            'by_town': {k: dict(v) for k, v in sorted(by_town.items())},
            'failure_examples': examples,
            'limits': [
                'No native RIFT or CARLA spawn certification.',
                'No score/OBB filter applied: support attribution, not training eligibility.',
                'Filtered V5 rows cannot estimate coverage of rejected V4C records.',
                'Without native anchors, candidate_anchor only repeats upstream support.',
                'First-failure bins use this diagnostic order, not causal independence.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--max-records', type=int, default=0, help='0=all; otherwise file prefix')
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing evidence: '+str(output))
    result = run(args.jsonl, args.max_records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('failure_examples', 'by_town')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
