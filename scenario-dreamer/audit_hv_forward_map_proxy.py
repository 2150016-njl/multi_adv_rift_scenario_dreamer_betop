#!/usr/bin/env python3
"""Compare saved S1 outputs and GT in x>0 FOV using centerline PROXIES.

No simulator, generation, state rewriting or off-road/native certification.
Align saved samples to original records by index AND scene_id; require S1.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def rows(path):
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def lane_segments(condition):
    payload = condition.get('map') or {}
    lines = payload.get('lane_polylines') or [
        l.get('centerline', []) for l in payload.get('lanes_v3', [])]
    starts, ends = [], []
    for line in lines:
        p = np.asarray(line, dtype=float)
        if p.ndim != 2 or p.shape[1] < 2 or len(p) < 2 or not np.isfinite(p).all():
            continue
        p = p[:, :2]
        valid = np.linalg.norm(p[1:]-p[:-1], axis=1) > 1.e-8
        starts.extend(p[:-1][valid])
        ends.extend(p[1:][valid])
    if not starts:
        return None
    return np.asarray(starts), np.asarray(ends)


def distances(points, segments):
    if segments is None:
        return np.full(len(points), np.nan)
    a, b = segments
    d = b-a
    t = np.clip(((points[:, None, :]-a)*d).sum(axis=2)/(d*d).sum(axis=1), 0., 1.)
    return np.linalg.norm(points[:, None, :]-a-t[:, :, None]*d, axis=2).min(axis=1)


class Accumulator:
    def __init__(self):
        self.counts = Counter()
        self.front_distances = []

    def add(self, states, segments, half_fov=32.):
        a = np.asarray(states, dtype=float)
        if a.shape != (2, 7) or not np.isfinite(a).all():
            raise ValueError('Expected two finite seven-dimensional raw states')
        front = (a[:, 0] > 0.) & (a[:, 0] <= half_fov) & (np.abs(a[:, 1]) <= half_fov)
        d = distances(a[:, :2], segments)
        self.counts['records'] += 1
        self.counts['actors'] += 2
        self.counts['x_nonpositive_actors'] += int((a[:, 0] <= 0.).sum())
        self.counts['forward_fov_actors'] += int(front.sum())
        self.counts['both_in_forward_fov_records'] += int(front.all())
        self.counts['at_least_one_in_forward_fov_records'] += int(front.any())
        self.counts['forward_map_unknown_actors'] += int((front & ~np.isfinite(d)).sum())
        self.front_distances.extend(d[front & np.isfinite(d)].tolist())
        for threshold in (1.25, 1.75, 3., 5.):
            supported = front & (d <= threshold)
            k = str(threshold)
            self.counts['forward_centerline_gt_'+k+'_actors'] += int((front & (d > threshold)).sum())
            self.counts['at_least_one_forward_centerline_le_'+k+'_records'] += int(supported.any())
            self.counts['both_forward_centerline_le_'+k+'_records'] += int(supported.all())
        self.counts['forward_stopped_le_0p1_actors'] += int((front & (a[:, 3] <= .1)).sum())
        self.counts['forward_stopped_centerline_gt_1p75_actors'] += int(
            (front & (a[:, 3] <= .1) & (d > 1.75)).sum())

    def result(self):
        c = self.counts
        return {'counts': dict(c),
                'both_forward_fov_rate': c['both_in_forward_fov_records']/c['records'],
                'forward_centerline_gt_1p75_rate': (
                    c['forward_centerline_gt_1.75_actors'] /
                    (c['forward_fov_actors']-c['forward_map_unknown_actors'])
                    if c['forward_fov_actors'] > c['forward_map_unknown_actors'] else None),
                'forward_centerline_distance_m': ({
                    'p50': float(np.percentile(self.front_distances, 50)),
                    'p90': float(np.percentile(self.front_distances, 90)),
                    'p99': float(np.percentile(self.front_distances, 99)),
                } if self.front_distances else None)}


def run(dataset, generated_files):
    generated = {}
    for path in generated_files:
        label = Path(path).stem
        if label in generated:
            raise ValueError('Repeated output label: '+label)
        samples = {}
        for r in rows(path):
            idx = r.get('record_idx', r.get('record_index'))
            sample = r.get('sample_idx', r.get('sample_index'))
            if not isinstance(idx, int) or sample != 0 or idx in samples:
                raise ValueError('Expected unique S1 index, sample_index=0: '+str(path))
            samples[idx] = r
        generated[label] = samples
    stats = {label: Accumulator() for label in ['gt']+list(generated)}
    n = 0
    for idx, record in enumerate(rows(dataset)):
        condition = record['condition']
        segments = lane_segments(condition)
        half = float(condition.get('fov', 64.))/2.
        stats['gt'].add(record['target']['generated_agent_states'], segments, half)
        for label, samples in generated.items():
            sample = samples.get(idx)
            if sample is None or sample['scene_id'] != record['scene_id']:
                raise ValueError('Dataset/sample alignment failure at '+str(idx))
            stats[label].add(sample['generated']['generated_agent_states'], segments, half)
        n += 1
    if n == 0 or any(len(s) != n for s in generated.values()):
        raise ValueError('Empty dataset or output/dataset count mismatch')
    return {'schema_version': 'hv_forward_map_proxy_comparison_v1',
            'dataset': str(dataset), 'dataset_sha256': digest(dataset),
            'generated_sources': {str(p): digest(p) for p in generated_files},
            'selection': 'all_records', 'num_records': n,
            'definition': '0<x<=fov/2, abs(y)<=fov/2; center-distance proxies only',
            'results': {k: v.result() for k, v in stats.items()},
            'limitations': [
                'Fixed centerline thresholds are NOT actual road polygons or lane widths.',
                'Missing map is unknown, never a pass.',
                'No CARLA preview, raw spawn, elevation, OBB or native interaction verification.',
                'Different generated states need not match the original GT lane.',
                'Stopped states are not automatically parked or invalid.',
            ]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-jsonl', required=True)
    p.add_argument('--generated-jsonl', action='append', required=True)
    p.add_argument('--output-json', required=True)
    args = p.parse_args()
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError('Refuse to overwrite evidence: '+str(output))
    report = run(args.dataset_jsonl, args.generated_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['results'], indent=2))


if __name__ == '__main__':
    main()
