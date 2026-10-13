#!/usr/bin/env python3
"""Read-only RIFT HD-map coordinate-mode diagnostic for strict HV dumps.

Uses the SAME CarlaMap / query_proximal_map_data implementation as RIFT.
Never spawns, changes states, resamples, registers CBVs, or modifies dumps.
"""
import argparse
import json
import math
from pathlib import Path


def _xy(point):
    if hasattr(point, 'x') and hasattr(point, 'y'):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def _segment_distance(point, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    sq = vx * vx + vy * vy
    u = 0.0 if sq < 1e-12 else min(1.0, max(0.0, ((point[0]-a[0])*vx+(point[1]-a[1])*vy)/sq))
    return math.hypot(point[0] - a[0] - u*vx, point[1] - a[1] - u*vy)


def _nearest(point, lines):
    best = float('inf')
    for line in lines:
        if len(line) < 2:
            continue
        for a, b in zip(line, line[1:]):
            best = min(best, _segment_distance(point, a, b))
    return None if not math.isfinite(best) else best


def _fmt(value):
    return '-' if value is None else '{:.2f}m'.format(value)


def _map_centerlines(map_api, x, y, mode, radius):
    # Import lazily so the geometry helpers remain testable without RIFT.
    from shapely.geometry import Point
    from nuplan_plugin.maps.maps_datatypes import SemanticMapLayer
    query_y = y if mode == 'legacy' else -y
    found = map_api.query_proximal_map_data(Point(x, query_y), radius)
    lines = []
    raw_count = 0
    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        for lane in found.get(layer, []) or []:
            raw_count += 1
            coords = getattr(lane, 'centerline', None)
            if coords is None:
                coords = getattr(lane, 'center_coords', None)
            if coords is None:
                continue
            points = []
            for p in coords:
                try:
                    px, py = _xy(p)
                    if math.isfinite(px) and math.isfinite(py):
                        points.append((px, py if mode == 'legacy' else -py))
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
            if len(points) >= 2:
                lines.append(points)
    return raw_count, lines


def analyze(file, seed, map_api, radius):
    data = json.loads(file.read_text(encoding='utf-8'))
    scene = data.get('scene') or {}
    proposal = data.get('proposal') or {}
    ego = scene.get('ego') or {}
    diag = scene.get('diagnostics') or {}
    if not ego:
        print('Seed {}: scene.ego missing'.format(seed))
        return
    x, y = float(ego['x']), float(ego['y'])
    yaw = math.radians(float(ego.get('yaw_deg', 0.0)))
    c, s = math.cos(yaw), math.sin(yaw)
    states = proposal.get('generated_agent_states') or []
    positions = [(x, y)]
    for v in states:
        if len(v) >= 2:
            lx, ly = float(v[0]), float(v[1])
            positions.append((x + c * lx - s * ly, y + s * lx + c * ly))
    print('\nSEED {} town={} ego=({:.2f}, {:.2f}) snapshot_lanes={} snapshot_mode={}'.format(
        seed, scene.get('town'), x, y, len(scene.get('map_lanes') or []), diag.get('map_coordinate_mode')))
    print('  snapshot_map_api_available={} raw_count={} serialized_count={} query_y={}'.format(
        diag.get('map_api_available'), diag.get('map_lane_query_raw_count'),
        diag.get('map_lane_serialized_count'), diag.get('map_lane_query_y')))
    for mode in ('legacy', 'right_handed'):
        raw, lines = _map_centerlines(map_api, x, y, mode, radius)
        values = [_fmt(_nearest(p, lines)) for p in positions]
        print('  {:12s} raw_lanes={} usable_lanes={} nearest_ego={} {}'.format(
            mode, raw, len(lines), values[0], ' '.join('HV{}={}'.format(i, z) for i, z in enumerate(values[1:]))))
    print('  WARNING: existence of lanes alone does not prove map-to-CARLA geometry agreement; check nearest_ego and HV distances.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', type=Path, required=True)
    ap.add_argument('--hd-map-dir', type=Path, required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument('--radius', type=float, default=64.0)
    args = ap.parse_args()
    if args.radius <= 0:
        ap.error('--radius must be positive')
    from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
    cached = {}
    for seed in args.seeds:
        tag = ('strict_hv_v2_strict_1711_seed0_float32_fix_dump' if seed == 0
               else 'strict_hv_v2_strict_1711_seed{}_dump'.format(seed))
        folder = args.eval_dir / tag
        files = sorted(folder.glob('*_online_generation.json'))
        if not files:
            print('\nSEED {}: no dump in {}'.format(seed, folder))
            continue
        for f in files:
            data = json.loads(f.read_text(encoding='utf-8'))
            town = (data.get('scene') or {}).get('town')
            if not town:
                print('SEED {}: missing town {}'.format(seed, f))
                continue
            if town not in cached:
                cached[town] = CarlaMap(carla_town_name=town, map_data_dir=str(args.hd_map_dir))
            analyze(f, seed, cached[town], args.radius)


if __name__ == '__main__':
    main()
