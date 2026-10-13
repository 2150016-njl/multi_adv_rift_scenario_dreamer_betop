#!/usr/bin/env python3
"""Observational audit only: compare raw HV states with model HD Map and CARLA roads.
Never spawns, resamples, changes actor states, or alters RIFT rules.
"""
import argparse
import json
import math
from pathlib import Path


def xy(p):
    if isinstance(p, dict):
        return float(p['x']), float(p['y'])
    return float(p[0]), float(p[1])


def segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    u = 0.0 if denom <= 1e-15 else max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / denom))
    return math.hypot(p[0] - (ax + u * vx), p[1] - (ay + u * vy))


def closest_polyline_distance(pos, polylines):
    best = float('inf')
    for poly in polylines:
        points = [xy(q) for q in poly]
        for a, b in zip(points, points[1:]):
            best = min(best, segment_distance(pos, a, b))
    return None if math.isinf(best) else best


def wrap_deg(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def fmt(v):
    return '-' if v is None else f'{v:.3f}'


def analyze(path, seed):
    d = json.loads(path.read_text())
    scene = d.get('scene') or {}
    proposal = d.get('proposal') or {}
    result = d.get('result') or {}
    ego = scene.get('ego') or {}
    lane_polylines = [lane.get('centerline', []) for lane in scene.get('map_lanes', []) if len(lane.get('centerline', [])) >= 2]
    route = scene.get('ego_route', [])
    routes = [route] if len(route) >= 2 else []
    attempts = {int(item['index']): item for item in result.get('spawn_attempts', [])}
    for failure in result.get('spawn_failures', []):
        for item in failure.get('spawn_attempts', []):
            attempts.setdefault(int(item['index']), item)
    if not ego:
        print(f'seed={seed} ERROR: no scene.ego in {path.name}')
        return
    ego_yaw = math.radians(float(ego.get('yaw_deg', 0)))
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    print(f'\nSEED {seed}  success={result.get("success")} stage={result.get("failure_stage")} lanes={len(lane_polylines)} route_points={len(route)} map_mode={scene.get("diagnostics", {}).get("map_coordinate_mode")}')
    states = proposal.get('generated_agent_states') or []
    for idx, state in enumerate(states):
        if len(state) < 4:
            print(f' HV{idx} malformed state={state!r}')
            continue
        lx, ly, heading, speed = (float(x) for x in state[:4])
        world = (float(ego['x']) + c * lx - s * ly, float(ego['y']) + s * lx + c * ly)
        hd = closest_polyline_distance(world, lane_polylines)
        route_dist = closest_polyline_distance(world, routes)
        attempt = attempts.get(idx, {})
        carla_dist = attempt.get('road_projection_distance_m')
        wp = attempt.get('road_waypoint') or {}
        heading_diff = abs(wrap_deg(math.degrees(ego_yaw + heading) - float(wp['yaw_deg']))) if 'yaw_deg' in wp else None
        accepted = attempt.get('spawned_actor_id') is not None
        reqerr = attempt.get('spawn_request_error')
        print(f' HV{idx} local=({lx:+.2f},{ly:+.2f}) yaw={math.degrees(heading):+.1f}deg v={speed:.2f}m/s '
              f'HDmap_centerline={fmt(hd)}m ego_route={fmt(route_dist)}m '
              f'CARLA_driving_waypoint={fmt(carla_dist)}m '
              f'CARLA_heading_diff={fmt(heading_diff)}deg '
              f'accepted={accepted} attempted={bool(attempt)}')
        if reqerr:
            print(f'   spawn_error: {reqerr}')
        if hd is not None and carla_dist is not None:
            if hd <= 3.0 and carla_dist >= 10.0:
                print('   FLAG: near MODEL HD Map lane but far from CARLA driving lane — inspect map semantics/coordinates')
            elif hd >= 10.0 and carla_dist >= 10.0:
                print('   FLAG: far from BOTH maps — prioritize generative geometry / raw AE-LDM audit')
    if not lane_polylines:
        print(' WARNING: snapshot has zero usable HD Map lane centerlines; inspect HD Map query and fallback')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', required=True, type=Path, help='data/hv_init/eval')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(6)))
    a = parser.parse_args()
    for seed in a.seeds:
        names = ([f'strict_hv_v2_strict_1711_seed0_float32_fix_dump'] if seed == 0 else [f'strict_hv_v2_strict_1711_seed{seed}_dump'])
        folder = a.eval_dir / names[0]
        files = sorted(folder.glob('*_online_generation.json'))
        if not files:
            print(f'\nSEED {seed}: missing original dump: {folder}')
        else:
            for p in files:
                analyze(p, seed)


if __name__ == '__main__':
    main()
