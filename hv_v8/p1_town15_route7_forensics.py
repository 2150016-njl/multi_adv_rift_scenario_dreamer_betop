#!/usr/bin/env python3
"""Read-only CARLA geometry investigation for P1 Town15 manifest rows 6..11.

No map changes by default, no actor spawn/destruction/tick, no P1 source edits.
Optional --load-town15-if-empty explicitly permits a map switch only if there
are zero vehicle actors on a dedicated CARLA instance.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROWS = (6, 7, 8, 9, 10, 11)
EXPECTED_RECORD_INDEX = (90, 95, 100, 105, 110, 115)


def normalize_town(name):
    return str(name or '').replace('\\', '/').rstrip('/').split('/')[-1]


def dist(a, b, include_z=False):
    squared = (float(a.x) - float(b.x)) ** 2 + (float(a.y) - float(b.y)) ** 2
    if include_z:
        squared += (float(a.z) - float(b.z)) ** 2
    return math.sqrt(squared)


def angle_diff(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def waypoint_evidence(wp, r, expected_location):
    if wp is None:
        return {'found': False}
    t = wp.transform
    return {
        'found': True,
        'road_id': int(wp.road_id),
        'section_id': int(wp.section_id),
        'lane_id': int(wp.lane_id),
        's': float(wp.s),
        'x': float(t.location.x),
        'y': float(t.location.y),
        'z': float(t.location.z),
        'yaw_deg': float(t.rotation.yaw),
        'lane_identity_equal': (
            int(wp.road_id) == int(r['road_id'])
            and int(wp.section_id) == int(r['section_id'])
            and int(wp.lane_id) == int(r['lane_id'])
        ),
        's_signed_delta_m': float(wp.s) - float(r['s']),
        's_abs_error_m': abs(float(wp.s) - float(r['s'])),
        'position_error_xy_m': dist(t.location, expected_location),
        'position_error_xyz_m': dist(t.location, expected_location, True),
        'yaw_error_deg': angle_diff(t.rotation.yaw, float(r['yaw_deg'])),
    }


def find_snapshot(manifest_path, raw):
    path = Path(str(raw)).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        candidate = manifest_path.parent / path
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError('snapshot file not found: {!r}'.format(raw))


def run(args):
    root = Path(args.root).expanduser().resolve()
    manifest = root / 'data/hv_init/v7/manifests/p1_calibration95.jsonl'
    map_manifest = root / 'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    for path in (manifest, map_manifest):
        if not path.is_file():
            raise RuntimeError('required input missing: {}'.format(path))
    provenance = json.loads(map_manifest.read_text(encoding='utf-8'))
    if provenance.get('schema_version') != 'p1_map_identity_manifest_v1':
        raise RuntimeError('unexpected trusted map manifest schema')
    matches = [m for m in provenance.get('maps', []) if normalize_town(m.get('town')) == 'Town15']
    if len(matches) != 1:
        raise RuntimeError('expected exactly one Town15 map identity entry, found {}'.format(len(matches)))
    expected_sha = str(matches[0]['opendrive_sha256']).lower()

    rows = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    candidates = []
    for ordinal, record_index in zip(ROWS, EXPECTED_RECORD_INDEX):
        if ordinal >= len(rows):
            raise RuntimeError('manifest too short for row {}'.format(ordinal))
        entry = rows[ordinal]
        if int(entry.get('record_index', -1)) != record_index:
            raise RuntimeError('manifest row {} record_index changed: {} != {}'.format(
                ordinal, entry.get('record_index'), record_index))
        path = find_snapshot(manifest, entry.get('snapshot_path'))
        raw = path.read_bytes()
        scene = json.loads(raw.decode('utf-8'))
        if normalize_town(scene.get('town', scene.get('map_name'))) != 'Town15':
            raise RuntimeError('record {} is not Town15'.format(record_index))
        route = scene.get('ego_route', [])
        if not isinstance(route, list) or len(route) <= 7:
            raise RuntimeError('record {} has no ego_route[7]'.format(record_index))
        candidates.append({
            'manifest_ordinal': ordinal,
            'record_index': record_index,
            'snapshot_path': str(path),
            'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
            'route_index': 7,
            'route_point': route[7],
        })

    carla_py = root / 'carla/PythonAPI/carla'
    carla_dist = carla_py / 'dist'
    if carla_dist.is_dir():
        for egg in sorted(carla_dist.glob('carla-0.9.15-*.egg')):
            sys.path.insert(0, str(egg))
    sys.path.insert(0, str(carla_py))
    import carla
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(float(args.rpc_timeout))
    world = client.get_world()
    current_map = world.get_map()
    did_load = False
    if normalize_town(current_map.name) != 'Town15':
        if not args.load_town15_if_empty:
            raise RuntimeError(
                'current CARLA map is {!r}, not Town15; no map was loaded. '
                'For an exclusively controlled server, use --load-town15-if-empty.'
                .format(current_map.name))
        vehicles = list(world.get_actors().filter('vehicle.*'))
        if vehicles:
            raise RuntimeError('refusing to load Town15: {} vehicles exist'.format(len(vehicles)))
        print('EXPLICIT_MAP_SWITCH=Town15; authorized by --load-town15-if-empty, zero vehicles confirmed')
        world = client.load_world('Town15')
        current_map = world.get_map()
        did_load = True
    if normalize_town(current_map.name) != 'Town15':
        raise RuntimeError('CARLA did not load Town15: {}'.format(current_map.name))
    actual_sha = hashlib.sha256(current_map.to_opendrive().encode('utf-8')).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError('Town15 OpenDRIVE SHA mismatch actual={} manifest={}'.format(actual_sha, expected_sha))
    print('P1_TOWN15_MAP_SHA256_MATCH_PASS={}'.format(actual_sha))

    for c in candidates:
        r = c['route_point']
        loc = carla.Location(x=float(r['x']), y=float(r['y']), z=float(r['z']))
        fetches = (
            ('xodr', lambda: current_map.get_waypoint_xodr(
                int(r['road_id']), int(r['lane_id']), float(r['s']))),
            ('exact', lambda: current_map.get_waypoint(
                loc, project_to_road=False, lane_type=carla.LaneType.Driving)),
            ('projected', lambda: current_map.get_waypoint(
                loc, project_to_road=True, lane_type=carla.LaneType.Driving)),
        )
        wps = {}
        for name, fn in fetches:
            try:
                wps[name] = fn()
                c[name] = waypoint_evidence(wps[name], r, loc)
            except Exception as exc:
                wps[name] = None
                c[name] = {'found': False, 'exception': '{}: {}'.format(type(exc).__name__, exc)}
        if wps.get('xodr') is not None and wps.get('exact') is not None:
            x = wps['xodr']
            e = wps['exact']
            c['xodr_vs_exact'] = {
                's_abs_delta_m': abs(float(x.s) - float(e.s)),
                'position_delta_xy_m': dist(x.transform.location, e.transform.location),
                'yaw_delta_deg': angle_diff(x.transform.rotation.yaw, e.transform.rotation.yaw),
            }
        c['old_audit_s_tolerance_m'] = 0.05
        c['old_audit_would_reject_due_to_exact_s'] = (
            not c['exact']['found'] or c['exact']['s_abs_error_m'] > 0.05
        )
        fmt = lambda k, m: c[k].get(m, None)
        print('record_index={ri} manifest_ordinal={idx} '
              'snapshot_s={s:.9f} exact_s_delta={es} xodr_s_delta={xs} '
              'exact_xy_error_m={ep} xodr_xy_error_m={xp} '
              'exact_identity={ei} xodr_identity={xi} old_005_reject={rej}'.format(
                  ri=c['record_index'], idx=c['manifest_ordinal'], s=float(r['s']),
                  es=fmt('exact', 's_signed_delta_m'), xs=fmt('xodr', 's_signed_delta_m'),
                  ep=fmt('exact', 'position_error_xy_m'), xp=fmt('xodr', 'position_error_xy_m'),
                  ei=fmt('exact', 'lane_identity_equal'), xi=fmt('xodr', 'lane_identity_equal'),
                  rej=c['old_audit_would_reject_due_to_exact_s']))

    result = {
        'schema_version': 'p1_town15_route7_geometry_forensics_v1',
        'scientific_audit_not_performed': True,
        'snapshot_data_modified': False,
        'actors_spawned_or_destroyed': False,
        'world_tick_called': False,
        'world_map_load_performed': did_load,
        'carla_map': current_map.name,
        'opendrive_sha256_expected': expected_sha,
        'opendrive_sha256_actual': actual_sha,
        'manifest_path': str(manifest),
        'candidates': candidates,
    }
    report_dir = root / 'data/hv_init/v7'
    outpath = Path(args.output).expanduser().resolve() if args.output else (
        report_dir / ('p1_town15_route7_forensics_{}_{}.json'.format(
            datetime.now().strftime('%Y%m%d_%H%M%S'), os.getpid())))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('P1_TOWN15_ROUTE7_FORENSICS_WRITTEN={}'.format(outpath))
    print('P1_GEOMETRY_DIAGNOSTIC_ONLY_NO_P1_PASS_CLAIM')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=None, help='project root; default parent of hv_v8 folder')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=31120)
    p.add_argument('--rpc-timeout', type=float, default=120.0)
    p.add_argument('--load-town15-if-empty', action='store_true',
                   help='explicitly allow loading Town15 if current map differs and zero vehicles exist')
    p.add_argument('--output', default=None, help='new JSON report filename (must not exist)')
    args = p.parse_args()
    if args.root is None:
        args.root = str(Path(__file__).resolve().parent.parent)
    try:
        return run(args)
    except Exception as exc:
        print('P1_TOWN15_ROUTE7_FORENSICS_STOP={}: {}'.format(type(exc).__name__, exc),
              file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
