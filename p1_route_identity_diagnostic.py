#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only CARLA identity diagnostic for an ego_route stored in a P1 manifest.

Does not modify snapshots, spawn actors, tick the world, or repair route points.
If the current map differs from the snapshot town, load_world is allowed only
when the current CARLA world has zero vehicles. Writes
<output>/route_identity_diagnostic.json.
"""
from __future__ import print_function
import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def normalize_map_name(name):
    if name is None:
        return None
    return str(name).replace('\\', '/').rstrip('/').split('/')[-1]


def read_jsonl(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError('Invalid JSONL at line {}: {}'.format(line_no, exc))
            if not isinstance(row, dict):
                raise ValueError('Manifest line {} is not a JSON object'.format(line_no))
            rows.append(row)
    return rows


def resolve_snapshot_path(raw_path, manifest_path):
    path = Path(str(raw_path)).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        candidate = Path(manifest_path).resolve().parent / path
        if candidate.is_file():
            return candidate.resolve()
    return path.resolve()


def select_manifest_row(rows, record_index, manifest_row):
    if manifest_row is not None:
        if manifest_row < 0 or manifest_row >= len(rows):
            raise IndexError('--manifest-row {} outside 0..{}'.format(manifest_row, len(rows) - 1))
        row = rows[manifest_row]
        return row, {'selection_method': 'manifest_row', 'selected_manifest_row': manifest_row,
                     'selected_record_index': row.get('record_index')}
    matches = [(i, row) for i, row in enumerate(rows)
               if row.get('record_index') == record_index]
    if len(matches) == 1:
        i, row = matches[0]
        return row, {'selection_method': 'record_index', 'selected_manifest_row': i,
                     'selected_record_index': row.get('record_index')}
    if len(matches) > 1:
        raise ValueError('Multiple manifest rows have record_index={}'.format(record_index))
    if 0 <= record_index < len(rows):
        row = rows[record_index]
        return row, {'selection_method': 'row_ordinal_fallback',
                     'selected_manifest_row': record_index,
                     'selected_record_index': row.get('record_index'),
                     'selection_note': 'No record_index match; used zero-based row ordinal'}
    raise IndexError('No record_index={} and ordinal outside manifest ({} rows)'.format(
        record_index, len(rows)))


def wrap_degrees(value):
    return abs((float(value) + 180.0) % 360.0 - 180.0)


def distance_3d(a, b):
    dx, dy, dz = float(a.x) - float(b.x), float(a.y) - float(b.y), float(a.z) - float(b.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def waypoint_dict(wp):
    if wp is None:
        return None
    tr = wp.transform
    return {'road_id': int(wp.road_id), 'section_id': int(wp.section_id),
            'lane_id': int(wp.lane_id), 's': float(wp.s),
            'x': float(tr.location.x), 'y': float(tr.location.y), 'z': float(tr.location.z),
            'yaw_deg': float(tr.rotation.yaw), 'lane_width_m': float(getattr(wp, 'lane_width', 0.0)),
            'is_junction': bool(getattr(wp, 'is_junction', False))}


def xodr_lookup(carla_map, road_id, lane_id, s_value):
    try:
        return carla_map.get_waypoint_xodr(int(road_id), int(lane_id), float(s_value)), None
    except Exception as exc:
        return None, '{}: {}'.format(type(exc).__name__, exc)


def coordinate_lookup(carla_map, carla, row, project):
    try:
        loc = carla.Location(x=float(row['x']), y=float(row['y']), z=float(row['z']))
        wp = carla_map.get_waypoint(loc, project_to_road=bool(project),
                                    lane_type=carla.LaneType.Driving)
        return wp, None
    except Exception as exc:
        return None, '{}: {}'.format(type(exc).__name__, exc)


def diagnose_point(carla, carla_map, row, index):
    out = {'route_index': int(index), 'expected': row, 'xodr_found': False,
           'exact_found': False, 'projected_found': False, 'xodr_error': None,
           'exact_error': None, 'projected_error': None, 'xodr_probe_results': []}
    if not isinstance(row, dict):
        out['failure'] = 'route_point_not_object'
        return out
    required = ('x', 'y', 'z', 'yaw_deg', 'road_id', 'lane_id', 'section_id', 's')
    missing = [k for k in required if row.get(k) is None]
    if missing:
        out['failure'] = 'missing_required_fields'
        out['missing_fields'] = missing
        return out
    try:
        road_id, lane_id = int(row['road_id']), int(row['lane_id'])
        section_id, expected_s = int(row['section_id']), float(row['s'])
        expected_loc = carla.Location(x=float(row['x']), y=float(row['y']), z=float(row['z']))
        expected_yaw = float(row['yaw_deg'])
    except Exception as exc:
        out['failure'] = 'invalid_numeric_fields'
        out['error'] = '{}: {}'.format(type(exc).__name__, exc)
        return out

    xodr_wp, out['xodr_error'] = xodr_lookup(carla_map, road_id, lane_id, expected_s)
    exact_wp, out['exact_error'] = coordinate_lookup(carla_map, carla, row, False)
    projected_wp, out['projected_error'] = coordinate_lookup(carla_map, carla, row, True)
    out['xodr_found'], out['exact_found'], out['projected_found'] = (
        xodr_wp is not None, exact_wp is not None, projected_wp is not None)
    out['xodr_waypoint'] = waypoint_dict(xodr_wp)
    out['exact_waypoint'] = waypoint_dict(exact_wp)
    out['projected_waypoint'] = waypoint_dict(projected_wp)

    def compare(wp, prefix):
        if wp is None:
            return
        tr = wp.transform
        out[prefix + '_comparison'] = {
            'road_id_equal': int(wp.road_id) == road_id,
            'section_id_equal': int(wp.section_id) == section_id,
            'lane_id_equal': int(wp.lane_id) == lane_id,
            's_abs_error_m': abs(float(wp.s) - expected_s),
            'position_error_3d_m': distance_3d(tr.location, expected_loc),
            'yaw_error_deg': wrap_degrees(float(tr.rotation.yaw) - expected_yaw)}
    compare(xodr_wp, 'xodr')
    compare(exact_wp, 'exact')
    compare(projected_wp, 'projected')

    # Neighboring s probes are evidence only; none is substituted into the route.
    for delta in (-1.0, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1.0):
        probe_s = expected_s + delta
        if probe_s < 0:
            out['xodr_probe_results'].append({'delta_s_m': delta, 'probe_s': probe_s,
                                               'found': False, 'skipped': 'negative_s'})
            continue
        wp, error = xodr_lookup(carla_map, road_id, lane_id, probe_s)
        item = {'delta_s_m': delta, 'probe_s': probe_s, 'found': wp is not None,
                'error': error, 'waypoint': waypoint_dict(wp)}
        if wp is not None:
            item['expected_section_id_equal'] = int(wp.section_id) == section_id
        out['xodr_probe_results'].append(item)

    out['identity_status'] = {
        'xodr_identity_match': bool(xodr_wp is not None and int(xodr_wp.road_id) == road_id
                                    and int(xodr_wp.section_id) == section_id
                                    and int(xodr_wp.lane_id) == lane_id),
        'exact_identity_match': bool(exact_wp is not None and int(exact_wp.road_id) == road_id
                                     and int(exact_wp.section_id) == section_id
                                     and int(exact_wp.lane_id) == lane_id),
        'projected_identity_match': bool(projected_wp is not None
                                         and int(projected_wp.road_id) == road_id
                                         and int(projected_wp.section_id) == section_id
                                         and int(projected_wp.lane_id) == lane_id)}
    return out


def snapshot_opendrive_sha(scene):
    meta = scene.get('native_map_metadata')
    if isinstance(meta, dict) and meta.get('opendrive_sha256'):
        return meta['opendrive_sha256']
    if scene.get('opendrive_sha256'):
        return scene['opendrive_sha256']
    extra = scene.get('extra')
    if isinstance(extra, dict):
        meta = extra.get('native_map_metadata')
        if isinstance(meta, dict) and meta.get('opendrive_sha256'):
            return meta['opendrive_sha256']
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='P1 manifest JSONL with snapshot_path')
    p.add_argument('--rift-root', required=True, help='RIFT root directory')
    p.add_argument('--carla-pythonapi', default=None, help='CARLA PythonAPI/carla directory')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=31120)
    p.add_argument('--rpc-timeout', type=float, default=120.0)
    p.add_argument('--record-index', type=int, default=0,
                   help='Prefer matching record_index; if absent, fall back to row ordinal')
    p.add_argument('--manifest-row', type=int, default=None,
                   help='Explicit zero-based manifest row; overrides --record-index')
    p.add_argument('--max-route-points', type=int, default=50)
    p.add_argument('--output', required=True, help='Output directory')
    args = p.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    report_path = output_dir / 'route_identity_diagnostic.json'
    if report_path.exists():
        print('Refusing to overwrite existing report: {}'.format(report_path), file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 'p1_route_identity_diagnostic_v1',
              'diagnostic_completed': False, 'manifest_path': str(Path(args.manifest).resolve()),
              'output_path': str(report_path), 'host': args.host, 'port': args.port,
              'record_index_requested': args.record_index, 'max_route_points': args.max_route_points,
              'route_checks': [], 'summary': {}}
    try:
        rows = read_jsonl(args.manifest)
        if not rows:
            raise ValueError('P1 manifest is empty')
        manifest_row, selection = select_manifest_row(rows, args.record_index, args.manifest_row)
        report['manifest_selection'] = selection
        report['manifest_record'] = manifest_row
        if not manifest_row.get('snapshot_path'):
            raise ValueError('Selected manifest row has no snapshot_path')
        snapshot_path = resolve_snapshot_path(manifest_row['snapshot_path'], args.manifest)
        if not snapshot_path.is_file():
            raise FileNotFoundError('Snapshot does not exist: {}'.format(snapshot_path))
        snapshot_bytes = snapshot_path.read_bytes()
        scene = json.loads(snapshot_bytes.decode('utf-8'))
        if not isinstance(scene, dict):
            raise ValueError('Snapshot root is not a JSON object')
        report['snapshot_path'] = str(snapshot_path)
        report['snapshot_sha256'] = sha256_bytes(snapshot_bytes)
        report['scene_id'] = scene.get('scene_id', manifest_row.get('scene_id'))
        native_meta = scene.get('native_map_metadata') or {}
        report['town_expected'] = scene.get('town') or scene.get('map_name') or native_meta.get('map_name')
        report['opendrive_sha256_snapshot'] = snapshot_opendrive_sha(scene)
        route = scene.get('ego_route') or []
        report['ego_route_length'] = len(route)

        for path in (args.rift_root, args.carla_pythonapi):
            if path:
                resolved = str(Path(path).expanduser().resolve())
                if resolved not in sys.path:
                    sys.path.insert(0, resolved)
        import carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(float(args.rpc_timeout))
        try:
            report['server_version'] = client.get_server_version()
        except Exception as exc:
            report['server_version_error'] = '{}: {}'.format(type(exc).__name__, exc)
        try:
            report['client_version'] = client.get_client_version()
        except Exception as exc:
            report['client_version_error'] = '{}: {}'.format(type(exc).__name__, exc)

        world = client.get_world()
        current_map = world.get_map()
        expected_town = report.get('town_expected')
        if expected_town and normalize_map_name(current_map.name) != normalize_map_name(expected_town):
            vehicles = list(world.get_actors().filter('vehicle.*'))
            report['map_switch'] = {'needed': True, 'current_map': current_map.name,
                                    'expected_town': expected_town,
                                    'current_vehicle_count': len(vehicles), 'performed': False}
            if vehicles:
                raise RuntimeError('Map differs (current={!r}, expected={!r}) and current world has '
                                   '{} vehicles; refusing load_world. Use a dedicated empty CARLA server.'
                                   .format(current_map.name, expected_town, len(vehicles)))
            world = client.load_world(str(expected_town))
            current_map = world.get_map()
            report['map_switch']['performed'] = True
        else:
            report['map_switch'] = {'needed': False, 'current_map': current_map.name,
                                    'expected_town': expected_town, 'performed': False}

        carla_map = world.get_map()
        report['carla_map_name'] = carla_map.name
        report['town_name_match'] = (normalize_map_name(carla_map.name) == normalize_map_name(expected_town)
                                     if expected_town else None)
        runtime_sha = sha256_bytes(carla_map.to_opendrive().encode('utf-8'))
        report['opendrive_sha256_runtime'] = runtime_sha
        expected_sha = report.get('opendrive_sha256_snapshot')
        report['map_hash_match'] = runtime_sha == expected_sha if expected_sha is not None else None
        report['map_hash_note'] = ('Compared against stored snapshot OpenDRIVE SHA-256'
                                   if expected_sha is not None else
                                   'Snapshot contains no OpenDRIVE SHA-256; comparison unavailable')

        checks = [diagnose_point(carla, carla_map, point, i)
                  for i, point in enumerate(route[:max(0, int(args.max_route_points))])]
        report['route_checks'] = checks
        xodr_missing = [r['route_index'] for r in checks if not r.get('xodr_found')]
        exact_missing = [r['route_index'] for r in checks if not r.get('exact_found')]
        exact_mismatch = [r['route_index'] for r in checks if r.get('exact_found') and not
                          r.get('identity_status', {}).get('exact_identity_match', False)]
        xodr_mismatch = [r['route_index'] for r in checks if r.get('xodr_found') and not
                         r.get('identity_status', {}).get('xodr_identity_match', False)]
        report['route_37'] = next((r for r in checks if r.get('route_index') == 37), None)
        report['summary'] = {'checked_points': len(checks),
                             'xodr_missing_count': len(xodr_missing),
                             'xodr_missing_indices': xodr_missing,
                             'exact_found_count': sum(1 for r in checks if r.get('exact_found')),
                             'exact_missing_count': len(exact_missing),
                             'exact_missing_indices': exact_missing,
                             'exact_identity_mismatch_count': len(exact_mismatch),
                             'exact_identity_mismatch_indices': exact_mismatch,
                             'xodr_identity_mismatch_count': len(xodr_mismatch),
                             'xodr_identity_mismatch_indices': xodr_mismatch,
                             'map_hash_match': report.get('map_hash_match'),
                             'town_name_match': report.get('town_name_match')}
        report['diagnostic_completed'] = True
    except Exception as exc:
        report['fatal_error'] = '{}: {}'.format(type(exc).__name__, exc)
        report['traceback'] = traceback.format_exc()
        report['diagnostic_completed'] = False

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
                           encoding='utf-8')
    print('P1_ROUTE_IDENTITY_DIAGNOSTIC_WRITTEN={}'.format(report_path))
    print(json.dumps({k: report.get(k) for k in (
        'diagnostic_completed', 'fatal_error', 'snapshot_path', 'scene_id', 'town_expected',
        'carla_map_name', 'server_version', 'opendrive_sha256_snapshot',
        'opendrive_sha256_runtime', 'map_hash_match', 'summary')}, indent=2, ensure_ascii=False))
    return 0 if report.get('diagnostic_completed') else 2


if __name__ == '__main__':
    sys.exit(main())
