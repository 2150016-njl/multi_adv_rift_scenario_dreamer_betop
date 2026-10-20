#!/usr/bin/env python3
"""P1 read-only route-geometry census against real CARLA 0.9.15.

NOT a native-RIFT, occupancy, GT-legality, or P1 acceptance test.
Loads each required town once. Does not spawn, move actors, or tick.
Fails closed on snapshot SHA, world occupation, Town, OpenDRIVE SHA.
Records query failures without converting them to scientific PASS.
Python 3.8 compatible; do not modify frozen p1_native_audit.py.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def angle_delta(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def xyz_error(p, q):
    return math.sqrt((float(p.x)-float(q.x))**2 + (float(p.y)-float(q.y))**2 + (float(p.z)-float(q.z))**2)


def xy_error(p, q):
    return math.hypot(float(p.x)-float(q.x), float(p.y)-float(q.y))


def identity(w):
    return [int(w.road_id), int(w.section_id), int(w.lane_id)]


def details(w, original):
    if w is None:
        return None
    loc = w.transform.location
    return {
        'road_id': int(w.road_id), 'section_id': int(w.section_id),
        'lane_id': int(w.lane_id), 's': float(w.s),
        's_delta_m': float(w.s) - float(original['s']),
        'xy_error_m': xy_error(loc, original['_location']),
        'xyz_error_m': xyz_error(loc, original['_location']),
        'yaw_error_deg': angle_delta(w.transform.rotation.yaw, original['yaw_deg']),
    }


def route_probe(carla, cm, r, road_lengths):
    source_id = [int(r[k]) for k in ('road_id','section_id','lane_id')]
    s = float(r['s'])
    loc = carla.Location(x=float(r['x']), y=float(r['y']), z=float(r['z']))
    reference = dict(r)
    reference['_location'] = loc
    xodr = cm.get_waypoint_xodr(source_id[0], source_id[2], s)
    exact = cm.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
    xdat = details(xodr, reference)
    edat = details(exact, reference)
    road_length = road_lengths.get(source_id[0])
    endpoint_delta = (s - road_length) if road_length is not None else None
    xodr_source_ok = bool(xdat and
        [xdat[k] for k in ('road_id','section_id','lane_id')] == source_id and
        abs(xdat['s_delta_m']) <= 0.0001 and
        xdat['xyz_error_m'] <= 0.05 and xdat['yaw_error_deg'] <= 0.5)
    exact_id_ok = bool(edat and [edat[k] for k in ('road_id','section_id','lane_id')] == source_id)
    exact_geometry_ok = bool(edat and exact_id_ok and edat['xy_error_m'] <= 0.30 and edat['yaw_error_deg'] <= 2.0)
    boundary_ok = bool(xdat is None and edat and exact_id_ok and endpoint_delta is not None and
        abs(endpoint_delta) <= 0.0001 and abs(edat['s_delta_m']) <= 0.0001 and
        edat['xyz_error_m'] <= 0.05 and edat['yaw_error_deg'] <= 0.5)
    comparison_ok = bool(xodr is not None and exact is not None and
        xy_error(xodr.transform.location, exact.transform.location) <= 0.30 and
        angle_delta(xodr.transform.rotation.yaw, exact.transform.rotation.yaw) <= 2.0)
    if xodr_source_ok and exact_geometry_ok and comparison_ok:
        status = 'verified_xodr_and_exact'
    elif boundary_ok:
        status = 'verified_xodr_road_end_exception'
    elif xodr_source_ok and exact is None:
        status = 'xodr_source_verified_exact_none'
    elif xodr_source_ok and edat and not exact_id_ok:
        status = 'xodr_source_verified_exact_other_lane'
    elif xodr_source_ok:
        status = 'xodr_source_verified_exact_geometry_conflict'
    elif xodr is None and exact is None:
        status = 'both_lookups_none'
    elif xodr is None:
        status = 'xodr_missing_not_verified_endpoint'
    else:
        status = 'xodr_source_geometry_or_identity_mismatch'
    out = {
        'status': status, 'source_id': source_id, 'source_s': s,
        'source_xyz': [float(r[k]) for k in ('x','y','z')],
        'source_yaw_deg': float(r['yaw_deg']),
        'xodr': xdat, 'exact_driving': edat,
        'road_length': road_length, 's_minus_road_end_m': endpoint_delta,
        'xodr_source_ok': xodr_source_ok, 'exact_identity_ok': exact_id_ok,
        'exact_geometry_ok': exact_geometry_ok, 'boundary_ok': boundary_ok,
        'xodr_vs_exact_ok': comparison_ok,
    }
    # Only diagnostic: projected waypoints are NEVER used to replace ground truth.
    if exact is None:
        any_wp = cm.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any)
        proj_wp = cm.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        out['exact_any'] = details(any_wp, reference)
        out['projected_driving_diagnostic'] = details(proj_wp, reference)
    return out


def read_inputs(root):
    base = root / 'data/hv_init/v7/manifests'
    manifest = base/'p1_calibration95.jsonl'
    mr = manifest.read_bytes()
    rows = [json.loads(t) for t in mr.decode('utf-8').splitlines() if t.strip()]
    require(len(rows) == 95, 'expected exactly 95 manifest entries')
    require(len({int(r['record_index']) for r in rows}) == 95, 'duplicate record_index')
    seen_paths = set()
    data = []
    for ordinal, row in enumerate(rows):
        path = Path(row['snapshot_path'])
        require(str(path) not in seen_paths, 'duplicate snapshot path '+str(path))
        seen_paths.add(str(path))
        raw = path.read_bytes()
        require(digest(raw) == row['snapshot_sha256'], 'snapshot SHA mismatch ordinal='+str(ordinal))
        scene = json.loads(raw)
        require(scene['town'] in ('Town04','Town11','Town12','Town13','Town15'), 'unrecognized town '+str(scene.get('town')))
        require(isinstance(scene.get('ego_route'), list) and scene['ego_route'], 'missing ego_route ordinal '+str(ordinal))
        data.append((ordinal, row, scene))
    map_manifest = base/'p1_map_identity_manifest_v2_20261017_154737.json'
    mmraw = map_manifest.read_bytes()
    mm = json.loads(mmraw)
    expected = {item['town']:item['opendrive_sha256'] for item in mm['maps']}
    require(len(expected) == len(mm['maps']), 'duplicate map manifest Town')
    for _,_,sc in data:
        require(sc['town'] in expected, 'Town omitted from map manifest')
    return data, expected, {'manifest_sha256':digest(mr), 'map_manifest_sha256':digest(mmraw)}


def run(carla, client, records, expected, hashes, outdir, route_limit):
    groups = collections.OrderedDict()
    for record in records:
        groups.setdefault(record[2]['town'], []).append(record)
    summary = {'status':'in_progress', 'input_hashes':hashes, 'towns':{},'point_status_counts':{},
               'points_scanned':0,'records_scanned':0,'records_expected':len(records),'unresolved_points':0,
               'important':'Geometry/query diagnostic only; not P1 PASS, RIFT PASS, or CARLA spawn proof.'}
    outdir.mkdir(parents=True, exist_ok=False)
    anomalypath = outdir/'anomalies.jsonl'
    recordpath = outdir/'records.jsonl'
    status_counts = collections.Counter()
    with anomalypath.open('w') as af, recordpath.open('w') as rf:
        for town, group in groups.items():
            town_counts = collections.Counter()
            print('GEOMETRY_TOWN_START='+town+' RECORDS='+str(len(group)),flush=True)
            # Refuse to affect a world with vehicles/walkers, whether or not map already loaded.
            w = client.get_world()
            require(not w.get_actors().filter('vehicle.*') and not w.get_actors().filter('walker.pedestrian.*'),
                    'occupied world before Town switch '+town)
            world = client.load_world(town)
            cm = world.get_map()
            mapname = cm.name.replace('\\','/').split('/')[-1]
            raw_opendrive = cm.to_opendrive()
            actual = digest(raw_opendrive.encode('utf-8'))
            # Fail before examining any record on unverified map.
            require(mapname == town and actual == expected[town],
                    'MAP_IDENTITY_FAIL town='+town+' actual_name='+cm.name+' actual_sha='+actual+' expected_sha='+expected[town])
            road_lengths = {int(x.get('id')):float(x.get('length')) for x in ET.fromstring(raw_opendrive).findall('road')}
            summary['towns'][town] = {'sha256':actual,'records':0,'points':0,'status_counts':{}}
            for ordinal,row,scene in group:
                idx = int(row['record_index'])
                counts = collections.Counter()
                for i,r in enumerate(scene['ego_route'][:route_limit]):
                    try:
                        result = route_probe(carla, cm, r, road_lengths)
                    except Exception as exc:
                        result = {'status':'query_exception','exception_type':type(exc).__name__, 'exception':str(exc)}
                    status = result['status']
                    counts[status] += 1
                    status_counts[status] += 1
                    town_counts[status] += 1
                    summary['points_scanned'] += 1
                    summary['towns'][town]['points'] += 1
                    if status not in ('verified_xodr_and_exact','verified_xodr_road_end_exception'):
                        summary['unresolved_points'] += 1
                        af.write(json.dumps({'ordinal':ordinal,'record_index':idx,'scene_id':scene.get('scene_id'),
                            'town':town,'route_index':i,**result},ensure_ascii=False,allow_nan=False)+'\n')
                        af.flush()
                rf.write(json.dumps({'ordinal':ordinal,'record_index':idx,'town':town,
                    'manifest_scene_id':row['scene_id'],'snapshot_sha256':row['snapshot_sha256'],
                    'route_points':min(len(scene['ego_route']),route_limit),'status_counts':dict(counts)},ensure_ascii=False)+'\n')
                rf.flush()
                summary['records_scanned'] += 1
                summary['towns'][town]['records'] += 1
                print('ROUTE_RECORD_DONE idx='+str(idx)+' town='+town+' statuses='+json.dumps(dict(counts),sort_keys=True), flush=True)
                # Persist progress across interruptions, do not count missing records as pass.
                summary['point_status_counts'] = dict(status_counts)
                (outdir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
            after_name = client.get_world().get_map().name.replace('\\','/').split('/')[-1]
            require(after_name == town, 'world changed during Town group '+town+': '+after_name)
            summary['towns'][town]['status_counts'] = dict(town_counts)
        summary['status']='complete_query_census_not_p1_acceptance'
        summary['point_status_counts']=dict(status_counts)
        (outdir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        print('ROUTE_GEOMETRY95_COMPLETE=1')
        print('ROUTE_GEOMETRY95_COUNTS='+json.dumps(dict(status_counts),sort_keys=True))
        print('ROUTE_GEOMETRY95_UNRESOLVED='+str(summary['unresolved_points']))
        print('ROUTE_GEOMETRY95_REPORT='+str(outdir))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=31120)
    p.add_argument('--route-limit',type=int,default=50)
    p.add_argument('--ack-dedicated',action='store_true')
    args=p.parse_args()
    try:
        require(args.ack_dedicated, '--ack-dedicated required: tool switches worlds')
        require(args.route_limit == 50, 'route-limit must remain 50 to match frozen audit')
        root=Path(args.root).resolve()
        output=Path(args.output_dir).resolve()
        require(not output.exists(), 'refuse overwrite '+str(output))
        records, expected, hashes=read_inputs(root)
        dist=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
        require(dist.is_file(), 'required pinned py3.7 CARLA egg missing '+str(dist))
        # Removing the py2 egg from sys.path avoids the previous audit launcher bug.
        sys.path=[x for x in sys.path if not ('carla-0.9.15-py2.7' in x)]
        sys.path.insert(0,str(dist))
        import carla
        client=carla.Client(args.host,args.port)
        client.set_timeout(120.0)
        require(client.get_client_version() == '0.9.15','wrong CARLA client')
        require(client.get_server_version() == '0.9.15','server version mismatch')
        run(carla,client,records,expected,hashes,output,args.route_limit)
    except Exception as exc:
        print('ROUTE_GEOMETRY95_FAIL_CLOSED='+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
