"""CARLA-only V7 map export. Fixed-cost waypoint sampling, no GT-dependent repairs.

Longitudinal edges come from CARLA Waypoint.next/previous. Lane-change links
come from CARLA lane markings and, independently, RIFT's already-built GRP.
The returned map is radius-limited: a missing edge remains UNKNOWN, not false.
"""
from collections import defaultdict
import math


def _uid(w):
    return '%s:%s:%s' % (w.road_id, w.section_id, w.lane_id)


def _driving(w):
    return w is not None and str(w.lane_type).rsplit('.', 1)[-1] == 'Driving'


def _yaw_distance_deg(a, b):
    return abs((float(a) - float(b) + 180.) % 360. - 180.)


def export_directed_lanes(world_map, ego, route_waypoints, radius_m=100.,
                          step_m=1., max_samples=250000, planner=None):
    """Return (lane geometry, map-only evidence), without target labels.

    generate_waypoints is queried ONCE. Unlike the old BFS, each seed is probed
    once for its immediate connections; it never recursively expands thousands
    of seeds with slightly different OpenDRIVE station phases.
    """
    if not (0.25 <= step_m <= 10. and radius_m > 0 and max_samples > 0):
        raise ValueError('Invalid CARLA map export bounds')

    def inside(w):
        p = w.transform.location
        return math.hypot(p.x - ego['x'], p.y - ego['y']) <= radius_m

    global_seeds = [w for w in world_map.generate_waypoints(step_m) if _driving(w) and inside(w)]
    route_seeds = [w for w in route_waypoints if _driving(w) and inside(w)]
    # Official topology endpoints carry lane/connector connectivity that a
    # regular grid may miss near a boundary. Never seed from held-out actors.
    topology_seeds = []
    for source, destination in getattr(world_map, "get_topology", lambda: [])():
        if _driving(source) and inside(source):
            topology_seeds.append(source)
        if _driving(destination) and inside(destination):
            topology_seeds.append(destination)
    seeds = global_seeds + topology_seeds + route_seeds
    if len(seeds) > max_samples:
        raise RuntimeError('Global/topology/route seed budget exceeded: %s > %s' % (len(seeds), max_samples))

    buckets = defaultdict(dict)
    successors = defaultdict(set)
    direction_votes = defaultdict(set)
    boundary = []
    transitions = {}
    checked = set()
    try:
        import carla
    except ImportError:
        # Offline geometry tests may not have the CARLA egg on sys.path.
        # Production runs use the actual CARLA enum; absence disables only
        # lane-marking discovery, never synthesizes a transition.
        carla = None

    def add_transition(source, dest, kind, evidence):
        if not (_driving(source) and _driving(dest)):
            return
        if not (inside(source) or inside(dest)):
            return
        key = (_uid(source), _uid(dest), kind,
               round(float(source.s), 2), round(float(dest.s), 2))
        transitions[key] = {'source': key[0], 'destination': key[1],
                            'kind': kind, 'source_s_m': float(source.s),
                            'destination_s_m': float(dest.s),
                            'source_xy': [float(source.transform.location.x), float(source.transform.location.y)],
                            'destination_xy': [float(dest.transform.location.x), float(dest.transform.location.y)],
                            'evidence': evidence}

    for w in seeds:
        lane_id = _uid(w)
        station = float(w.s)
        # A 1-m bin is about one geometric vertex, irrespective of phase.
        # Prioritize topology/route seeds (processed after global seeds) at
        # equal bins; all underlying geometry stays CARLA-derived.
        bin_id = int(math.floor(station / step_m))
        buckets[lane_id][bin_id] = w
        key = (lane_id, round(station, 2))
        if key in checked:
            continue
        checked.add(key)
        for direction in ('next', 'previous'):
            try:
                neighbors = getattr(w, direction)(step_m)
            except Exception as exc:
                raise RuntimeError('Waypoint.%s failed on %s@%.3f: %s' %
                                   (direction, lane_id, station, exc)) from exc
            for other in neighbors:
                if not _driving(other):
                    continue
                src, dst = (w, other) if direction == 'next' else (other, w)
                if _uid(src) == _uid(dst):
                    if inside(other):
                        other_bin = int(math.floor(float(other.s) / step_m))
                        buckets[lane_id].setdefault(other_bin, other)
                    if abs(float(dst.s) - float(src.s)) > 1.e-6:
                        direction_votes[lane_id].add(1 if dst.s > src.s else -1)
                else:
                    successors[_uid(src)].add(_uid(dst))
                    add_transition(src, dst, 'longitudinal', 'CARLA Waypoint.next/previous')
                if not inside(other):
                    boundary.append({'source': _uid(src), 'destination': _uid(dst),
                                     'kind': 'radius_boundary', 'source_s_m': float(src.s),
                                     'destination_s_m': float(dst.s)})

        # Match RIFT GRP _lane_change_link(): check the relevant source lane
        # marking AND the adjacent Driving waypoint. Never link by XY distance.
        for side, enum_name in (('left', 'Left'), ('right', 'Right')):
            marking = getattr(w, side + '_lane_marking', None)
            allowed = getattr(marking, 'lane_change', None)
            flag = getattr(getattr(carla, 'LaneChange', None), enum_name, None)
            if flag is None or allowed is None or not (allowed & flag):
                continue
            adjacent = getattr(w, 'get_' + side + '_lane')()
            if not _driving(adjacent) or not inside(adjacent):
                continue
            if _yaw_distance_deg(w.transform.rotation.yaw, adjacent.transform.rotation.yaw) > 45.:
                continue
            add_transition(w, adjacent, 'lane_change_' + side,
                           'CARLA source lane marking and adjacent Driving waypoint')

    # Independent planner evidence, if RIFT GRP actually exposes the edge.
    if planner is not None and getattr(planner, '_graph', None) is not None:
        for _, _, edge in planner._graph.edges(data=True):
            option = str(edge.get('type', ''))
            if 'CHANGELANE' not in option.upper():
                continue
            src, dst = edge.get('entry_waypoint'), edge.get('exit_waypoint')
            if not (_driving(src) and _driving(dst)):
                continue
            if _yaw_distance_deg(src.transform.rotation.yaw, dst.transform.rotation.yaw) > 45.:
                continue
            side = 'left' if 'LEFT' in option.upper() else 'right'
            add_transition(src, dst, 'lane_change_' + side,
                           'RIFT GRP graph RoadOption.' + option)

    # We only expose map graph vertices backed by at least two samples.
    # Source/target station of typed transitions must not be discarded.
    for row in transitions.values():
        for lane_id, s in ((row['source'], row['source_s_m']),
                           (row['destination'], row['destination_s_m'])):
            if lane_id in buckets:
                # Existing same-bin sample is sufficient for lane coverage.
                if int(math.floor(s / step_m)) not in buckets[lane_id]:
                    # The transition is still preserved with its exact station
                    # even if the coarse geometry omits this vertex.
                    boundary.append({'source': row['source'], 'destination': row['destination'],
                                     'kind': 'transition_station_not_sampled',
                                     'source_s_m': row['source_s_m'],
                                     'destination_s_m': row['destination_s_m']})

    lanes = []
    for lane_id, sampled in sorted(buckets.items()):
        pts = list(sampled.values())
        if len(pts) < 2:
            continue
        votes = direction_votes[lane_id]
        if len(votes) > 1:
            raise ValueError('Conflicting CARLA same-lane travel directions: ' + lane_id)
        if votes:
            sign = next(iter(votes))
        else:
            # When next/previous produce no same-lane sample, use the native
            # waypoint heading relative to station geometry; no target yaw.
            first, last = sorted(pts, key=lambda x: float(x.s))[0], sorted(pts, key=lambda x: float(x.s))[-1]
            dx = last.transform.location.x - first.transform.location.x
            dy = last.transform.location.y - first.transform.location.y
            tangent = math.degrees(math.atan2(dy, dx))
            sign = 1 if _yaw_distance_deg(first.transform.rotation.yaw, tangent) <= 90 else -1
        pts.sort(key=lambda p: float(p.s), reverse=sign < 0)
        # Distinguish a valid 1-m sampling from an incomplete/disjoint lane.
        station_gaps = [abs(float(b.s)-float(a.s)) for a,b in zip(pts,pts[1:])]
        first = pts[0]
        lanes.append({'lane_uid': lane_id, 'road_id': int(first.road_id),
                      'section_id': int(first.section_id), 'lane_id': int(first.lane_id),
                      'is_junction': bool(first.is_junction),
                      'centerline': [[p.transform.location.x,p.transform.location.y,p.transform.location.z] for p in pts],
                      'width_m': [float(p.lane_width) for p in pts],
                      's_samples_m': [float(p.s) for p in pts],
                      'yaw_samples_deg': [float(p.transform.rotation.yaw) for p in pts],
                      'station_travel_direction': sign,
                      'max_station_gap_m': max(station_gaps,default=0.),
                      'successors': sorted(successors[lane_id]), 'predecessors': [],
                      'topology_source': 'CARLA global grid + route + topology endpoints; no target seeds'})

    # Prevent zero-length XY segments reaching project()/heading estimation.
    # Deterministic map-only cleanup; no actor pose or oracle used.
    pruned_vertices=0
    for lane in lanes:
        xyz=lane['centerline']
        selected=[0]
        for i in range(1,len(xyz)):
            p,q=xyz[selected[-1]],xyz[i]
            if math.hypot(q[0]-p[0],q[1]-p[1])<1.e-4:
                selected[-1]=i
                pruned_vertices+=1
            else:
                selected.append(i)
        if len(selected)<2:
            raise ValueError('Degenerate native lane after near-zero XY cleanup: '+lane['lane_uid'])
        for key in ('centerline','width_m','s_samples_m','yaw_samples_deg'):
            lane[key]=[lane[key][i] for i in selected]
        lane['max_station_gap_m']=max((abs(b-a) for a,b in zip(
            lane['s_samples_m'],lane['s_samples_m'][1:])),default=0.)

    known = {r['lane_uid'] for r in lanes}
    for lane in lanes:
        lane['successors'] = sorted(s for s in lane['successors'] if s != lane['lane_uid'])
    for lane in lanes:
        for other in lane['successors']:
            if other in known:
                next(l for l in lanes if l['lane_uid']==other)['predecessors'].append(lane['lane_uid'])
    typed = [r for r in transitions.values() if r['source'] in known and r['destination'] in known]
    typed.sort(key=lambda x: (x['source'],x['destination'],x['kind'],x['source_s_m']))
    return lanes, {'export_schema_version':'hv_v7_native_map_v2',
                   'query_radius_m': float(radius_m),'sampling_resolution_m': float(step_m),
                   'global_seed_count': len(global_seeds),'topology_seed_count': len(topology_seeds),
                   'route_seed_count': len(route_seeds),'waypoint_probe_count': len(checked),
                   'visited_samples': sum(len(v) for v in buckets.values()),
                   'near_zero_xy_vertices_pruned':pruned_vertices,
                   'lane_count': len(lanes),'boundary_samples': boundary,
                   'typed_transitions': typed,
                   'lane_change_edges': [r for r in typed if r['kind'].startswith('lane_change_')],
                   'lane_change_status': 'CARLA lane marking + RIFT GRP if supplied',
                   'topology_complete': False,
                   'scope': 'Radius-limited typed native lane graph; no GT route seeds'}
