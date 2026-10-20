#!/usr/bin/env python3
"""Offline mocked regression for proposed P1 contract; not a CARLA PASS."""
import math
import types
from p1_route140_xodr_anchor_patch import patch, OLD_BLOCK

FIXTURE_HEAD = '''def route_waypoints(carla, carla_map, scene):
    out = []
    checks = []
    for i, r in enumerate(scene["ego_route"]):
        expected = carla.Transform(carla.Location(r["x"], r["y"], r["z"]),
                                   carla.Rotation(yaw=r["yaw_deg"]))
        xodr_wp = carla_map.get_waypoint_xodr(r["road_id"], r["lane_id"], r["s"])
        exact_wp = carla_map.get_waypoint(expected.location, project_to_road=False,
                                          lane_type=carla.LaneType.Driving)
        require(exact_wp is not None, "exact lookup not found")
        require(exact_wp.road_id == r["road_id"] and
                exact_wp.section_id == r["section_id"] and
                exact_wp.lane_id == r["lane_id"], "exact lane identity")
        wp = exact_wp
        xodr_boundary_verified = bool(carla_map.boundary_verified)
        check = {"road_id_equal": True, "section_id_equal": True,
                 "lane_id_equal": True,
                 "s_abs_error_m": abs(wp.s - r["s"]),
                 "position_error_m": distance2(wp.transform.location, expected.location),
                 "yaw_error_deg": angle(wp.transform.rotation.yaw, r["yaw_deg"]),
                 "xodr_exact_position_error_m": None if xodr_wp is None else distance2(
                     xodr_wp.transform.location, exact_wp.transform.location),
                 "xodr_exact_yaw_error_deg": None if xodr_wp is None else angle(
                     xodr_wp.transform.rotation.yaw, exact_wp.transform.rotation.yaw)}
        check["route_index"] = i
        check["xodr_found"] = xodr_wp is not None
        check["xodr_endpoint_exception_verified"] = xodr_boundary_verified
'''
FIXTURE_TAIL = '''
        checks.append(check)
        require(check["pass"], "route identity mismatch")
        out.append(wp)
    return out, checks

def cumulative_route(route):
    return {}
'''

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def distance2(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)


def angle(a, b):
    return abs((a-b+180.0) % 360.0-180.0)


class Location:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
    def distance(self, other):
        return math.sqrt((self.x-other.x)**2 +
                         (self.y-other.y)**2 + (self.z-other.z)**2)


class Rotation:
    def __init__(self, yaw):
        self.yaw = yaw


class Transform:
    def __init__(self, location, rotation):
        self.location, self.rotation = location, rotation


class Waypoint:
    def __init__(self, s, x=0, y=0, z=0, yaw=0, road=140, section=0, lane=-3):
        self.s, self.road_id, self.section_id, self.lane_id = s, road, section, lane
        self.transform = Transform(Location(x, y, z), Rotation(yaw))


class Map:
    def __init__(self, xodr, exact, endpoint=False):
        self.xodr, self.exact, self.boundary_verified = xodr, exact, endpoint
    def get_waypoint_xodr(self, road, lane, s):
        return self.xodr
    def get_waypoint(self, location, project_to_road=False, lane_type=None):
        return self.exact


def main():
    fixture = FIXTURE_HEAD + OLD_BLOCK + FIXTURE_TAIL
    changed = patch(fixture)
    require(changed != fixture and 'out.append(xodr_wp if ' in changed,
            'route reference not changed')
    try:
        patch(changed)
        raise AssertionError('double patch unexpectedly accepted')
    except RuntimeError:
        pass

    fake = types.SimpleNamespace(Transform=Transform, Location=Location, Rotation=Rotation,
                                 LaneType=types.SimpleNamespace(Driving=1))
    env = {'require':require, 'distance2':distance2, 'angle':angle}
    exec('from __future__ import annotations\n'+changed, env)
    run = env['route_waypoints']
    p = {'x':0, 'y':0, 'z':0, 'yaw_deg':0, 's':128.94887733459467,
         'road_id':140, 'section_id':0, 'lane_id':-3}

    def scenario(name, xodr, exact, accept=True, endpoint=False):
        error = None
        try:
            out, checks = run(fake, Map(xodr, exact, endpoint), {'ego_route':[p,p]})
            require(all(c['pass'] for c in checks), 'checks are not true')
            require(out[0] is (xodr if xodr is not None else exact),
                    'output route is not source anchor')
            require(checks[0]['exact_location_s_over_005'] ==
                    (abs(exact.s-p['s']) > 0.05), 'exact s discrepancy hidden')
            require(checks[0]['exact_location_s_abs_error_m'] ==
                    abs(exact.s-p['s']), 'exact s error was overwritten')
        except RuntimeError as exc:
            error = exc
        if accept and error is not None:
            raise RuntimeError('{} should pass but failed: {}'.format(name,error))
        if not accept and error is None:
            raise RuntimeError('{} should fail but passed'.format(name))
        print('MOCK_{}={}'.format(name, 'PASS' if accept else 'REJECTED'))

    s=p['s']
    good=Waypoint(s+0.0000057)
    exact=Waypoint(s-0.05094719, x=-0.05077684, yaw=0.00669)
    scenario('TOWN15_RECORD100_STYLE',good,exact)
    scenario('EXACT_MATCH',Waypoint(s),Waypoint(s))
    scenario('WRONG_SOURCE_S',Waypoint(s+0.001),exact,accept=False)
    scenario('WRONG_SOURCE_XYZ',Waypoint(s+0.0000057,x=0.5),exact,accept=False)
    scenario('WRONG_ELEVATION',Waypoint(s,z=1),exact,accept=False)
    scenario('WRONG_XODR_LANE',Waypoint(s,lane=-2),exact,accept=False)
    scenario('WRONG_EXACT_LANE',good,Waypoint(s,road=141),accept=False)
    scenario('WRONG_EXACT_XYZ',good,Waypoint(s,x=0.4),accept=False)
    scenario('XODR_NONE_NOT_ENDPOINT',None,exact,accept=False)
    scenario('VERIFIED_ENDPOINT',None,Waypoint(s),endpoint=True)
    print('P1_ROUTE140_OFFLINE_CONTRACT_POS_NEG_PASS (NOT CARLA PASS)')


if __name__ == '__main__':
    main()
