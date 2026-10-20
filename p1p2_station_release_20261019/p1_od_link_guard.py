#!/usr/bin/env python3
"""Strict, evidence-only OpenDRIVE road/lane link verifier for adjacent route points.
No projection, snapping, GT modification, or CARLA mutations.
Python 3.8+.  A missing explicit link means UNPROVEN, not illegal.
"""
from __future__ import annotations
import functools
import math
import xml.etree.ElementTree as ET


def _key(w):
    return int(w.road_id), int(w.section_id), int(w.lane_id)


@functools.lru_cache(maxsize=8)
def _read(xodr):
    root = ET.fromstring(xodr)
    roads = {int(r.get('id')): r for r in root.findall('road')}
    junctions = {int(j.get('id')): j for j in root.findall('junction')}
    return roads, junctions


def _find_lane(section, lane_id):
    side = 'right' if int(lane_id) < 0 else 'left'
    for node in section.findall('./%s/lane' % side):
        if int(node.get('id', '0')) == int(lane_id) and node.get('type') == 'driving':
            return node
    return None


def _boundary_lane(road, lane_id, section_index, is_outgoing):
    """Trace real laneSection lane links to the road boundary (not just nearest section).

    is_outgoing=True follows travel direction; False walks from target section
    backward against travel toward the incoming road boundary.
    """
    sections = road.findall('./lanes/laneSection')
    if not sections or not (0 <= int(section_index) < len(sections)):
        return None, None, 'invalid_section'
    increasing = int(lane_id) < 0
    boundary_at_end = increasing if is_outgoing else not increasing
    destination_index = len(sections)-1 if boundary_at_end else 0
    i = int(section_index)
    lane_node = _find_lane(sections[i],lane_id)
    if lane_node is None:return None,None,'source_section_not_driving'
    active_id = int(lane_id)
    trace=[]
    while i != destination_index:
        direction = 1 if destination_index > i else -1
        rel = 'successor' if direction > 0 else 'predecessor'
        link = lane_node.find('./link/' + rel)
        if link is None or link.get('id') is None:
            return None, None, 'missing_intersection_lane_link_'+rel
        next_id = int(link.get('id'))
        i += direction
        nxt = _find_lane(sections[i],next_id)
        if nxt is None:
            return None, None, 'adjacent_section_driving_lane_missing'
        trace.append({'from_lane':active_id,'to_lane':next_id,'to_section':i})
        lane_node=nxt;active_id=next_id
    return lane_node,active_id,trace


def _travel_link(road, lane_id):
    direction = 'successor' if lane_id < 0 else 'predecessor'
    return road.find('./link/' + direction), direction


def verify_pair(xodr, a, b, maximum_advance_m=5.0):
    """Verify one explicit directed OpenDRIVE link (road or junction connector).

    Source and target MUST have independently verified original XODR anchors before
    this function's result can be used in a route-pass decision.
    """
    ka, kb = _key(a), _key(b)
    result = {'status': 'unproven', 'source_lane': list(ka), 'target_lane': list(kb),
              'link_kind': None, 'reason': None}
    if ka == kb:
        result['reason'] = 'same_lane_requires_station_check'
        return result
    try:
        roads, junctions = _read(xodr)
        ra, rb = roads.get(ka[0]), roads.get(kb[0])
        if ra is None or rb is None:
            result['reason'] = 'road_missing'
            return result
        if ka[2] == 0 or kb[2] == 0:
            result['reason'] = 'center_lane_not_drivable'
            return result
        source, boundary_a, trace_a = _boundary_lane(ra, ka[2], ka[1], True)
        target, boundary_b, trace_b = _boundary_lane(rb, kb[2], kb[1], False)
        if source is None or target is None:
            result['reason'] = 'source_or_target_boundary_lane_unproven'
            result['section_transition_details'] = [trace_a,trace_b]
            return result
        result['boundary_source_lane'] = boundary_a
        result['boundary_target_lane'] = boundary_b
        result['section_transition_trace'] = {'source':trace_a,'target_reverse':trace_b}
        # Snapshot's section_id must agree with the OpenDRIVE laneSection
        # containing its original s. Otherwise never trust a link proof.
        for road, k, w in ((ra,ka,a),(rb,kb,b)):
            sections = road.findall('./lanes/laneSection')
            s = float(w.s)
            lo = float(sections[k[1]].get('s'))
            hi = (float(sections[k[1]+1].get('s')) if k[1]+1<len(sections)
                  else float(road.get('length')))
            if not lo-1e-4 <= s <= hi+1e-4:
                result['reason']='original_s_not_in_source_section'
                return result
        length_a, length_b = float(ra.get('length')), float(rb.get('length'))
        left_a = length_a - float(a.s) if ka[2] < 0 else float(a.s)
        inside_b = float(b.s) if kb[2] < 0 else length_b - float(b.s)
        advance = left_a + inside_b
        result.update({'source_to_exit_m': left_a, 'entry_to_target_m': inside_b,
                       'combined_boundary_advance_m': advance})
        if not all(math.isfinite(v) for v in (left_a, inside_b, advance)) or min(left_a, inside_b) < -1e-3:
            result['reason'] = 'invalid_boundary_station'
            return result
        if advance > maximum_advance_m + 0.5:
            result['reason'] = 'boundary_distance_exceeds_route_step'
            return result
        link, direction = _travel_link(ra, ka[2])
        if link is None:
            result['reason'] = 'outgoing_road_link_missing'
            return result
        kind, dest_id = link.get('elementType'), int(link.get('elementId', '-1'))
        entry = 'start' if kb[2] < 0 else 'end'
        if kind == 'road' and dest_id == kb[0]:
            lane_link = source.find('./link/' + direction)
            if lane_link is None or int(lane_link.get('id', '0')) != boundary_b:
                result['reason'] = 'direct_road_lane_link_missing'
                return result
            if link.get('contactPoint') != entry:
                result['reason'] = 'direct_road_contact_point_wrong_direction'
                return result
            result.update({'status': 'verified', 'link_kind': 'road_lane_link',
                           'reason': 'explicit_road_lane_link_and_boundary_stations'})
            return result
        if kind == 'junction' and dest_id in junctions:
            matches = []
            for c in junctions[dest_id].findall('connection'):
                if int(c.get('incomingRoad', '-1')) != ka[0] or int(c.get('connectingRoad', '-1')) != kb[0]:
                    continue
                if c.get('contactPoint') != entry:
                    continue
                if any(int(l.get('from', '0')) == boundary_a and int(l.get('to', '0')) == boundary_b
                       for l in c.findall('laneLink')):
                    matches.append(c.get('id'))
            if len(matches) == 1:
                result.update({'status': 'verified', 'link_kind': 'junction_laneLink',
                               'junction_id': dest_id, 'connection_id': matches[0],
                               'reason': 'explicit_junction_laneLink_and_boundary_stations'})
                return result
            result['reason'] = 'junction_laneLink_missing_or_ambiguous'
            return result
        result['reason'] = 'outgoing_road_link_does_not_reach_target'
        return result
    except (ET.ParseError, ValueError, TypeError, KeyError, OverflowError) as exc:
        result['reason'] = type(exc).__name__ + ': ' + str(exc)
        return result
