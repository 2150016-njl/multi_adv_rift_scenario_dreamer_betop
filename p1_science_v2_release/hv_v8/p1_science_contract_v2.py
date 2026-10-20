"""P1 source-anchor-first geometry contract. No snapping, projection or GT edit.

This module never resolves a junction overlap by trusting the reverse lookup.
Route vertices must have independent native CARLA/XODR source identities and
pairwise directed station/waypoint.next evidence. Inconclusive => FAIL.
"""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from typing import Any
from .common import require


def lane(w):
    return (int(w.road_id), int(w.section_id), int(w.lane_id))


def ang(a,b):
    return abs((float(a)-float(b)+180.0)%360.0-180.0)


def xyz(a,b):
    return float(a.distance(b))


def directed_link(a,b,allowed_distance_m=5.0):
    """Conservative directed continuity proof, not spatial projection.

    Same-lane: signed OpenDRIVE station advances in direction of travel.
    Lane transition: actual directed CARLA next() steps must reach destination
    lane identity and coordinate. Never assume neighboring lanes connect.
    """
    ka,kb=lane(a),lane(b)
    dist=xyz(a.transform.location,b.transform.location)
    if not math.isfinite(dist) or dist>allowed_distance_m:
        return False,'consecutive_xyz_gap_or_nonfinite'
    if ka==kb:
        delta=float(b.s)-float(a.s)
        signed=delta*(1 if ka[2]<0 else -1)
        if signed < -1e-3:
            return False,'wrong_way_s'
        if signed>allowed_distance_m+1.0:
            return False,'station_jump'
        return True,'same_directed_lane_station'
    # Cross-road or cross-lane requires proof from directed CARLA next().
    # A direct endpoint transition will have a small physical gap.
    if dist<0.02:
        return False,'coincident_but_transition_unproven'
    step=max(0.25,min(dist,2.0))
    try:
        pending=[a];seen=set()
        for depth in range(1,8):
            candidates=[]
            for wp in pending:
                for nxt in wp.next(step):
                    signature=(lane(nxt),round(float(nxt.s),3))
                    if signature in seen:continue
                    seen.add(signature)
                    remaining=xyz(nxt.transform.location,b.transform.location)
                    if lane(nxt)==kb and remaining <= max(0.65,step*0.65):
                        return True,'directed_next_chain'
                    if remaining<=dist+2 and depth*step<=dist+3:
                        candidates.append(nxt)
            pending=candidates[:96]
            if not pending:break
    except Exception as e:
        return False,'next_query_error:'+type(e).__name__+':'+str(e)
    return False,'directed_next_chain_unproven'


def route_waypoints_v2(carla:Any,cmap:Any,scene:dict):
    rows=scene.get('ego_route',[])
    require(isinstance(rows,list) and len(rows)>=2,'missing/short ego_route')
    import hashlib
    road_length={}
    try:
        root=ET.fromstring(cmap.to_opendrive())
        road_length={int(x.attrib['id']):float(x.attrib['length']) for x in root.findall('road')}
    except Exception as e:
        raise RuntimeError('OpenDRIVE road length parse failed: '+repr(e))
    route=[];checks=[]
    for i,r in enumerate(rows[:50]):
        for f in ('x','y','z','yaw_deg','road_id','section_id','lane_id','s','is_junction'):
            require(f in r,'ego_route[%s] missing %s'%(i,f))
        require(all(math.isfinite(float(r[k])) for k in ('x','y','z','yaw_deg','s')),'nonfinite route point')
        expected_key=(int(r['road_id']),int(r['section_id']),int(r['lane_id']))
        loc=carla.Location(x=float(r['x']),y=float(r['y']),z=float(r['z']))
        xodr=cmap.get_waypoint_xodr(expected_key[0],expected_key[2],float(r['s']))
        exact=cmap.get_waypoint(loc,project_to_road=False,lane_type=carla.LaneType.Driving)
        exact_key=lane(exact) if exact is not None else None
        exact_xyz=xyz(exact.transform.location,loc) if exact is not None else None
        exact_yaw=ang(exact.transform.rotation.yaw,r['yaw_deg']) if exact is not None else None
        exact_s=abs(float(exact.s)-float(r['s'])) if exact is not None and exact_key==expected_key else None
        source_s=abs(float(xodr.s)-float(r['s'])) if xodr is not None else None
        source_xyz=xyz(xodr.transform.location,loc) if xodr is not None else None
        source_yaw=ang(xodr.transform.rotation.yaw,r['yaw_deg']) if xodr is not None else None
        source_ok=(xodr is not None and lane(xodr)==expected_key and
                   source_s<=1e-4 and source_xyz<=0.05 and source_yaw<=0.5 and
                   bool(xodr.is_junction)==bool(r['is_junction']) and
                   xodr.lane_type==carla.LaneType.Driving)
        endpoint=False
        if xodr is None and exact is not None and exact_key==expected_key:
            length=road_length.get(expected_key[0]); endpoint=(
                length is not None and abs(float(r['s'])-length)<=1e-4 and
                exact_s is not None and exact_s<=1e-4 and exact_xyz<=0.05 and exact_yaw<=0.5 and
                exact.lane_type==carla.LaneType.Driving)
        if source_ok:
            anchor=xodr;reference='xodr_original_s'
        elif endpoint:
            anchor=exact;reference='verified_road_end_exact'
        else:
            raise ValueError('ego_route[%d] SOURCE_ANCHOR_FAIL road=%s s=%s xodr=%s exact=%s'
                             %(i,expected_key,r['s'],source_s,exact_key))
        exact_matches=(exact is not None and exact_key==expected_key and exact_xyz<=0.30 and
                       exact_yaw<=2.0)
        junction_ambiguity=bool(source_ok and r['is_junction'] is True and
                                bool(xodr.is_junction) and exact is not None and
                                exact_key!=expected_key and exact_xyz<=0.05)
        require(exact_matches or junction_ambiguity,
                'ego_route[%d] exact_lookup_unresolved expected=%s actual=%s source_strict=%s junction=%s'
                %(i,expected_key,exact_key,bool(source_ok),r['is_junction']))
        link_status='first'
        if route:
            linked,link_status=directed_link(route[-1],anchor)
            require(linked,'ego_route[%d] DIRECTED_CONTINUITY_UNPROVEN %s->%s: %s'
                    %(i,lane(route[-1]),lane(anchor),link_status))
        checks.append({
            'route_index':i,'pass':True,'route_identity_pass':True,
            'expected_road_id':expected_key[0],'expected_section_id':expected_key[1],
            'expected_lane_id':expected_key[2],'actual_road_id':lane(anchor)[0],
            'actual_section_id':lane(anchor)[1],'actual_lane_id':lane(anchor)[2],
            'route_reference_kind':reference,'xodr_found':xodr is not None,
            'xodr_source_anchor_pass':bool(source_ok),'xodr_endpoint_exception_verified':endpoint,
            'xodr_source_s_abs_error_m':source_s,'xodr_source_xyz_error_m':source_xyz,
            'xodr_source_yaw_error_deg':source_yaw,'exact_lookup_status':
                ('matching_source_lane' if exact_matches else 'verified_junction_overlap_other_road'),
            'exact_lookup_lane':list(exact_key) if exact_key else None,
            'exact_location_s_abs_error_m':exact_s,
            'exact_location_xyz_error_m':exact_xyz,'exact_location_yaw_error_deg':exact_yaw,
            'directed_link_to_previous':link_status,
            'route_input_yaw_deg':float(r['yaw_deg']),
        })
        route.append(anchor)
    return route,checks
