"""Conservative *oracle-side* station reachability audit, NOT an online generator.

Only map geometry/typed edges define the graph. The actor's original position and
native interaction waypoint are supplied solely by offline diagnostics. A positive
means progress-monotone path in a sampled lane graph, NOT physical Model3 feasibility.
"""
import heapq
import math
from collections import defaultdict

from .directed_map import project


PROGRESS_TOL_M = 0.15
STATION_BORDER_TOL_M = 0.50


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def station_to_progress(lane, station):
    """Interpolate native OpenDRIVE station on exported travel-ordered samples.

    Returns None, never extrapolates beyond a short sample-boundary tolerance.
    Duplicate stations or nonmonotone station data fail closed.
    """
    ss = lane.get('opendrive_s_samples')
    pp = lane.get('directed_progress_m')
    if not ss or not pp or len(ss) != len(pp) or len(ss) < 2 or not _finite(station):
        return None
    if not all(_finite(x) for x in ss + pp):
        return None
    sign = lane.get('station_travel_direction')
    if sign not in (-1, 1):
        return None
    if any((float(b) - float(a))*sign <= 1e-7 for a,b in zip(ss,ss[1:])):
        return None
    if any(float(b) < float(a)-1e-8 for a,b in zip(pp,pp[1:])):
        return None
    lo,hi=min(ss),max(ss)
    s=float(station)
    if s < lo-STATION_BORDER_TOL_M or s > hi+STATION_BORDER_TOL_M:
        return None
    s=max(lo,min(hi,s))
    for i in range(len(ss)-1):
        a,b=float(ss[i]),float(ss[i+1])
        if min(a,b)-1e-9 <= s <= max(a,b)+1e-9:
            t=(s-a)/(b-a)
            return float(pp[i]+t*(pp[i+1]-pp[i]))
    return None


def typed_station_path(graph,start_uid,start_progress,end_uid,end_progress,
                       max_transitions=96):
    """Search station-ordered CARLA typed transitions without oracle hints.

    Dominance: earlier arrival progress at the same lane dominates later one.
    Edges without station identity NEVER grant station-level reachability.
    """
    lanes={r['lane_uid']:r for r in graph['lanes']}
    if start_uid not in lanes or end_uid not in lanes:
        return {'status':'unknown','reason':'start_or_goal_lane_missing'}
    if not (_finite(start_progress) and _finite(end_progress)):
        return {'status':'unknown','reason':'missing_start_or_goal_progress'}
    for uid in (start_uid,end_uid):
        lane=lanes[uid]
        if not lane.get('opendrive_s_samples') or lane.get('station_travel_direction') not in (-1,1):
            return {'status':'unknown','reason':'station_identity_unavailable'}
    if start_uid == end_uid and start_progress <= end_progress+PROGRESS_TOL_M:
        return {'status':'supported','reason':'same_lane_forward','lane_path':[start_uid],
                'typed_transition_count':0,'transition_trace':[]}
    outgoing=defaultdict(list)
    uncertain=0
    for edge in graph['edges']:
        if edge['source'] not in lanes or edge['destination'] not in lanes:
            continue
        if edge.get('kind') not in ('longitudinal','lane_change_left','lane_change_right'):
            continue
        src=station_to_progress(lanes[edge['source']],edge.get('source_s_m'))
        dst=station_to_progress(lanes[edge['destination']],edge.get('destination_s_m'))
        if src is None or dst is None:
            uncertain+=1
            continue
        # Reject impossible same-lane backward jumps from bad native evidence.
        if edge['source']==edge['destination'] and dst < src-PROGRESS_TOL_M:
            uncertain+=1
            continue
        outgoing[edge['source']].append((src,dst,edge))
    # Progress is not a global metric; heap priority only selects reproducible order.
    serial=0
    exhausted_budget=False
    queue=[(0,serial,start_uid,float(start_progress),[start_uid],[])]
    best={start_uid:float(start_progress)}
    while queue:
        steps,_,uid,at,path,trace=heapq.heappop(queue)
        if at > best[uid]+PROGRESS_TOL_M:
            continue
        if uid==end_uid and at <= end_progress+PROGRESS_TOL_M:
            return {'status':'supported','reason':'typed_station_monotone_path',
                    'lane_path':path,'typed_transition_count':len(trace),
                    'transition_trace':trace}
        if steps>=max_transitions:
            exhausted_budget=True
            continue
        for src,dst,edge in outgoing[uid]:
            if src+PROGRESS_TOL_M < at:
                continue
            v=edge['destination']
            if dst < best.get(v,float('inf'))-PROGRESS_TOL_M:
                best[v]=dst; serial+=1
                info={k:edge.get(k) for k in ('source','destination','kind','source_s_m','destination_s_m')}
                heapq.heappush(queue,(steps+1,serial,v,dst,path+[v],trace+[info]))
    # Radius-limited graph -> lack of path is UNKNOWN, never a negative target.
    return {'status':'unknown','reason':'transition_hop_budget_exceeded' if exhausted_budget else
            'no_station_path_in_radius_limited_graph',
            'typed_edges_without_usable_station':uncertain}


def actor_station_audit(condition,actor,anchor_projection=None,anchor_match=None,
                        width_tolerance_m=0.10,heading_tolerance_rad=0.38):
    """Oracle-only check using actual raw pose (NOT preview offset) and interaction.

    Provides separate map-station and geometry-membership flags. It is forbidden
    to use these results in condition/approach candidate generation.
    """
    from .coordinates import world_to_local, wrap
    from .directed_map import lane_uid
    c=condition['condition'];nodes={n['lane_uid']:n for n in c['directed_map']['lanes']}
    preview=(actor.get('road_geometry') or {}).get('preview_waypoint')
    waypoint=(actor.get('native') or {}).get('interaction_waypoint')
    if not preview or not waypoint:
        return {'status':'unknown','reason':'native_endpoint_identity_unavailable',
                'not_physics_certification':True}
    uid=lane_uid(preview);goal_uid=lane_uid(waypoint)
    if uid not in nodes or goal_uid not in nodes:
        return {'status':'unknown','reason':'endpoint_lane_not_exported',
                'not_physics_certification':True}
    ego=c['ego_world_pose'];pose=actor['pose']
    p=project(nodes[uid]['centerline_local_xyz'],world_to_local(ego,[pose[k] for k in ('x','y','z')]))
    q=project(nodes[goal_uid]['centerline_local_xyz'],
              world_to_local(ego,[waypoint[k] for k in ('x','y','z')]))
    if p is None or q is None:
        return {'status':'unknown','reason':'polyline_projection_unavailable',
                'not_physics_certification':True}
    half_width=nodes[uid]['width_m'][min(p['segment_index'],len(nodes[uid]['width_m'])-1)]/2.
    # The native OpenDRIVE station is an independent check on the goal projection.
    # Failure must be UNKNOWN; do not silently extrapolate a projected point.
    station_goal=station_to_progress(nodes[goal_uid],waypoint.get('s'))
    goal_projection_discrepancy=(None if station_goal is None else
        abs(station_goal-q['progress_m']))
    head=abs(wrap(math.radians(pose['yaw_deg']-ego['yaw_deg'])-p['tangent_yaw_rad']))
    lateral_ok=p['distance_xy_m']<=half_width+width_tolerance_m
    heading_ok=head<=heading_tolerance_rad
    # Historical native anchor must coincide with a *condition-only* route envelope.
    match_ids=anchor_match if anchor_match is not None else []
    if not match_ids:
        return {'status':'unknown','reason':'interaction_not_in_condition_anchor_envelope',
                'start_distance_xy_m':p['distance_xy_m'],'half_lane_width_m':half_width,
                'heading_error_rad':head,'lateral_membership':lateral_ok,
                'heading_membership':heading_ok,'not_physics_certification':True}
    if station_goal is None or q['distance_xy_m'] > 0.25 or goal_projection_discrepancy > 0.75:
        path={'status':'unknown','reason':'native_goal_station_or_geometry_inconsistent'}
    else:
        path=typed_station_path(c['directed_map'],uid,p['progress_m'],goal_uid,station_goal)
    # Explicitly distinguish geometrical constraints from station graph connectivity.
    if path['status']=='supported' and not (lateral_ok and heading_ok):
        path.update(status='unknown',reason='initial_lateral_or_heading_membership_unverified')
    path.update({'start_uid':uid,'goal_uid':goal_uid,
                'start_progress_m':p['progress_m'],'goal_progress_m':station_goal,
                'goal_projected_progress_m':q['progress_m'],
                'goal_projected_distance_xy_m':q['distance_xy_m'],
                'goal_station_projection_discrepancy_m':goal_projection_discrepancy,
                'start_distance_xy_m':p['distance_xy_m'],'half_lane_width_m':half_width,
                'heading_error_rad':head,'lateral_membership':lateral_ok,
                'heading_membership':heading_ok,'anchor_match_count':len(match_ids),
                'not_physics_certification':True,
                'meaning':'station-monotone map evidence only; no lane chart / sedan geometry / physics'})
    return path
