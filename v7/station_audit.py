"""Strict oracle-side station representability audit for V7 P2.

This module never feeds GT/oracle results back into condition or approach
generation.  It audits whether the *already serialized native* preview and
interaction waypoints can be represented by the frozen CARLA directed map.

Key rules:
- initial progress is derived from the native preview waypoint station plus
  the raw actor's longitudinal offset to that waypoint; it is not obtained by
  unconstrained nearest-polyline projection;
- lateral membership is measured as signed cross-track distance to the native
  preview waypoint, not Euclidean distance to whichever exported polyline
  segment happens to be nearest;
- typed lane transitions must be represented at their recorded station
  intervals;
- zero-length/non-monotone station samples fail closed;
- graph support is map evidence only, never a Model3/physics certificate.
"""
import heapq
import math
from collections import defaultdict

from .directed_map import project


PROGRESS_TOL_M = 0.15
STATION_BORDER_TOL_M = 0.50
TRACE_TRANSITION_TOL_M = 2.0
PREVIEW_LONGITUDINAL_TOL_M = 4.0
LANE_WIDTH_TOL_M = 0.10
HEADING_TOL_RAD = 0.38


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _station_bracket(lane, station):
    ss = lane.get('opendrive_s_samples')
    if not ss or len(ss) < 2 or not _finite(station):
        return None
    if any(not _finite(x) for x in ss):
        return None
    direction = lane.get('station_travel_direction')
    if direction not in (-1, 1):
        return None
    if any((float(b)-float(a))*direction <= 1.e-7 for a,b in zip(ss,ss[1:])):
        return None
    s=float(station)
    lo,hi=min(ss),max(ss)
    if s < lo-STATION_BORDER_TOL_M or s > hi+STATION_BORDER_TOL_M:
        return None
    s=max(lo,min(hi,s))
    for i,(a,b) in enumerate(zip(ss,ss[1:])):
        a=float(a);b=float(b)
        if min(a,b)-1.e-9 <= s <= max(a,b)+1.e-9:
            den=b-a
            if abs(den)<=1.e-12:
                return None
            return i,(s-a)/den,s
    return None


def station_to_progress(lane, station):
    """Interpolate travel-ordered progress at an OpenDRIVE station."""
    pp=lane.get('directed_progress_m')
    ss=lane.get('opendrive_s_samples')
    if not ss or not pp or len(ss)!=len(pp) or len(ss)<2:
        return None
    if any(not _finite(x) for x in pp):
        return None
    if any(float(b)<float(a)-1.e-8 for a,b in zip(pp,pp[1:])):
        return None
    bracket=_station_bracket(lane,station)
    if bracket is None:
        return None
    i,t,_=bracket
    return float(pp[i]+t*(float(pp[i+1])-float(pp[i])))


def station_to_value(lane, station, key):
    """Linearly interpolate a scalar sample aligned with OpenDRIVE station."""
    values=lane.get(key)
    ss=lane.get('opendrive_s_samples')
    if not values or not ss or len(values)!=len(ss):
        return None
    bracket=_station_bracket(lane,station)
    if bracket is None:
        return None
    i,t,_=bracket
    a=float(values[i]);b=float(values[i+1])
    if not (_finite(a) and _finite(b)):
        return None
    return float(a+t*(b-a))


def _transition_candidates(graph, source, destination, source_s, destination_s,
                           tol=TRACE_TRANSITION_TOL_M):
    out=[]
    for edge in graph.get('edges',[]):
        if edge.get('source')!=source or edge.get('destination')!=destination:
            continue
        if edge.get('kind') not in ('longitudinal','lane_change_left','lane_change_right'):
            continue
        es=edge.get('source_s_m'); ed=edge.get('destination_s_m')
        if not (_finite(es) and _finite(ed) and _finite(source_s) and _finite(destination_s)):
            continue
        ds=abs(float(es)-float(source_s))
        dd=abs(float(ed)-float(destination_s))
        if ds<=tol and dd<=tol:
            out.append((ds+dd,edge))
    out.sort(key=lambda x:(x[0],x[1].get('kind',''),x[1].get('source_s_m',0.)))
    return [e for _,e in out]


def trace_station_audit(condition, actor, preview, interaction, trace):
    """Verify the serialized native preview->trace->interaction chain.

    Same-lane motion is checked for travel-direction monotonicity. A lane
    change requires a typed CARLA lane-change edge whose endpoint stations are
    close to the corresponding native trace transition. This is an audit, not
    a source of candidate edges.
    """
    graph=condition['condition']['directed_map']
    lanes={x['lane_uid']:x for x in graph['lanes']}
    waypoints=[preview]+list(trace or [])+[interaction]
    compact=[]
    for wp in waypoints:
        if not wp:
            continue
        uid='%s:%s:%s'%(wp['road_id'],wp['section_id'],wp['lane_id'])
        item={'lane_uid':uid,'s':wp.get('s'),'x':wp.get('x'),'y':wp.get('y'),
              'yaw_deg':wp.get('yaw_deg')}
        if not compact or compact[-1]['lane_uid']!=uid or compact[-1]['s']!=item['s']:
            compact.append(item)
    missing=[x['lane_uid'] for x in compact if x['lane_uid'] not in lanes]
    if missing:
        return {'status':'unknown','reason':'trace_lane_missing_from_export',
                'missing_lane_uids':missing,'transitions':[],'waypoint_count':len(compact)}
    if any(not _finite(x['s']) for x in compact):
        return {'status':'unknown','reason':'trace_station_missing',
                'transitions':[],'waypoint_count':len(compact)}
    transitions=[]
    for a,b in zip(compact,compact[1:]):
        la,lb=lanes[a['lane_uid']],lanes[b['lane_uid']]
        sa,sb=float(a['s']),float(b['s'])
        pa=station_to_progress(la,sa);pb=station_to_progress(lb,sb)
        if pa is None or pb is None:
            transitions.append({'source':a,'destination':b,'status':'unknown',
                                'reason':'trace_station_outside_export_span'})
            continue
        if a['lane_uid']==b['lane_uid']:
            if pb+PROGRESS_TOL_M < pa:
                transitions.append({'source':a,'destination':b,'status':'unknown',
                                    'reason':'same_lane_trace_goes_backward'})
            else:
                transitions.append({'source':a,'destination':b,'status':'supported',
                                    'kind':'longitudinal','source_progress_m':pa,
                                    'destination_progress_m':pb})
        else:
            matches=_transition_candidates(graph,a['lane_uid'],b['lane_uid'],sa,sb)
            if not matches:
                # Do not accept a generic directed edge for a cross-lane
                # transition. P2 requires a typed lane-change occurrence.
                transitions.append({'source':a,'destination':b,'status':'unknown',
                                    'reason':'typed_lane_change_missing',
                                    'candidate_edges':[
                                        {'kind':e.get('kind'),'source_s_m':e.get('source_s_m'),
                                         'destination_s_m':e.get('destination_s_m')}
                                        for e in graph.get('edges',[])
                                        if e.get('source')==a['lane_uid'] and
                                           e.get('destination')==b['lane_uid']
                                    ]})
            else:
                transitions.append({'source':a,'destination':b,'status':'supported',
                                    'kind':matches[0].get('kind'),
                                    'matched_transition':matches[0]})
    bad=[x for x in transitions if x['status']!='supported']
    return {'status':'unknown' if bad else 'supported',
            'reason':bad[0]['reason'] if bad else 'native_trace_represented',
            'transitions':transitions,'waypoint_count':len(compact)}


def typed_station_path(graph,start_uid,start_progress,end_uid,end_progress,
                       max_transitions=96):
    """Search station-ordered CARLA typed transitions without oracle hints."""
    lanes={r['lane_uid']:r for r in graph['lanes']}
    if start_uid not in lanes or end_uid not in lanes:
        return {'status':'unknown','reason':'start_or_goal_lane_missing'}
    if not (_finite(start_progress) and _finite(end_progress)):
        return {'status':'unknown','reason':'missing_start_or_goal_progress'}
    for uid in (start_uid,end_uid):
        lane=lanes[uid]
        if (not lane.get('opendrive_s_samples') or
                lane.get('station_travel_direction') not in (-1,1)):
            return {'status':'unknown','reason':'station_identity_unavailable'}
    if start_uid==end_uid and start_progress<=end_progress+PROGRESS_TOL_M:
        return {'status':'supported','reason':'same_lane_forward','lane_path':[start_uid],
                'typed_transition_count':0,'transition_trace':[]}
    outgoing=defaultdict(list);uncertain=0
    for edge in graph['edges']:
        if edge.get('source') not in lanes or edge.get('destination') not in lanes:
            continue
        if edge.get('kind') not in ('longitudinal','lane_change_left','lane_change_right'):
            continue
        src=station_to_progress(lanes[edge['source']],edge.get('source_s_m'))
        dst=station_to_progress(lanes[edge['destination']],edge.get('destination_s_m'))
        if src is None or dst is None:
            uncertain+=1
            continue
        if edge['source']==edge['destination'] and dst<src-PROGRESS_TOL_M:
            uncertain+=1
            continue
        outgoing[edge['source']].append((src,dst,edge))
    serial=0;exhausted=False
    queue=[(0,serial,start_uid,float(start_progress),[start_uid],[])]
    # Store the earliest progress reached on each lane. A later arrival is
    # never more useful for a progress-monotone path to a fixed goal.
    best={start_uid:float(start_progress)}
    while queue:
        steps,_,uid,at,path,trace=heapq.heappop(queue)
        if at>best[uid]+PROGRESS_TOL_M:
            continue
        if uid==end_uid and at<=end_progress+PROGRESS_TOL_M:
            return {'status':'supported','reason':'typed_station_monotone_path',
                    'lane_path':path,'typed_transition_count':len(trace),
                    'transition_trace':trace}
        if steps>=max_transitions:
            exhausted=True
            continue
        for src,dst,edge in outgoing[uid]:
            if src+PROGRESS_TOL_M<at:
                continue
            v=edge['destination']
            if dst<best.get(v,float('inf'))-PROGRESS_TOL_M:
                best[v]=dst;serial+=1
                info={k:edge.get(k) for k in
                      ('source','destination','kind','source_s_m','destination_s_m')}
                heapq.heappush(queue,(steps+1,serial,v,dst,path+[v],trace+[info]))
    return {'status':'unknown',
            'reason':'transition_hop_budget_exceeded' if exhausted else
                     'no_station_path_in_radius_limited_graph',
            'typed_edges_without_usable_station':uncertain}


def _native_initial_projection(lane, actor_pose, preview):
    """Use native preview identity + raw pose offset to locate the initial state."""
    s=preview.get('s')
    pp=station_to_progress(lane,s)
    if pp is None:
        return None
    yaw=math.radians(float(preview['yaw_deg']))
    dx=float(actor_pose['x'])-float(preview['x'])
    dy=float(actor_pose['y'])-float(preview['y'])
    longitudinal=dx*math.cos(yaw)+dy*math.sin(yaw)
    lateral=-dx*math.sin(yaw)+dy*math.cos(yaw)
    width=station_to_value(lane,s,'width_m')
    if width is None:
        return None
    return {'preview_station_m':float(s),'preview_progress_m':pp,
            'initial_progress_m':float(pp+longitudinal),
            'longitudinal_offset_m':float(longitudinal),
            'signed_lateral_offset_m':float(lateral),
            'lateral_distance_m':abs(float(lateral)),
            'lane_width_m':float(width),'half_width_m':float(width)/2.,
            'heading_error_rad':abs(_wrap(math.radians(
                float(actor_pose['yaw_deg'])-float(preview['yaw_deg']))))}


def _wrap(x):
    return (x+math.pi)%(2.*math.pi)-math.pi


def actor_station_audit(condition,actor,anchor_projection=None,anchor_match=None,
                        width_tolerance_m=LANE_WIDTH_TOL_M,
                        heading_tolerance_rad=HEADING_TOL_RAD):
    """Strict station audit using native preview identity and raw pose."""
    from .coordinates import world_to_local
    from .directed_map import lane_uid
    c=condition['condition']
    nodes={n['lane_uid']:n for n in c['directed_map']['lanes']}
    preview=(actor.get('road_geometry') or {}).get('preview_waypoint')
    waypoint=(actor.get('native') or {}).get('interaction_waypoint')
    trace=(actor.get('native') or {}).get('target_route_to_interaction') or []
    if not preview or not waypoint:
        return {'status':'unknown','reason':'native_endpoint_identity_unavailable',
                'not_physics_certification':True}
    uid=lane_uid(preview);goal_uid=lane_uid(waypoint)
    if uid not in nodes or goal_uid not in nodes:
        return {'status':'unknown','reason':'endpoint_lane_not_exported',
                'not_physics_certification':True}
    ego=c['ego_world_pose'];pose=actor['pose']
    initial=_native_initial_projection(nodes[uid],pose,preview)
    if initial is None:
        return {'status':'unknown','reason':'native_preview_station_unrepresentable',
                'not_physics_certification':True}
    goal_local=world_to_local(ego,[waypoint[k] for k in ('x','y','z')])
    q=project(nodes[goal_uid]['centerline_local_xyz'],goal_local)
    if q is None:
        return {'status':'unknown','reason':'goal_polyline_projection_unavailable',
                'not_physics_certification':True}
    station_goal=station_to_progress(nodes[goal_uid],waypoint.get('s'))
    goal_projection_discrepancy=(None if station_goal is None else
                                  abs(station_goal-q['progress_m']))
    # The native trace is a required fidelity witness. It is not used to
    # manufacture a missing edge.
    trace_check=trace_station_audit(condition,actor,preview,waypoint,trace)
    match_ids=anchor_match if anchor_match is not None else []
    if not match_ids:
        return {'status':'unknown','reason':'interaction_not_in_condition_anchor_envelope',
                **initial,'trace_audit':trace_check,
                'not_physics_certification':True}
    if station_goal is None or q['distance_xy_m']>0.25 or goal_projection_discrepancy>0.75:
        path={'status':'unknown','reason':'native_goal_station_or_geometry_inconsistent'}
    else:
        path=typed_station_path(c['directed_map'],uid,initial['initial_progress_m'],
                                goal_uid,station_goal)
    lateral_ok=initial['lateral_distance_m']<=initial['half_width_m']+width_tolerance_m
    heading_ok=initial['heading_error_rad']<=heading_tolerance_rad
    if path['status']=='supported' and trace_check['status']!='supported':
        path.update(status='unknown',reason='native_preview_to_trace_not_represented')
    if path['status']=='supported' and not lateral_ok:
        path.update(status='unknown',reason='initial_lateral_membership_failed')
    if path['status']=='supported' and not heading_ok:
        path.update(status='unknown',reason='initial_heading_membership_failed')
    # Cross-check the old unconstrained polyline projection, but never use it
    # as the primary identity. This exposes branch/projection errors instead
    # of turning them into false negatives.
    actor_local=world_to_local(ego,[pose[k] for k in ('x','y','z')])
    center_projection=project(nodes[uid]['centerline_local_xyz'],actor_local)
    result=dict(path)
    result.update({'start_uid':uid,'goal_uid':goal_uid,
                   'start_progress_m':initial['initial_progress_m'],
                   'preview_progress_m':initial['preview_progress_m'],
                   'preview_station_m':initial['preview_station_m'],
                   'longitudinal_offset_m':initial['longitudinal_offset_m'],
                   'signed_lateral_offset_m':initial['signed_lateral_offset_m'],
                   'lateral_distance_m':initial['lateral_distance_m'],
                   'half_width_m':initial['half_width_m'],
                   'lane_width_m':initial['lane_width_m'],
                   'heading_error_rad':initial['heading_error_rad'],
                   'lateral_membership':lateral_ok,
                   'heading_membership':heading_ok,
                   'goal_progress_m':station_goal,
                   'goal_projected_progress_m':q['progress_m'],
                   'goal_projected_distance_xy_m':q['distance_xy_m'],
                   'goal_station_projection_discrepancy_m':goal_projection_discrepancy,
                   'anchor_match_count':len(match_ids),
                   'trace_audit':trace_check,
                   'center_projection_to_preview_lane':center_projection,
                   'center_projection_discrepancy_m':(
                       None if center_projection is None else
                       abs(center_projection['progress_m']-initial['initial_progress_m'])),
                   'not_physics_certification':True,
                   'meaning':'native-station map evidence + raw-pose lane membership; no Model3 physics'})
    return result
