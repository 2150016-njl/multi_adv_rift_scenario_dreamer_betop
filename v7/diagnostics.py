"""Oracle is read ONLY here, after map-only candidates are frozen."""
import math
import numpy as np
from .coordinates import world_to_local,state_to_local,wrap,in_domain
from .directed_map import lane_uid,project,directed_path
from .station_audit import actor_station_audit


def legacy_reasons(actor,old_condition,legacy_v6):
    from high_value_init.v5_candidates import make_candidates,closest,get_lanes
    state=actor['local_state'];point=state[:2]
    candidates=make_candidates(old_condition)
    def matches(items):
        metrics=[]
        for item in items:
            d,s,h,_=closest(np.asarray(item['polyline']),point)
            metrics.append({'distance_m':d,'heading_error_rad':abs(wrap(state[2]-h)),
                'before_anchor':s<item['anchor_s']-.25,'lane_index':item['lane_id'],
                'anchor_s':item['anchor_s'],'projected_s':s})
        return metrics
    metrics=matches(candidates)
    near=[m for m in metrics if m['distance_m']<=1.25]
    heading=[m for m in near if m['heading_error_rad']<=.38]
    before=[m for m in heading if m['before_anchor']]
    if before:reason5='covered'
    elif not candidates:reason5='no_legacy_candidates'
    elif not near:
        uncapped=matches(make_candidates(old_condition,max_candidates=100000))
        if any(m['distance_m']<=1.25 and m['heading_error_rad']<=.38 and m['before_anchor'] for m in uncapped):
            reason5='candidate_cap_32'
        else:
            distances=[closest(np.asarray(line)[:,:2],point)[0] for line in get_lanes(old_condition) if len(line)>=2]
            nearest=min(distances,default=float('inf'))
            preview=actor.get('road_geometry',{}).get('preview_waypoint')
            half_width=(preview or {}).get('lane_width_m',0)/2
            reason5=('legacy_anchor_or_fragment_filter_excludes_near_lane' if nearest<=1.25 else
                     'fixed_1p25_distance_gate_narrower_than_lane' if nearest<=half_width else
                     'historical_map_centerline_far_from_GT')
    elif not heading:reason5='heading_error_gt_0p38'
    else:reason5='not_before_legacy_anchor'
    nodes={n['lane_uid']:n for n in legacy_v6['map']['lanes_v6']}
    v6metrics=[]
    for approach in legacy_v6['approaches']:
        lane=nodes[approach['lane_uid']];line=np.asarray(lane['centerline'])[:,:2]
        d,s,h,_=closest(line,point)
        nearest=int(np.argmin(np.linalg.norm(line-np.asarray(point),axis=1)))
        v6metrics.append({'lane_uid':lane['lane_uid'],'distance_m':d,
                          'half_width_m':lane['width_m'][nearest]/2,
                          'heading_error_rad':abs(wrap(state[2]-h))})
    near6=[m for m in v6metrics if m['distance_m']<=m['half_width_m']]
    match6=[m for m in near6 if m['heading_error_rad']<=.38]
    preview=actor.get('road_geometry',{}).get('preview_waypoint')
    uid=lane_uid(preview) if preview else None
    if not legacy_v6['topology_route_roots_known']:reason6='unknown_ego_route_roots'
    elif match6:reason6='covered'
    elif not near6:
        reason6='preview_lane_not_in_legacy_approaches' if uid not in {a['lane_uid'] for a in legacy_v6['approaches']} else 'lane_width_membership_failed'
    else:reason6='heading_error_gt_0p38'
    return {'v5_audit_geometry':{'reported':actor.get('candidate_geometry_supported'),'recomputed':bool(before),
                'reason':reason5,'nearest_candidates':sorted(metrics,key=lambda m:m['distance_m'])[:3],
                'meaning':'legacy audit gates 1.25m/.38rad/anchor; not all V5 training filters'},
            'v6':{'reported':actor.get('v6_approach_geometry_supported'),
                  'recomputed':bool(match6) if legacy_v6['topology_route_roots_known'] else None,
                  'reason':reason6,'preview_lane_uid':uid,
                  'nearest_approaches':sorted(v6metrics,key=lambda m:m['distance_m'])[:3]}}


def inspect_actor(condition,actor,anchors,approaches):
    graph=condition['condition']['directed_map'];nodes={n['lane_uid']:n for n in graph['lanes']}
    ego=condition['condition']['ego_world_pose'];native=actor.get('native') or {}
    preview=(actor.get('road_geometry') or {}).get('preview_waypoint')
    interaction=native.get('interaction_waypoint')
    start=lane_uid(preview) if preview else None;end=lane_uid(interaction) if interaction else None
    trace=native.get('target_route_to_interaction') or []
    ids=[]
    for wp in trace:
        uid=lane_uid(wp)
        if not ids or ids[-1]!=uid:ids.append(uid)
    # Include BOTH terminal links: preview -> trace[0] and trace[-1] -> interaction.
    # No GT/native trace is ever passed back to the condition graph builder.
    full_trace=[uid for uid in [start]+ids+[end] if uid is not None]
    joined=[]
    for uid in full_trace:
        if not joined or uid!=joined[-1]:joined.append(uid)
    missing=[uid for uid in joined if uid not in nodes]
    exported={(e['source'],e['destination']) for e in graph['edges']
              if e.get('destination_exported',True)}
    missing_edges=[{'source':a,'destination':b,'kind':
                    'possible_lane_change_not_in_next_graph'
                    if a.split(':')[:2]==b.split(':')[:2] else 'unexported_transition'}
                   for a,b in zip(joined,joined[1:]) if (a,b) not in exported]
    path=directed_path(graph,start,end)
    anchor_match=[];anchor_projection=None
    if end in nodes and interaction:
        local=world_to_local(ego,[interaction[k] for k in ('x','y','z')])
        anchor_projection=project(nodes[end]['centerline_local_xyz'],local)
        for anchor in anchors:
            if anchor['lane_uid']!=end:continue
            lo,hi=anchor['lane_progress_envelope_m']
            if lo-.25<=anchor_projection['progress_m']<=hi+.25 and anchor_projection['distance_xy_m']<=.25:
                anchor_match.append(anchor['anchor_id'])
    options=[a for a in approaches if a['start_lane_uid']==start and a['anchor_id'] in anchor_match]
    projection=None
    direction_error=None
    if start in nodes and preview:
        query=project(nodes[start]['centerline_world_xyz'],[preview[k] for k in ('x','y','z')])
        direction_error=abs(wrap(math.radians(preview['yaw_deg'])-query['tangent_yaw_rad']))
    if start in nodes:
        projection=project(nodes[start]['centerline_local_xyz'],world_to_local(ego,[actor['pose'][k] for k in ('x','y','z')]))
        projection['heading_error_rad']=abs(wrap(state_to_local(ego,actor['pose'])[2]-projection['tangent_yaw_rad']))
        projection['half_width_m']=nodes[start]['width_m'][projection['segment_index']]/2
    domain=in_domain(actor['local_state'],front=actor['forward_fov'])
    progress_supported=bool(projection and any(any(lo-1.e-6<=projection['progress_m']<=hi+1.e-6
        for lo,hi in a['centerline_front_intervals_m' if actor['forward_fov'] else 'centerline_full_intervals_m']) for a in options))
    failures=[]
    if not preview:failures.append('preview_identity_missing')
    elif start not in nodes:failures.append('preview_lane_missing_from_export')
    if not interaction:failures.append('native_interaction_identity_missing')
    elif end not in nodes:failures.append('interaction_lane_missing_from_export')
    if start in nodes and end in nodes and path is None:failures.append('directed_connectivity_missing_in_export')
    if not anchor_match:failures.append('native_anchor_outside_condition_anchor_envelope')
    if not domain:failures.append('historical_actor_outside_slot1_domain')
    if path and anchor_match and not options:failures.append('start_lane_centerline_outside_output_domain')
    if options and not progress_supported:failures.append('center_projection_outside_initial_intervals')
    topology_support=not failures
    if direction_error is not None and direction_error>math.pi/2:
        failures.append('exported_centerline_direction_opposes_native_waypoint')
    residuals=[]
    for wp in trace:
        uid=lane_uid(wp)
        if uid in nodes:
            q=project(nodes[uid]['centerline_world_xyz'],[wp[k] for k in ('x','y','z')])
            residuals.append(q['distance_xy_m'])
    # GT-only station audit; never leaks into the condition or candidate builder.
    station_audit=actor_station_audit(condition,actor,anchor_projection,anchor_match)
    # This is an upper-level topology/support check, NOT a legal state/chart certificate.
    return {'actor_id':actor['actor_id'],'historical_type_id':actor['type_id'],
            'front':actor['forward_fov'],'native_positive':native.get('native_candidate'),
            'preview_lane_uid':start,'interaction_lane_uid':end,'preview_lane_in_graph':start in nodes,
            'native_route_lane_ids':ids,'native_route_missing_nodes':missing,
            'native_route_missing_edges':missing_edges,'map_directed_path':path,
            'condition_anchor_ids':anchor_match,'native_anchor_projection':anchor_projection,
            'anchor_failure_detail':None if anchor_match else 'interaction_lane_missing' if end not in nodes else
                'interaction_identity_present_geometry_residual_gt_0p25' if anchor_projection['distance_xy_m']>.25 else
                'outside_map_only_route_anchor_envelope',
            'center_projection_to_preview_lane':projection,'initial_progress_supported':progress_supported,
            'preliminary_topology_support':topology_support,
            'preview_lane_direction_error_rad':direction_error,
            'preview_lane_direction_opposed':direction_error is not None and direction_error>math.pi/2,
            'preliminary_directed_support':not failures,'failure_reasons':failures,
            'support_status':'unknown_map_support' if missing or path is None else
                'supported_preliminary' if not failures else 'not_supported_by_current_export_or_candidates',
            'native_route_max_xy_residual_m':max(residuals) if residuals else None,
            'native_trace_station_audit':station_audit.get('trace_audit'),
            'lane_chart_representability':'not_tested','fixed_sedan_geometry_compatible':None,
            'fixed_sedan_raw_spawn_accepted':None,
            'new_native_replay_status':'not_run','station_audit':station_audit}
