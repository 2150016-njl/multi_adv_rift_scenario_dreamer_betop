"""Live map-only collector with station identity. Simulator dependency is deferred."""
from collections import defaultdict,deque
import math


def export_directed_lanes(world_map,ego,route_waypoints,radius_m=100.,step_m=1.,max_samples=250000,planner=None):
    from high_value_init.native_state_audit import waypoint_dict
    def uid(w):return '%s:%s:%s'%(w.road_id,w.section_id,w.lane_id)
    def driving(w):return w is not None and str(w.lane_type).rsplit('.',1)[-1]=='Driving'
    def inside(w):
        p=w.transform.location
        return math.hypot(p.x-ego['x'],p.y-ego['y'])<=radius_m
    # Global sampled seeds can fail for large-map exports; independently seed
    # from the observed ego future route. Neither seed path reads a target.
    queue=deque(w for w in world_map.generate_waypoints(step_m) if driving(w) and inside(w))
    queue.extend(w for w in route_waypoints if driving(w) and inside(w))
    nodes=defaultdict(dict);successors=defaultdict(set);seen=set();boundary=[];station_directions=defaultdict(set)
    while queue:
        w=queue.popleft();key=(uid(w),round(w.s,3))
        if key in seen:continue
        if len(seen)>=max_samples:
            raise RuntimeError('Map sample budget reached; incomplete map must not be silently accepted')
        seen.add(key)
        if not inside(w):continue
        nodes[uid(w)][round(w.s,3)]=waypoint_dict(w)
        for direction in ('next','previous'):
            for other in getattr(w,direction)(step_m):
                if not driving(other):continue
                src,dst=(w,other) if direction=='next' else (other,w)
                if uid(src)!=uid(dst):successors[uid(src)].add(uid(dst))
                elif abs(dst.s-src.s)>1.e-6:
                    station_directions[uid(src)].add(1 if dst.s>src.s else -1)
                if inside(other):queue.append(other)
                else:boundary.append({'source':uid(src),'destination':uid(dst),'kind':'query_radius_boundary',
                                      'source_s':src.s,'destination_s':dst.s})
    lanes=[]
    for lane_id,points in sorted(nodes.items()):
        samples=list(points.values())
        if len(samples)<2:continue
        directions=station_directions[lane_id]
        if len(directions)!=1:
            raise ValueError('Cannot establish unambiguous same-lane station direction: '+lane_id)
        sign=next(iter(directions))
        # Never compare station s across a successor road/section reset.
        samples.sort(key=lambda p:p['s'],reverse=sign<0)
        first=samples[0]
        lanes.append({'lane_uid':lane_id,'road_id':first['road_id'],'section_id':first['section_id'],
            'lane_id':first['lane_id'],'is_junction':first['is_junction'],
            'centerline':[[p[k] for k in ('x','y','z')] for p in samples],
            'width_m':[p['lane_width_m'] for p in samples],'s_samples_m':[p['s'] for p in samples],
            'yaw_samples_deg':[p['yaw_deg'] for p in samples],'station_travel_direction':sign,
            'successors':sorted(successors[lane_id]),'predecessors':[],
            'topology_source':'CARLA next/previous; global plus ego-route seeds; no GT seeds'})
    by_id={r['lane_uid']:r for r in lanes}
    for lane in lanes:
        for nxt in lane['successors']:
            if nxt in by_id:by_id[nxt]['predecessors'].append(lane['lane_uid'])
    lane_changes=[]
    if planner is not None:
        for _,_,edge in planner._graph.edges(data=True):
            if 'CHANGELANE' not in str(edge.get('type')):continue
            a,b=edge['entry_waypoint'],edge['exit_waypoint']
            if inside(a) or inside(b):
                lane_changes.append({'source':uid(a),'destination':uid(b),'kind':str(edge['type']),
                    'source_waypoint':waypoint_dict(a),'destination_waypoint':waypoint_dict(b),
                    'evidence':'existing RIFT GRP CARLA lane-marking map links; no target route used'})
    return lanes,{'query_radius_m':radius_m,'sampling_resolution_m':step_m,
        'visited_samples':len(seen),'boundary_samples':boundary,'lane_change_edges':lane_changes,
        'lane_change_status':'map_planner_links_collected_separately_not_merged_into_next_graph' if planner else 'not_collected',
        'topology_complete':False,'scope':'radius-limited next graph; not full-road-network completeness'}
