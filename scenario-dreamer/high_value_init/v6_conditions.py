"""Lossless condition foundation; no target labels or generated states read."""
from collections import deque
import math


def local_xyz(ego,p):
    x,y=p[0]-ego['x'],p[1]-ego['y'];a=math.radians(ego['yaw_deg'])
    return [math.cos(a)*x+math.sin(a)*y,-math.sin(a)*x+math.cos(a)*y,p[2]-ego['z']]


def make_condition(scene,lanes,context):
    """Preserve directed map and individual actor tokens, with an explicit FOV.

    Approach reachability is a graph condition, not native interaction truth.
    Unknown/boundary topology is retained, never changed into infeasibility.
    """
    ego=scene['ego'];rows=[]
    for lane in lanes:
        row=dict(lane)
        row['centerline']=[local_xyz(ego,p) for p in lane['centerline']]
        rows.append(row)
    ids={r['lane_uid']:r for r in rows}
    route=scene['ego_route']
    roots={('%s:%s:%s' % (p['road_id'],p['section_id'],p['lane_id']))
           for p in route if all(k in p for k in ('road_id','section_id','lane_id'))}&set(ids)
    path={i:[i] for i in roots};queue=deque(sorted(roots))
    while queue:
        node=queue.popleft()
        for pred in ids[node]['predecessors']:
            if pred in ids and pred not in path:
                path[pred]=[pred]+path[node];queue.append(pred)
    approaches=[]
    for uid in sorted(path):
        row=ids[uid]
        visible=[p for p in row['centerline'] if 0<p[0]<=32 and abs(p[1])<=32]
        if visible:
            approaches.append({'lane_uid':uid,'directed_lane_path_to_ego_route':path[uid],
                               'forward_visible_points':visible,'native_interaction_certified':False})
    tokens=[]
    for actor in context:
        xyz=local_xyz(ego,[actor['x'],actor['y'],actor['z']])
        h=math.radians(actor['yaw_deg']-ego['yaw_deg'])
        tokens.append({'actor_token':str(actor['actor_id']),'type_id':actor['type_id'],
                       'state':[xyz[0],xyz[1],math.atan2(math.sin(h),math.cos(h)),
                                actor.get('speed_mps',0.),actor['length'],actor['width'],1.],
                       'z_m':xyz[2],'height_m':actor.get('height'),
                       'velocity_world':[actor.get(k) for k in ('vx','vy','vz')]})
    return {'schema_version':'hv_v6_condition_foundation_v1',
            'ego_route':{'polyline':[local_xyz(ego,[p['x'],p['y'],p['z']])[:2] for p in route]},
            'map':{'lanes_v6':rows,'lane_polylines':[r['centerline'] for r in rows]},
            'actor_tokens':tokens,'observed_agent_states':[t['state'] for t in tokens],
            'output_domain':{'frame':'ego_local','x_min_exclusive':0.,'x_max_inclusive':32.,
                             'y_min_inclusive':-32.,'y_max_inclusive':32.,
                             'contract_only_existing_checkpoint_not_constrained':True},
            'approaches':approaches,
            'topology_route_roots_known':bool(roots),
            'unresolved_successor_edges':sum(s not in ids for r in rows for s in r['successors']),
            'feasibility_status':'graph_approaches_exist' if approaches else
                                 'no_approach_in_exported_graph_NOT_proven_infeasible'}
