"""Target-free directed CARLA map conversion with typed/station transition evidence."""
from collections import deque
import numpy as np
from .coordinates import world_to_local


def lane_uid(point):
    return '%s:%s:%s' % (point['road_id'],point['section_id'],point['lane_id'])


def project(line,point):
    points=np.asarray(line,dtype=float);p=np.asarray(point,dtype=float)
    if len(points)<2:return None
    delta=points[1:]-points[:-1];length=np.linalg.norm(delta[:,:2],axis=1)
    t=np.clip(((p[:2]-points[:-1,:2])*delta[:,:2]).sum(axis=1)/np.maximum(length**2,1.e-15),0,1)
    q=points[:-1]+t[:,None]*delta
    distances=np.linalg.norm(q[:,:2]-p[:2],axis=1);i=int(np.argmin(distances))
    return {'distance_xy_m':float(distances[i]),
            'progress_m':float(np.r_[0,np.cumsum(length)][i]+t[i]*length[i]),
            'projected_xyz':q[i].tolist(),
            'tangent_yaw_rad':float(np.arctan2(delta[i,1],delta[i,0])),
            'segment_index':i,'segment_fraction':float(t[i])}


class DirectedMapBuilder:
    def build(self,lanes,ego,opendrive_sha256,evidence=None):
        evidence=evidence or {}
        ids=[r['lane_uid'] for r in lanes]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate lane identity')
        known=set(ids);nodes=[];edges=[];boundary=[]
        for row in lanes:
            if row['lane_uid']!=lane_uid(row):raise ValueError('Lane ID differs from road/section/lane')
            line=row['centerline'];width=row['width_m']
            if len(line)<2 or len(width)!=len(line) or min(width)<=0:
                raise ValueError('Invalid lane samples')
            local=world_to_local(ego,line)
            steps=np.linalg.norm(np.diff(np.asarray(local)[:,:2],axis=0),axis=1)
            node={key:row[key] for key in ('lane_uid','road_id','section_id','lane_id','is_junction')}
            node.update(centerline_world_xyz=line,centerline_local_xyz=local,width_m=width,
                successors=list(row.get('successors',[])),predecessors=list(row.get('predecessors',[])),
                directed_progress_m=np.r_[0,np.cumsum(steps)].tolist(),
                opendrive_s_samples=row.get('s_samples_m'),
                yaw_samples_world_deg=row.get('yaw_samples_deg'),
                station_travel_direction=row.get('station_travel_direction'),
                max_station_gap_m=row.get('max_station_gap_m'),
                sample_identity_status='station_samples_unavailable_in_legacy_export'
                    if not row.get('s_samples_m') else 'sampled',
                topology_source=row.get('topology_source'),
                travel_direction_status='unverified_legacy_point_order'
                    if not row.get('s_samples_m') else 'station_order_available',
                max_sample_step_m=float(steps.max()))
            nodes.append(node)
            for dst in row.get('successors',[]):
                edge={'source':row['lane_uid'],'destination':dst,
                      'kind':'longitudinal','source_s_m':None,'destination_s_m':None,
                      'destination_exported':dst in known,
                      'evidence':'CARLA Waypoint.next export'}
                edges.append(edge)
                if dst not in known:boundary.append(edge)
        # Station-tagged transitions come from *native map evidence*, not
        # GT trace_route labels. Do not assume an entire lane allows changing.
        for transition in evidence.get('typed_transitions',[]):
            if transition['kind'] not in ('longitudinal','lane_change_left','lane_change_right'):
                raise ValueError('Unknown CARLA transition type: '+str(transition['kind']))
            src,dst=transition['source'],transition['destination']
            if src not in known:continue
            item={k:transition.get(k) for k in ('source','destination','kind','source_s_m','destination_s_m','evidence')}
            item['destination_exported']=dst in known
            edges.append(item)
            if dst not in known:boundary.append(item)
        seen=set();unique=[]
        for edge in edges:
            key=(edge['source'],edge['destination'],edge['kind'],edge.get('source_s_m'),edge.get('destination_s_m'))
            if key not in seen:seen.add(key);unique.append(edge)
        asymmetry=[e for e in unique if e['kind']=='longitudinal' and e['source_s_m'] is None and
                   e['destination_exported'] and e['source'] not in
                   next(n for n in nodes if n['lane_uid']==e['destination'])['predecessors']]
        return {'lanes':nodes,'connectors':[n['lane_uid'] for n in nodes if n['is_junction']],
                'edges':unique,'boundary_edges':boundary,'reciprocity_mismatches':asymmetry,
                'opendrive_sha256':opendrive_sha256,
                'map_export_schema_version':evidence.get('export_schema_version','legacy_map'),
                'typed_edge_count':sum(e['source_s_m'] is not None for e in unique),
                'lane_change_edge_count':sum(e['kind'].startswith('lane_change_') for e in unique),
                'topology_complete':False,
                'coverage_status':'radius_limited_graph_topology_not_certified',
                'station_identity_complete':all(n['opendrive_s_samples'] is not None for n in nodes)}


def directed_path(graph,start,end):
    known={n['lane_uid'] for n in graph['lanes']}
    if start not in known or end not in known:return None
    successors={name:set() for name in known}
    for edge in graph['edges']:
        if edge['source'] in known and edge['destination'] in known:
            successors[edge['source']].add(edge['destination'])
    queue=deque([start]);paths={start:[start]}
    while queue:
        node=queue.popleft()
        if node==end:return paths[node]
        for dst in sorted(successors[node]):
            if dst not in paths:paths[dst]=paths[node]+[dst];queue.append(dst)
    return None


def associate_route(graph,local_points,identity_points=None,tolerance=.75):
    result=[];distance=0.;previous=None
    known={n['lane_uid']:n for n in graph['lanes']}
    for i,p in enumerate(local_points):
        if previous is not None:
            distance+=float(np.linalg.norm(np.asarray(p[:2])-np.asarray(previous[:2])))
        identity=None
        if identity_points and i<len(identity_points) and all(k in identity_points[i] for k in ('road_id','section_id','lane_id')):
            identity=lane_uid(identity_points[i])
        candidates=[]
        for lane in graph['lanes']:
            # Route waypoint identity is authoritative when serialized; do
            # not attach an alternative physically nearby opposing lane.
            if identity and lane['lane_uid']!=identity:continue
            match=project(lane['centerline_local_xyz'],p)
            if match and match['distance_xy_m']<=tolerance:
                if len(p)>2 and abs(p[2]-match['projected_xyz'][2])>1.5:continue
                candidates.append({'lane_uid':lane['lane_uid'],**match})
        result.append({'local_xyz':p,'ego_route_distance_m':distance,'declared_lane_uid':identity,
                       'lane_matches':sorted(candidates,key=lambda x:(x['distance_xy_m'],x['lane_uid'])),
                       'identity_status':'serialized' if identity else 'map_association_only',
                       'opendrive_s':identity_points[i].get('s') if identity_points and i<len(identity_points) else None,
                       'declared_lane_exported':identity in known if identity else None})
        previous=p
    return {'waypoints':result,'matching_tolerance_m':tolerance,
            'identity_complete':all(p['declared_lane_uid'] is not None for p in result)}
