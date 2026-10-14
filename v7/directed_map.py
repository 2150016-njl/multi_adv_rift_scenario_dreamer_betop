"""Map-only conversion. Preserve identity/unknown edges; never repair by XY proximity."""
from collections import deque
import numpy as np
from .coordinates import world_to_local


def lane_uid(point):
    return '%s:%s:%s'%(point['road_id'],point['section_id'],point['lane_id'])


def project(line,point):
    points=np.asarray(line,dtype=float);p=np.asarray(point,dtype=float)
    if len(points)<2:return None
    delta=points[1:]-points[:-1];length=np.linalg.norm(delta[:,:2],axis=1)
    t=np.clip(((p[:2]-points[:-1,:2])*delta[:,:2]).sum(axis=1)/np.maximum(length**2,1.e-15),0,1)
    q=points[:-1]+t[:,None]*delta
    distances=np.linalg.norm(q[:,:2]-p[:2],axis=1);i=int(np.argmin(distances))
    return {'distance_xy_m':float(distances[i]),'progress_m':float(np.r_[0,np.cumsum(length)][i]+t[i]*length[i]),
            'projected_xyz':q[i].tolist(),'tangent_yaw_rad':float(np.arctan2(delta[i,1],delta[i,0])),
            'segment_index':i,'segment_fraction':float(t[i])}


class DirectedMapBuilder:
    def build(self,lanes,ego,opendrive_sha256):
        ids=[r['lane_uid'] for r in lanes]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate lane identity')
        known=set(ids);nodes=[];edges=[];boundary=[]
        for row in lanes:
            if row['lane_uid']!=lane_uid(row):raise ValueError('Lane ID differs from road/section/lane')
            line=row['centerline'];width=row['width_m']
            if len(line)<2 or len(width)!=len(line) or min(width)<=0:raise ValueError('Invalid lane samples')
            local=world_to_local(ego,line)
            steps=np.linalg.norm(np.diff(np.asarray(local)[:,:2],axis=0),axis=1)
            nodes.append({key:row[key] for key in ('lane_uid','road_id','section_id','lane_id','is_junction')})
            nodes[-1].update(centerline_world_xyz=line,centerline_local_xyz=local,width_m=width,
                successors=list(row.get('successors',[])),predecessors=list(row.get('predecessors',[])),
                directed_progress_m=np.r_[0,np.cumsum(steps)].tolist(),
                opendrive_s_samples=row.get('s_samples_m'),
                yaw_samples_world_deg=row.get('yaw_samples_deg'),
                station_travel_direction=row.get('station_travel_direction'),
                sample_identity_status='station_samples_unavailable_in_legacy_export' if not row.get('s_samples_m') else 'sampled',
                topology_source=row.get('topology_source'),
                travel_direction_status='unverified_legacy_point_order' if not row.get('s_samples_m') else 'station_order_available',
                max_sample_step_m=float(steps.max()))
            for dst in row.get('successors',[]):
                edge={'source':row['lane_uid'],'destination':dst,'destination_exported':dst in known,
                      'evidence':'CARLA Waypoint.next export'}
                edges.append(edge)
                if dst not in known:boundary.append(edge)
        asymmetry=[e for e in edges if e['destination_exported'] and e['source'] not in
                   next(n for n in nodes if n['lane_uid']==e['destination'])['predecessors']]
        return {'lanes':nodes,'connectors':[n['lane_uid'] for n in nodes if n['is_junction']],
                'edges':edges,'boundary_edges':boundary,'reciprocity_mismatches':asymmetry,
                'opendrive_sha256':opendrive_sha256,'topology_complete':False,
                'coverage_status':'legacy_local_export_extent_not_certified',
                'station_identity_complete':all(n['opendrive_s_samples'] is not None for n in nodes)}


def directed_path(graph,start,end):
    known={n['lane_uid'] for n in graph['lanes']}
    if start not in known or end not in known:return None
    edges={n['lane_uid']:[x for x in n['successors'] if x in known] for n in graph['lanes']}
    queue=deque([start]);paths={start:[start]}
    while queue:
        node=queue.popleft()
        if node==end:return paths[node]
        for dst in edges[node]:
            if dst not in paths:paths[dst]=paths[node]+[dst];queue.append(dst)
    return None


def associate_route(graph,local_points,identity_points=None,tolerance=.25):
    result=[];distance=0.;previous=None
    for i,p in enumerate(local_points):
        if previous is not None:distance+=float(np.linalg.norm(np.asarray(p[:2])-np.asarray(previous[:2])))
        candidates=[]
        for lane in graph['lanes']:
            match=project(lane['centerline_local_xyz'],p)
            if match and match['distance_xy_m']<=tolerance:
                if len(p)>2 and abs(p[2]-match['projected_xyz'][2])>1.5:continue
                candidates.append({'lane_uid':lane['lane_uid'],**match})
        identity=None
        if identity_points and all(k in identity_points[i] for k in ('road_id','section_id','lane_id')):
            identity=lane_uid(identity_points[i])
        result.append({'local_xyz':p,'ego_route_distance_m':distance,'declared_lane_uid':identity,
                       'lane_matches':sorted(candidates,key=lambda x:(x['distance_xy_m'],x['lane_uid'])),
                       'identity_status':'serialized' if identity else 'map_association_only',
                       'opendrive_s':identity_points[i].get('s') if identity_points else None})
        previous=p
    return {'waypoints':result,'matching_tolerance_m':tolerance,
            'identity_complete':all(p['declared_lane_uid'] is not None for p in result)}
