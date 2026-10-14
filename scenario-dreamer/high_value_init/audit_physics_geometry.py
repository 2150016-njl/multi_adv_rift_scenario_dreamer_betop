"""Read-only OpenDRIVE lane ribbons and 3D box diagnostics.

Lane polygons are sampled at a declared resolution, not Unreal road meshes.
Level-box overlap is a conservative test, not a mesh collision certificate.
Actual collision sensor events are collected separately by dynamic probes.
"""
import math
import numpy as np


def box_record(carla, box, transform=None):
    if transform is None:  # get_level_bbs are already in world coordinates.
        transform=carla.Transform(box.location,box.rotation)
        center=np.array([box.location.x,box.location.y,box.location.z],dtype=float)
        axes=np.asarray(transform.get_matrix(),dtype=float)[:3,:3]
        extent=np.array([box.extent.x,box.extent.y,box.extent.z],dtype=float)
        vertices=[(center+axes@(extent*np.array([x,y,z]))).tolist()
                  for x in (-1,1) for y in (-1,1) for z in (-1,1)]
    else:
        matrix=np.asarray(transform.get_matrix(),dtype=float)
        center=(matrix@np.array([box.location.x,box.location.y,box.location.z,1.]))[:3]
        rotation=np.asarray(carla.Transform(rotation=box.rotation).get_matrix(),dtype=float)[:3,:3]
        axes=matrix[:3,:3]@rotation
        extent=np.array([box.extent.x,box.extent.y,box.extent.z],dtype=float)
        vertices=[[float(v.x),float(v.y),float(v.z)] for v in box.get_world_vertices(transform)]
    return {'center':center.tolist(),'axes':axes.tolist(),'extent':extent.tolist(),'vertices':vertices}


def boxes_intersect(a,b,epsilon=1.e-6):
    """Full 15-axis separating-axis test for oriented 3D boxes."""
    ca,cb=np.array(a['center']),np.array(b['center'])
    aa,ab=np.array(a['axes']),np.array(b['axes'])
    ea,eb=np.array(a['extent']),np.array(b['extent'])
    axes=[aa[:,i] for i in range(3)]+[ab[:,i] for i in range(3)]
    axes += [np.cross(aa[:,i],ab[:,j]) for i in range(3) for j in range(3)]
    for axis in axes:
        if np.linalg.norm(axis)<1.e-9:continue
        ra=float(np.abs(axis@aa)@ea);rb=float(np.abs(axis@ab)@eb)
        if abs(float((cb-ca)@axis))>ra+rb+epsilon:return False
    return True


class StaticBoxCollection(list):
    """Batch the first three existing SAT tests; keep the exact narrow test."""
    def __init__(self,entries):
        super().__init__(entries)
        self.centers=np.asarray([e['box']['center'] for e in entries],dtype=float).reshape(-1,3)
        self.axes=np.asarray([e['box']['axes'] for e in entries],dtype=float).reshape(-1,3,3)
        self.extents=np.asarray([e['box']['extent'] for e in entries],dtype=float).reshape(-1,3)
        self.finite=(np.isfinite(self.centers).all(axis=1)&np.isfinite(self.axes).all(axis=(1,2))&
                     np.isfinite(self.extents).all(axis=1))

    def candidates(self,box,epsilon=1.e-6):
        a=np.asarray(box['axes'],dtype=float);e=np.asarray(box['extent'],dtype=float)
        c=np.asarray(box['center'],dtype=float)
        if not len(self):return []
        if not (np.isfinite(a).all() and np.isfinite(c).all() and np.isfinite(e).all()):
            return list(range(len(self)))
        projection=np.abs((self.centers-c)@a)
        rotations=np.einsum('ki,nkj->nij',a,self.axes)
        rb=np.sum(np.abs(rotations)*self.extents[:,None,:],axis=2)
        ra=np.abs(a.T@a)@e
        # Extra roundoff margin only admits additional narrow-phase checks.
        margin=epsilon+1.e-10*(1.+projection+rb+ra)
        active=np.linalg.norm(a,axis=0)>=1.e-9
        keep=np.all((projection<=ra+rb+margin)|~active,axis=1)|~self.finite
        return np.flatnonzero(keep).tolist()


def lane_strip(a,b):
    from shapely.geometry import Polygon
    def edge(p,sign):
        yaw=math.radians(p['yaw_deg']);w=p['lane_width_m']/2.
        return (p['x']-sign*math.sin(yaw)*w,p['y']+sign*math.cos(yaw)*w)
    return Polygon([edge(a,1),edge(b,1),edge(b,-1),edge(a,-1)]).buffer(0)


def whole_box_road(box,segments,z_tolerance=1.5):
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union
    vertices=np.asarray(box['vertices'],dtype=float)
    hull=MultiPoint(vertices[:,:2]).convex_hull
    bottom=float(vertices[:,2].min())
    minx,miny,maxx,maxy=hull.bounds
    polygons=[]
    for segment in segments:
        a,b=segment['points']
        width=max(a['lane_width_m'],b['lane_width_m'])/2.
        if max(a['x'],b['x'])+width<minx or min(a['x'],b['x'])-width>maxx or \
                max(a['y'],b['y'])+width<miny or min(a['y'],b['y'])-width>maxy:
            continue
        if bottom<min(a['z'],b['z'])-z_tolerance or bottom>max(a['z'],b['z'])+z_tolerance:
            continue
        polygons.append(lane_strip(a,b))
    road=unary_union(polygons)
    outside=float(hull.difference(road).area)
    return {'evaluated':True,'whole_box_in_sampled_driving_polygon':outside<=1.e-5,
            'footprint_area_m2':float(hull.area),'outside_area_m2':outside,
            'surface_z_tolerance_m':z_tolerance,'matching_lane_strips':len(polygons),
            'source':'OpenDRIVE width/yaw/elevation ribbons; not Unreal road mesh'}


class MapGeometry:
    def __init__(self,world_map,resolution=1.):
        from .native_state_audit import waypoint_dict
        self.resolution=float(resolution)
        self.segments=[]
        for wp in world_map.generate_waypoints(self.resolution):
            if str(wp.lane_type).rsplit('.',1)[-1]!='Driving':continue
            for nxt in wp.next(self.resolution):
                if str(nxt.lane_type).rsplit('.',1)[-1]!='Driving':continue
                self.segments.append({'points':[waypoint_dict(wp),waypoint_dict(nxt)]})

    def local_lanes(self,ego,radius=100.):
        """Directed sample edges; never infer successors from XY proximity."""
        from collections import defaultdict
        grouped=defaultdict(list)
        def key(p):return '%s:%s:%s' % (p['road_id'],p['section_id'],p['lane_id'])
        for edge in self.segments:
            a,b=edge['points']
            if min(math.hypot(p['x']-ego['x'],p['y']-ego['y']) for p in (a,b))>radius:continue
            grouped[key(a)].append((a,b))
        result=[]
        for uid,edges in sorted(grouped.items()):
            a,b=edges[0];direction=1 if b['s']>=a['s'] else -1
            edges.sort(key=lambda e:direction*e[0]['s'])
            points=[e[0] for e in edges]+[edges[-1][1]]
            successors=sorted({key(y) for x,y in edges if key(y)!=uid})
            result.append({'lane_uid':uid,'successors':successors,'predecessors':[],
                           'centerline':[[p['x'],p['y'],p['z']] for p in points],
                           'width_m':[p['lane_width_m'] for p in points],
                           'road_id':a['road_id'],'section_id':a['section_id'],'lane_id':a['lane_id'],
                           'is_junction':a['is_junction'],'layer':'lane_connector' if a['is_junction'] else 'lane',
                           'topology_source':'CARLA Waypoint.next; local radius boundary retained'})
        # Match exact OpenDRIVE station. Unmatched terminal IDs remain explicit
        # boundary edges; no nearest-neighbour connection is invented.
        ids={r['lane_uid']:r for r in result}
        for r in result:
            for successor in r['successors']:
                if successor in ids:ids[successor]['predecessors'].append(r['lane_uid'])
        return result


def static_boxes(world,carla):
    boxes=[];errors=[]
    categories=('Buildings','Fences','Walls','Poles','TrafficSigns','TrafficLight',
                'Vegetation','GuardRail','Static','Car','Bus','Truck','Motorcycle','Bicycle')
    for name in categories:
        try:
            category=getattr(carla.CityObjectLabel,name)
            for box in world.get_level_bbs(category):
                boxes.append({'category':name,'box':box_record(carla,box)})
        except Exception as exc:
            errors.append('%s: %s: %s' % (name,type(exc).__name__,exc))
    return StaticBoxCollection(boxes),errors


def cohort_geometry(world,carla,actors,geometry,static,progress=None):
    """Use actual blueprint box offsets, height, pitch and roll for every actor."""
    boxes={source:box_record(carla,a.bounding_box,t) for source,(a,t) in actors.items()}
    reports={}
    static_list,static_errors=static
    if not isinstance(static_list,StaticBoxCollection):static_list=StaticBoxCollection(static_list)
    for source,box in boxes.items():
        if progress:progress('whole_road',source_actor_id=source,lane_segments=len(geometry.segments))
        report={'actual_box_world':box,'whole_road':whole_box_road(box,geometry.segments),
                'box_transform_source':'raw_requested_spawn_command_pre_tick; actual blueprint geometry',
                'peer_3d_overlap_ids':[i for i,b in boxes.items() if i!=source and boxes_intersect(box,b)],
                'static_3d_box_overlaps':[],'static_query_errors':static_errors,
                'static_mesh_collision_checked':False,'polygon_sampling_resolution_m':geometry.resolution}
        indices=static_list.candidates(box)
        report['static_level_box_count']=len(static_list)
        report['static_narrow_phase_candidates']=len(indices)
        if progress:progress('static_narrow_phase',source_actor_id=source,total_boxes=len(static_list),candidates=len(indices))
        for i in indices:
            item=static_list[i]
            if boxes_intersect(box,item['box']):
                report['static_3d_box_overlaps'].append({'box_index':i,'category':item['category']})
        reports[source]=report
    return reports
