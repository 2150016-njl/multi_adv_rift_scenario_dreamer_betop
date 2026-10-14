"""Preliminary map-only support, separate from lane-chart or native certification."""
from collections import deque
from .coordinates import in_domain


def intervals(line,front=False):
    """Clip directed centerline segments against output rectangle, preserving arc s."""
    import math
    answer=[];progress=0.
    for a,b in zip(line,line[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        lo,hi=0.,1.;xmin=0. if front else -32.
        for direction,offset in [(-dx,a[0]-xmin),(dx,32.-a[0]),(-dy,a[1]+32.),(dy,32.-a[1])]:
            if abs(direction)<1.e-12:
                if offset<0:lo,hi=1.,0.;break
            elif direction<0:lo=max(lo,offset/direction)
            else:hi=min(hi,offset/direction)
        if lo<=hi and length>1.e-9:
            midpoint=[a[0]+(lo+hi)/2*dx,a[1]+(lo+hi)/2*dy]
            if in_domain(midpoint,front):
                item=[progress+lo*length,progress+hi*length]
                if answer and abs(answer[-1][1]-item[0])<1.e-7:answer[-1][1]=item[1]
                else:answer.append(item)
        progress+=length
    return answer


class ConflictAnchorBuilder:
    def build(self,condition):
        # No GT/native interaction point is read. Anchors retain the entire saved
        # ego future route, including portions outside the generation rectangle.
        grouped={}
        for point in condition['condition']['ego_route']['waypoints']:
            matches=point['lane_matches']
            declared=point['declared_lane_uid']
            for match in matches:
                if declared and match['lane_uid']!=declared:continue
                uid=match['lane_uid']
                group=grouped.setdefault(uid,{'anchor_id':'route-lane:'+uid,'lane_uid':uid,
                    'route_samples':[],'native_certified':False,
                    'support_kind':'ego_route_lane_sample_envelope'})
                group['route_samples'].append({'ego_route_distance_m':point['ego_route_distance_m'],
                    'lane_progress_m':match['progress_m'],'local_xyz':match['projected_xyz']})
        for group in grouped.values():
            samples=group['route_samples']
            group['lane_progress_envelope_m']=[min(p['lane_progress_m'] for p in samples),max(p['lane_progress_m'] for p in samples)]
            group['envelope_not_exact_continuous_anchor_certificate']=True
        return list(grouped.values())


class ApproachBuilder:
    def build(self,condition,anchors):
        graph=condition['condition']['directed_map'];nodes={n['lane_uid']:n for n in graph['lanes']}
        reverse={uid:[] for uid in nodes}
        for edge in graph['edges']:
            if edge['destination'] in nodes:reverse[edge['destination']].append(edge['source'])
        output=[]
        for anchor in anchors:
            end=anchor['lane_uid'];queue=deque([end]);paths={end:[end]}
            while queue:
                node=queue.popleft()
                for predecessor in sorted(reverse[node]):
                    if predecessor not in paths:
                        paths[predecessor]=[predecessor]+paths[node];queue.append(predecessor)
            for uid,path in sorted(paths.items()):
                lane=nodes[uid];front=intervals(lane['centerline_local_xyz'],True)
                full=intervals(lane['centerline_local_xyz'],False)
                if not full:continue
                output.append({'approach_id':uid+'->'+anchor['anchor_id'],
                    'start_lane_uid':uid,'anchor_id':anchor['anchor_id'],
                    'directed_lane_path':path,'centerline_front_intervals_m':front,
                    'centerline_full_intervals_m':full,'native_certified':False,
                    'representation_status':'pre_chart_support_only'})
        return output
