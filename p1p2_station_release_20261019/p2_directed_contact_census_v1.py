#!/usr/bin/env python3
"""Read-only CARLA 0.9.15 Town-local directed lane-edge contact sample census.

Samples topology endpoint.next(2m) and initial waypoint.next(2m), keeps
ONLY explicit CARLA successor waypoint pairs within 3m whose lane-edge is in
frozen topology. Station samples do not prove exact OpenDRIVE contact points,
longitudinal continuous occupancy, or dynamic feasibility.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,sys
from pathlib import Path

MANIFEST_SHA='9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'
class Failure(RuntimeError):pass
def need(b,why):
    if not b:raise Failure(why)
def lane(w):return (int(w.road_id),int(w.section_id),int(w.lane_id))
def dist(a,b):
    u=a.transform.location;v=b.transform.location
    return math.sqrt((u.x-v.x)**2+(u.y-v.y)**2+(u.z-v.z)**2)
def point(w):
    p=w.transform.location
    return {'lane':list(lane(w)),'s':float(w.s),'xyz':[float(p.x),float(p.y),float(p.z)],
            'yaw_deg':float(w.transform.rotation.yaw)}

def collect(cmap,carla,graph):
    allowed={(tuple(a['from']),tuple(a['to'])) for a in graph['directed_edges']}
    seen={}; errors=[]; n=0
    for a,b in cmap.get_topology():
        # Known CARLA get_topology endpoints can span long road segments.
        # The sample is the actual next() successor, not the distant endpoint.
        for start in (a,b):
            if start.lane_type!=carla.LaneType.Driving:continue
            try: successors=start.next(2.0)
            except RuntimeError as e:
                errors.append(type(e).__name__+': '+str(e)); continue
            for finish in successors:
                if finish.lane_type!=carla.LaneType.Driving:continue
                edge=(lane(start),lane(finish))
                if edge not in allowed or edge[0]==edge[1]:continue
                dd=dist(start,finish)
                if dd>3.0 or not math.isfinite(dd):continue
                n+=1
                key=(edge[0],edge[1],round(float(start.s),3),round(float(finish.s),3))
                seen[key]={'from_lane':list(edge[0]),'to_lane':list(edge[1]),
                           'source':point(start),'target':point(finish),
                           'distance_m':dd,'method':'CARLA_waypoint_next_2m_local_successor'}
    out=sorted(seen.values(),key=lambda x:(x['from_lane'],x['to_lane'],x['source']['s'],x['target']['s']))
    return {'samples':out,'num_unique_frozen_cross_lane_edges_with_samples':len({(tuple(s['from_lane']),tuple(s['to_lane'])) for s in out}),
            'num_frozen_graph_edges':len(allowed),'num_raw_successor_edges':n,
            'CARLA_next_errors':errors}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for k in ('root','topology','town','output'):a.add_argument('--'+k,required=True)
    a.add_argument('--host',default='127.0.0.1');a.add_argument('--port',type=int,default=31120)
    x=a.parse_args()
    try:
        root=Path(x.root).resolve();out=Path(x.output)
        need(not out.exists(),'refuse overwrite')
        mp=root/'data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json'
        need(hashlib.sha256(mp.read_bytes()).hexdigest()==MANIFEST_SHA,'map identity manifest changed')
        expect={z['town']:z['opendrive_sha256'] for z in json.loads(mp.read_text())['maps']}
        need(x.town in expect,'unknown Town')
        graph=json.loads(Path(x.topology).read_text())
        need(graph['town']==x.town and graph['map_sha256']==expect[x.town],'topology map wrong Town/SHA')
        egg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
        need(egg.exists(),'pinned CARLA egg missing')
        sys.path.insert(0,str(egg));import carla
        need(str(carla.__file__).startswith(str(egg)+'/'),'wrong CARLA import')
        client=carla.Client(x.host,x.port);client.set_timeout(120)
        need(client.get_server_version()==client.get_client_version()=='0.9.15','CARLA version changed')
        world=client.get_world();cm=world.get_map()
        need(cm.name.split('/')[-1]==x.town,'wrong world Town; NO map switch')
        ms=hashlib.sha256(cm.to_opendrive().encode()).hexdigest()
        need(ms==expect[x.town],'CARLA map SHA changed')
        c=collect(cm,carla,graph)
        after=client.get_world();mp2=after.get_map()
        need(after.id==world.id and mp2.name==cm.name and
             hashlib.sha256(mp2.to_opendrive().encode()).hexdigest()==ms,
             'world episode changed during topology sampling')
        doc={'schema':'p2_CARLA_local_directed_contact_samples_v1','town':x.town,
             'map_sha256':ms,'topology_sha256':hashlib.sha256(Path(x.topology).read_bytes()).hexdigest(),
             'episode_id':int(world.id),'GT_used':False,'not_continuous_trajectory_proof':True,**c}
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as f:json.dump(doc,f,ensure_ascii=False,indent=2);f.write('\n')
        print('P2_CARLA_CONTACT_EDGES_WITH_SAMPLES='+str(c['num_unique_frozen_cross_lane_edges_with_samples']))
        print('P2_CARLA_CONTACT_ERROR_COUNT='+str(len(c['CARLA_next_errors'])))
        print('P2_CARLA_CONTACT_OUTPUT='+str(out));print('P2_STILL_NOT_CONTINUOUS_PATH_PROOF')
        return 0
    except (Failure,KeyError,TypeError,ValueError,OSError,ImportError) as e:
        print('P2_CONTACT_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
