#!/usr/bin/env python3
"""Read-only XODR/exact diagnostic on same episode AFTER frozen P1 route failure.
This does not change native audit status and cannot turn any FAIL into PASS.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path

def waypoint_view(w,ref):
    if w is None:return None
    p=w.transform.location;raw=ref
    xyz=((p.x-float(raw['x']))**2+(p.y-float(raw['y']))**2+(p.z-float(raw['z']))**2)**0.5
    yaw=abs((float(w.transform.rotation.yaw)-float(raw['yaw_deg'])+180)%360-180)
    return {'road_id':int(w.road_id),'section_id':int(w.section_id),'lane_id':int(w.lane_id),
            's':float(w.s),'xyz_error_m':xyz,'s_error_m':abs(float(w.s)-float(raw['s'])),
            'yaw_error_deg':yaw}

def run(args):
    # The witness ONLY inspects the existing world. A map change fails closed.
    scene=json.loads(Path(args.snapshot).read_text(encoding='utf-8'))
    r=scene['ego_route'][args.route_index]
    expected=json.loads(Path(args.map_manifest).read_text(encoding='utf-8'))
    hashes={x['town']:x['opendrive_sha256'] for x in expected['maps']}
    town=scene['town'];wanted=hashes[town]
    transition=json.loads(Path(args.transition).read_text(encoding='utf-8'))
    assert transition['status']=='verified' and transition['verified_town']==town
    assert transition['verified_sha256']==wanted and transition['record_id']==args.record_index
    import carla
    c=carla.Client(args.host,args.port);c.set_timeout(40)
    world=c.get_world();m=world.get_map()
    sha=hashlib.sha256(m.to_opendrive().encode()).hexdigest()
    assert m.name.split('/')[-1]==town and sha==wanted and int(world.id)==int(transition['verified_episode_id']), 'world changed since route failure'
    loc=carla.Location(float(r['x']),float(r['y']),float(r['z']))
    xw=m.get_waypoint_xodr(int(r['road_id']),int(r['lane_id']),float(r['s']))
    ew=m.get_waypoint(loc,False,carla.LaneType.Driving)
    x=waypoint_view(xw,r);e=waypoint_view(ew,r)
    source=(int(r['road_id']),int(r['section_id']),int(r['lane_id']))
    x_ok=bool(x and (x['road_id'],x['section_id'],x['lane_id'])==source and
              x['s_error_m']<=1e-4 and x['xyz_error_m']<=0.05 and x['yaw_error_deg']<=0.5)
    exact_equal=bool(e and (e['road_id'],e['section_id'],e['lane_id'])==source)
    verdict=('xodr_source_verified_exact_lookup_disagreement' if x_ok and not exact_equal else
             'xodr_source_verified_exact_agrees' if x_ok and exact_equal else 'xodr_source_unverified')
    again=c.get_world()
    assert int(again.id)==int(world.id) and again.get_map().name==m.name,'world changed during witness'
    report={'schema':'p1_route_failure_witness_v1','record_index':args.record_index,'town':town,
            'map_sha256_verified':sha,'episode_id':int(world.id),'route_index':args.route_index,
            'source':{'road_id':source[0],'section_id':source[1],'lane_id':source[2], 's':float(r['s']),
                      'x':float(r['x']),'y':float(r['y']),'z':float(r['z']),'yaw_deg':float(r['yaw_deg']),
                      'is_junction':r.get('is_junction')},'xodr_at_original_s':x,
            'strict_xyz_driving':e,'xodr_source_strict_pass':x_ok,
            'strict_xyz_same_lane':exact_equal,'classification':verdict,
            'original_frozen_audit_status':'FAIL_UNCHANGED',
            'may_be_used_to_relax_frozen_route_gate':False}
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    print('P1_ROUTE_WITNESS='+json.dumps({'idx':args.record_index,'point':args.route_index,'classification':verdict,'xodr':x,'exact':e},ensure_ascii=False),flush=True)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for v in ('snapshot','map-manifest','transition','output'):
        p.add_argument('--'+v,required=True)
    p.add_argument('--record-index',type=int,required=True);p.add_argument('--route-index',type=int,required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=31120)
    a=p.parse_args()
    try:return run(a)
    except Exception as e:
        print('P1_WITNESS_FAIL_CLOSED='+type(e).__name__+': '+str(e),file=sys.stderr)
        return 2
if __name__=='__main__':sys.exit(main())
