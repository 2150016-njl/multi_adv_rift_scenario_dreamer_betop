#!/usr/bin/env python3
"""Independent real-CARLA P1 read-only forensic for 1039 / 1279.

Checks original snapshot/map SHA, episode and exact get_waypoint on route pair;
reuses original 95-route census road-end classification and prints explicit
OpenDRIVE road/junction/lane links independently of station distance gate.
Neither 'verified_road_end_exception' nor explicit XML link means full P1 PASS.
Requires exclusive CARLA server if --ack-dedicated is used to switch towns.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys, xml.etree.ElementTree as ET
from pathlib import Path
from p1_od_link_guard import verify_pair

TARGETS={'Town12':(1039,14),'Town13':(1279,19)}
MANIFEST_SHA='92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
MAP_MANIFEST_SHA='9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'

class ContractError(RuntimeError):pass
def need(ok,msg):
    if not ok:raise ContractError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lane(obj):return [int(obj[k]) for k in ('road_id','section_id','lane_id')]
def geom_error(w,p):
    if w is None:return None
    loc=w.transform.location
    xyz=math.sqrt(sum((float(getattr(loc,k))-float(p[k]))**2 for k in ('x','y','z')))
    return {'identity':[int(w.road_id),int(w.section_id),int(w.lane_id)],
            's':float(w.s),'s_delta_m':float(w.s)-float(p['s']),
            'xyz_error_m':xyz,'yaw_error_deg':abs((float(w.transform.rotation.yaw)-float(p['yaw_deg'])+180)%360-180)}
def xml_details(root,a,b):
    roads={int(r.get('id')):r for r in root.findall('road')}
    junctions={int(r.get('id')):r for r in root.findall('junction')}
    r=roads.get(int(a['road_id']));q=roads.get(int(b['road_id']))
    need(r is not None and q is not None,'road missing from OpenDRIVE')
    def road_dump(road,point):
        link_direction='successor' if int(point['lane_id'])<0 else 'predecessor'
        linked=road.find('./link/'+link_direction)
        sects=road.findall('./lanes/laneSection')
        sid=int(point['section_id'])
        lnode=None
        if sid < len(sects) and sid>=0:
            side='left' if int(point['lane_id'])>0 else 'right'
            for node in sects[sid].findall('./'+side+'/lane'):
                if int(node.get('id','0'))==int(point['lane_id']):lnode=node;break
        def xmlattrs(node):return node.attrib if node is not None else None
        return {'road_id':int(road.get('id')),'road_length':float(road.get('length')),
                'snapshot_s':float(point['s']),'distance_to_s_0':float(point['s']),
                'distance_to_s_length':float(road.get('length'))-float(point['s']),
                'travel_link_direction':link_direction,'road_link':xmlattrs(linked),
                'source_section_count':len(sects),'source_section_lane':xmlattrs(lnode),
                'source_section_lane_links':[{**n.attrib,'relation':n.tag} for n in lnode.findall('./link/*')] if lnode is not None else []}
    d={'source':road_dump(r,a),'target':road_dump(q,b)}
    source_id=int(a['road_id']);target_id=int(b['road_id'])
    direct=[];connected=[]
    for j in junctions.values():
        for c in j.findall('connection'):
            incoming=int(c.get('incomingRoad','-1'));con=int(c.get('connectingRoad','-1'))
            if incoming==source_id or con==target_id:
                o={'junction_id':int(j.get('id')),'connection_id':c.get('id'),
                   'incomingRoad':incoming,'connectingRoad':con,'contactPoint':c.get('contactPoint'),
                   'laneLinks':[dict(l.attrib) for l in c.findall('laneLink')]}
                if incoming==source_id and con==target_id:direct.append(o)
                else:connected.append(o)
    # Evidence only; sorted for deterministic audit. The non-direct list is capped
    # to avoid bloating logs while full XML SHA is always available in the record.
    d['direct_junction_connections']=direct
    d['nearby_junction_connections_count']=len(connected)
    d['nearby_junction_connections_sample']=connected[:50]
    return d

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True);ap.add_argument('--town',choices=tuple(TARGETS),required=True)
    ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=31120)
    ap.add_argument('--output',required=True);ap.add_argument('--ack-dedicated',action='store_true')
    args=ap.parse_args()
    try:
        root=Path(args.root).resolve();base=root/'data/hv_init/v7';out=Path(args.output)
        need(not out.exists(),'refuse overwrite evidence')
        mf=base/'manifests/p1_calibration95.jsonl'
        mm=base/'manifests/p1_map_identity_manifest_v2_20261017_154737.json'
        need(sha(mf)==MANIFEST_SHA and sha(mm)==MAP_MANIFEST_SHA,'immutable manifest SHA mismatch')
        mp={x['town']:x['opendrive_sha256'] for x in read(mm)['maps']}
        records={int(x['record_index']):x for x in (json.loads(s) for s in mf.read_text().splitlines() if s.strip())}
        idx,routeidx=TARGETS[args.town]; manifest=records[idx];snap=Path(manifest['snapshot_path'])
        need(sha(snap)==manifest['snapshot_sha256'],'snapshot SHA mismatch')
        scene=read(snap);need(scene['town']==args.town,'snapshot Town mismatch')
        originals=scene['ego_route'][routeidx-1:routeidx+1]
        need(len(originals)==2,'route index invalid')
        egg=root/'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg'
        need(egg.is_file(),'pinned CARLA egg missing')
        sys.path.insert(0,str(egg));import carla
        need(str(carla.__file__).startswith(str(egg)+'/'),'CARLA egg imported from wrong location')
        client=carla.Client(args.host,args.port);client.set_timeout(120)
        need(client.get_server_version()==client.get_client_version()=='0.9.15','CARLA wrong version')
        world=client.get_world();cmap=world.get_map();transition={'no_load':True}
        if cmap.name.split('/')[-1]!=args.town:
            need(args.ack_dedicated,'wrong Town; explicit --ack-dedicated required')
            actors=world.get_actors()
            need(len(actors.filter('vehicle.*'))==0 and len(actors.filter('walker.pedestrian.*'))==0,
                 'refuse to switch occupied CARLA world')
            guardfile=root/'p1_science_v2_release/hv_v8/p1_world_guard.py'
            need(guardfile.is_file(),'missing pinned world guard')
            spec=importlib.util.spec_from_file_location('p1_temporary_worldguard',str(guardfile))
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            world,cmap,_,transition=mod.load_verified_world(client,args.town,mp[args.town])
        xodr=cmap.to_opendrive()
        actualsha=hashlib.sha256(xodr.encode()).hexdigest()
        need(cmap.name.split('/')[-1]==args.town and actualsha==mp[args.town],
             'Town / OpenDRIVE SHA wrong')
        root_xml=ET.fromstring(xodr)
        roadlen={int(r.get('id')):float(r.get('length')) for r in root_xml.findall('road')}
        # EXACT historical route-census status, not a newly relaxed tolerance.
        legacy=root/'hv_v8/p1_route_geometry95_forensic_v1.py'
        if not legacy.is_file():legacy=Path(__file__).with_name('p1_route_geometry95_forensic_v1.py')
        need(legacy.is_file(),'missing frozen route census helper')
        spec=importlib.util.spec_from_file_location('p1_route_census_v1_temp',str(legacy))
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        checks=[old.route_probe(carla,cmap,p,roadlen) for p in originals]
        link=xml_details(root_xml,originals[0],originals[1])
        # New source proof only if old independent source-anchor or strict
        # previously defined road-end exception succeeded for EACH endpoint.
        accepted={'verified_xodr_and_exact','verified_xodr_road_end_exception',
                  'xodr_source_verified_exact_other_lane','xodr_source_verified_exact_none'}
        source_anchor_statuses=[v['status'] for v in checks]
        roadend_validated=all(v in accepted for v in source_anchor_statuses)
        xodr_wp=[]
        for p in originals:
            xodr_wp.append(cmap.get_waypoint_xodr(int(p['road_id']),int(p['lane_id']),float(p['s'])))
        od=verify_pair(xodr,xodr_wp[0],xodr_wp[1]) if all(xodr_wp) else {'status':'unproven','reason':'xodr_missing_use_strict_road_end_probe'}
        after=client.get_world();am=after.get_map()
        need(after.id==world.id and am.name==cmap.name and hashlib.sha256(am.to_opendrive().encode()).hexdigest()==actualsha,
             'episode/map changed during probe')
        result={'schema':'p1_endpoint_xml_link_diagnostic_v2','town':args.town,'record_index':idx,
                'route_pair_indexes':[routeidx-1,routeidx],'episode_id':world.id,
                'map_sha256':actualsha,'snapshot_sha256':manifest['snapshot_sha256'],
                'old_census_source_statuses':source_anchor_statuses,
                'each_anchor_meets_old_geometry_classification':roadend_validated,
                'original_route_pair':originals,'station_and_exact_diagnostics':checks,
                'xml_link_evidence_without_short_distance_filter':link,
                'old_guard_link_result':od,'transition':transition,'scientific_P1_pass':False,
                'not_a_route_continuity_proof':True}
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
        print('P1_REAL_ENDPOINT_CASE='+str(idx))
        print('P1_OLD_GEOMETRY_STATUSES='+json.dumps(source_anchor_statuses))
        print('P1_ORIGINAL_ANCHORS_VERIFIED='+str(roadend_validated))
        print('P1_EXPLICIT_XML_JUNCTION_CANDIDATES='+str(len(link['direct_junction_connections'])))
        print('P1_LINK_PROOF_STATUS='+str(od['status']))
        print('P1_EVIDENCE='+str(out))
        print('P1_STILL_NOT_ACCEPTANCE_PASS')
        return 0
    except (ContractError,KeyError,TypeError,ValueError,ET.ParseError,OSError,ImportError) as exc:
        print('P1_DIAGNOSTIC_FAIL_CLOSED='+type(exc).__name__+': '+str(exc),file=sys.stderr)
        return 2
if __name__=='__main__':sys.exit(main())
