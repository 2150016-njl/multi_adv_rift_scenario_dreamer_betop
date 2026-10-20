#!/usr/bin/env python3
"""Offline unit tests for V2 station screen & P1 XML diagnostic contracts."""
import ast,importlib.util,io,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('p2',str(R/'p2_station_order_v2.py'))
p2=importlib.util.module_from_spec(spec);spec.loader.exec_module(p2)
spec=importlib.util.spec_from_file_location('p1',str(R/'p1_od_link_guard.py'))
p1=importlib.util.module_from_spec(spec);spec.loader.exec_module(p1)

class Test(unittest.TestCase):
    def test_python38_parse(self):
        for fn in R.glob('*.py'):
            ast.parse(fn.read_text(),filename=fn.name,feature_version=(3,8))
    def test_same_lane_direction(self):
        ranges={(10,0,-1):(0,100)}
        path={'lane':[10,0,-1],'anchor_lane':[10,0,-1],'lane_path_to_anchor':[[10,0,-1]],'path_edge_types':[]}
        anchor={'ego_s':40,'anchor_lane':[10,0,-1]}
        r=p2.verify_path(path,anchor,ranges,set(),{})
        self.assertTrue(r['station_screen_possible'])
        self.assertEqual(r['origin_station_feasible_observed'],[0,40])
        ranges={(10,0,1):(0,100)}
        path={key:([-10,0,1] if False else [10,0,1]) if key in ('lane','anchor_lane') else [[10,0,1]] if key=='lane_path_to_anchor' else [] for key in path}
        r=p2.verify_path(path,anchor,ranges,set(),{})
        self.assertEqual(r['origin_station_feasible_observed'],[40,100])
    def test_lateral_before_anchor(self):
        src=(140,0,-2);dst=(140,0,-3)
        ranges={src:(0,350),dst:(0,350)}
        la={(src,dst):[{'from_lane':list(src),'to_lane':list(dst),'source_s_min':20,'source_s_max':80,
                        'source_marking_permission_checked':True,'geometry_checks_passed_at_samples':True}]}
        path={'lane':list(src),'anchor_lane':list(dst),'lane_path_to_anchor':[list(src),list(dst)],'path_edge_types':['lateral_marking_samples']}
        a={'ego_s':60}
        r=p2.verify_path(path,a,ranges,set(),la)
        self.assertTrue(r['station_screen_possible']); self.assertEqual(r['origin_station_feasible_observed'],[0,60])
        a={'ego_s':10}
        r=p2.verify_path(path,a,ranges,set(),la)
        self.assertFalse(r['station_screen_possible']);self.assertEqual(r['status'],'station_order_conflict')
    def test_disallow_missing_lateral(self):
        a=(140,0,-2);b=(140,0,-3)
        p={'lane':list(a),'anchor_lane':list(b),'lane_path_to_anchor':[list(a),list(b)],'path_edge_types':['lateral_marking_samples']}
        r=p2.verify_path(p,{'ego_s':40},{a:(0,350),b:(0,350)},set(),{})
        self.assertFalse(r['station_screen_possible'])
    def test_gt_free_rejection(self):
        top={'town':'T','map_sha256':'a','directed_edges':[]}
        cond={'town':'T','record_index':1,'gt_target_positions_in_condition':False,'ego_route':[],'targets':[1]}
        geo={'town':'T','map_sha256':'a','GT_used':False,'sampled_lane_station_ranges':[]}
        lat={'town':'T','opendrive_sha256':'a','GT_used':False,'intervals':[]}
        appr={'town':'T','GT_used_in_make':False,'records':[{'record_index':1,'approaches':[]}]}
        with self.assertRaisesRegex(p2.ContractError,'GT_IN_P2_MAKE_CONDITION'):
            p2.make(top,[cond],geo,lat,appr)
    def test_xml_guard_real_class_fixture(self):
        road=lambda n,l,sign: f'<road id="{n}" length="{l}" junction="-1"><link><{"successor" if sign<0 else "predecessor"} elementType="junction" elementId="8"/></link><lanes><laneSection s="0"><{"right" if sign<0 else "left"}><lane id="{sign}" type="driving"/></{"right" if sign<0 else "left"}></laneSection></lanes></road>'
        xml='<OpenDRIVE>'+road(1,10,-1)+road(2,10,-1)+'<junction id="8"><connection id="0" incomingRoad="1" connectingRoad="2" contactPoint="start"><laneLink from="-1" to="-1"/></connection></junction></OpenDRIVE>'
        x=NS(road_id=1,section_id=0,lane_id=-1,s=9.0);y=NS(road_id=2,section_id=0,lane_id=-1,s=1.0)
        self.assertEqual(p1.verify_pair(xml,x,y)['status'],'verified')
        self.assertEqual(p1.verify_pair(xml,x,NS(road_id=2,section_id=0,lane_id=-1,s=8.0))['status'],'unproven')
    def test_no_gt_access_in_make(self):
        s=(R/'p2_station_order_v2.py').read_text()
        segment=s[s.index('def make('):s.index('def evaluate(')]
        self.assertNotIn('gt_targets',segment)
    def test_contact_sample_never_claims_complete_path(self):
        src=(10,0,-1);dst=(20,0,-1)
        path={'lane':list(src),'anchor_lane':list(dst),
              'lane_path_to_anchor':[list(src),list(dst)],'path_edge_types':['longitudinal']}
        samples={(src,dst):[{'source':{'s':90},'target':{'s':0}}]}
        r=p2.verify_path(path,{'ego_s':50},{src:(0,100),dst:(0,100)},
                         {(src,dst)},{},samples)
        self.assertTrue(r['station_screen_possible'])
        self.assertEqual(r['origin_station_feasible_observed'],[0,90])
        self.assertFalse(r['edges'][0]['complete_longitudinal_connectivity_proven'])
    def test_contact_source_sha_guard(self):
        top={'town':'T','map_sha256':'a','directed_edges':[]}
        geo={'town':'T','map_sha256':'a','GT_used':False,'sampled_lane_station_ranges':[]}
        lat={'town':'T','opendrive_sha256':'a','GT_used':False,'intervals':[]}
        app={'town':'T','GT_used_in_make':False,'records':[]}
        bad={'schema':'p2_CARLA_local_directed_contact_samples_v1',
             'town':'X','map_sha256':'a','GT_used':False,'samples':[]}
        with self.assertRaisesRegex(p2.ContractError,'CARLA_CONTACT_SOURCE_OR_MAP_NOT_VERIFIED'):
            p2.make(top,[],geo,lat,app,bad)

if __name__=='__main__':unittest.main(verbosity=2)
