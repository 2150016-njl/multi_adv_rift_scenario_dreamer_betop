#!/usr/bin/env python3
"""Offline contract tests on explicit lane-link verifier and repaired P2 sampling."""
from __future__ import annotations
import json,importlib.util,math,subprocess,sys,tempfile
from pathlib import Path
from p1_od_link_guard import verify_pair
from p2_full_lane_census_v2 import collect

class W:
 def __init__(self,road,section,lane,s,x=0,y=0,z=0,yaw=0,junc=False,permission='None'):
  self.road_id=road;self.section_id=section;self.lane_id=lane;self.s=s;self.lane_type='Driving';self.is_junction=junc;self.lane_change=permission
  self.transform=type('T',(),{'location':type('P',(),{'x':x,'y':y,'z':z})(),
                             'rotation':type('R',(),{'yaw':yaw})()})()
 def get_left_lane(self):return None
 def get_right_lane(self):return None
class C:
 class LaneType:Driving='Driving'
 class LaneChange:Left='Left';Right='Right';Both='Both'
class M:
 def __init__(self,rows):self.rows=rows
 def generate_waypoints(self,step):return self.rows
 def get_topology(self):return [] # old sampler would have sampled ZERO! new sample succeeds

X='''<OpenDRIVE><road id="10" length="20" junction="-1"><link><successor elementType="junction" elementId="100"/></link><lanes><laneSection s="0"><right><lane id="-1" type="driving"/></right></laneSection></lanes></road><road id="20" length="30" junction="100"><lanes><laneSection s="0"><right><lane id="-2" type="driving"/></right></laneSection></lanes></road><junction id="100"><connection id="1" incomingRoad="10" connectingRoad="20" contactPoint="start"><laneLink from="-1" to="-2"/></connection></junction></OpenDRIVE>'''

def test():
 a,b=W(10,0,-1,19,0,0),W(20,0,-2,1,0,1)
 assert verify_pair(X,a,b)['status']=='verified'
 assert verify_pair(X,a,W(20,0,-3,1))['status']=='unproven'
 assert verify_pair(X,W(10,0,-1,3),b)['status']=='unproven'
 assert verify_pair(X.replace('to="-2"','to="-3"'),a,b)['status']=='unproven'
 # Same geometry class as Town13 +lane to -lane across a junction with a
 # genuine intermediate laneSection: blindly requiring section 0 is wrong.
 x2='''<OpenDRIVE><road id="19894" length="40"><link><predecessor elementType="junction" elementId="200"/></link><lanes><laneSection s="0"><left><lane id="3" type="driving"/></left></laneSection><laneSection s="1"><left><lane id="2" type="driving"><link><predecessor id="3"/></link></lane></left></laneSection></lanes></road><road id="19897" length="9"><lanes><laneSection s="0"><right><lane id="-4" type="driving"/></right></laneSection></lanes></road><junction id="200"><connection id="7" incomingRoad="19894" connectingRoad="19897" contactPoint="start"><laneLink from="3" to="-4"/></connection></junction></OpenDRIVE>'''
 assert verify_pair(x2,W(19894,1,2,1.2),W(19897,0,-4,0.5))['status']=='verified'
 assert verify_pair(x2.replace('predecessor id="3"','predecessor id="4"'),
                    W(19894,1,2,1.2),W(19897,0,-4,0.5))['status']=='unproven'
 print('TEST_P1_LANESECTION_CHAIN_POSITIVE_TO_NEGATIVE=PASS')
 print('TEST_P1_STRICT_OPENDRIVE_JUNCTION_AND_NEGATIVES=PASS')
 waypoints=[W(140,0,-3,float(s),x=float(s),y=0) for s in range(0,30,2)]
 waypoints += [W(140,0,-2,float(s),x=float(s),y=3.5) for s in range(0,30,2)]
 cs=[{'record_index':100,'town':'Town15','ego_route':[{'road_id':140,'section_id':0,'lane_id':-3,
       'x':float(s),'y':0.0,'z':0.0,'yaw_deg':0.0} for s in range(0,30,2)],
      'gt_target_positions_in_condition':False}]
 lat,geo,stat=collect(M(waypoints),C,cs,'Town15',2.0)
 assert stat['driving_samples']==30 and stat['distinct_lane_keys']==2
 assert len(geo['sampled_lane_station_ranges'])==2
 assert geo['sampled_waypoint_count']==30
 assert geo['records'][0]['record_index']==100
 print('TEST_P2_GENERATE_WAYPOINTS_COVERS_MISSING_TOPOLOGY_SEGMENTS=PASS')
 try:
  bad=dict(cs[0],agents=[{'actor_id':1}])
  collect(M(waypoints),C,[bad],'Town15',2.0)
 except Exception:pass
 else:raise AssertionError('GT leakage was accepted')
 print('TEST_P2_GT_IN_CONDITIONS_REJECTED=PASS')

if __name__=='__main__':test()
