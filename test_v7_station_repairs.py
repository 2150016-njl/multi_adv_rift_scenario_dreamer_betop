#!/usr/bin/env python3
import math
import sys
import types
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parent
V7=ROOT/"v7"

def load_v7_modules():
    pkg=types.ModuleType("v7")
    pkg.__path__=[str(V7)]
    sys.modules["v7"]=pkg

    coord_spec=importlib.util.spec_from_file_location("v7.coordinates",V7/"coordinates.py")
    coord=importlib.util.module_from_spec(coord_spec)
    sys.modules["v7.coordinates"]=coord
    coord_spec.loader.exec_module(coord)

    dm_spec=importlib.util.spec_from_file_location("v7.directed_map",V7/"directed_map.py")
    dm=importlib.util.module_from_spec(dm_spec)
    sys.modules["v7.directed_map"]=dm
    dm_spec.loader.exec_module(dm)

    sa_spec=importlib.util.spec_from_file_location("v7.station_audit",V7/"station_audit.py")
    sa=importlib.util.module_from_spec(sa_spec)
    sys.modules["v7.station_audit"]=sa
    sa_spec.loader.exec_module(sa)
    return dm,sa

def lane(uid, road, lane_id, x0, x1):
    pts=[[float(x),0.,0.] for x in range(x0,x1+1)]
    ss=[float(x) for x in range(x0,x1+1)]
    return {
        "lane_uid":uid,"road_id":road,"section_id":0,"lane_id":lane_id,
        "is_junction":False,"centerline_world_xyz":pts,"centerline_local_xyz":pts,
        "width_m":[4.]*len(pts),"directed_progress_m":ss,
        "opendrive_s_samples":ss,"station_travel_direction":1,
    }

def main():
    dm,sa=load_v7_modules()

    p=dm.project([[0,0,0],[1,0,0],[1,0,0],[2,0,0]],[1.5,.2,0])
    assert p is not None and abs(p["progress_m"]-1.5)<1e-9
    assert p["segment_index"]==2

    A=lane("1:0:1",1,1,0,10)
    B=lane("2:0:1",2,1,0,10)
    graph={"lanes":[A,B],"edges":[
        {"source":"1:0:1","destination":"2:0:1","kind":"lane_change_right",
         "source_s_m":5.,"destination_s_m":5.}
    ]}
    ego={"x":0.,"y":0.,"z":0.,"yaw_deg":0.}
    condition={"condition":{"ego_world_pose":ego,"directed_map":graph}}

    preview={"x":2.,"y":0.,"z":0.,"yaw_deg":0.,"road_id":1,"section_id":0,
             "lane_id":1,"s":2.}
    trace=[
        {"x":5.,"y":0.,"z":0.,"yaw_deg":0.,"road_id":1,"section_id":0,"lane_id":1,"s":5.},
        {"x":5.,"y":0.,"z":0.,"yaw_deg":0.,"road_id":2,"section_id":0,"lane_id":1,"s":5.},
    ]
    interaction={"x":8.,"y":0.,"z":0.,"yaw_deg":0.,"road_id":2,"section_id":0,
                 "lane_id":1,"s":8.}
    actor={"pose":{"actor_id":9,"x":0.,"y":0.,"z":0.,"yaw_deg":0.},
           "road_geometry":{"preview_waypoint":preview},
           "native":{"interaction_waypoint":interaction,
                     "target_route_to_interaction":trace}}

    t=sa.trace_station_audit(condition,actor,preview,interaction,trace)
    assert t["status"]=="supported",t

    a=sa.actor_station_audit(condition,actor,anchor_match=["anchor"])
    assert a["status"]=="supported",a
    assert abs(a["start_progress_m"])<1e-9
    assert a["lateral_membership"] is True
    assert a["heading_membership"] is True
    assert a["trace_audit"]["status"]=="supported"

    actor_bad=dict(actor)
    actor_bad["pose"]=dict(actor["pose"],y=3.0)
    b=sa.actor_station_audit(condition,actor_bad,anchor_match=["anchor"])
    assert b["status"]=="unknown" and b["reason"]=="initial_lateral_membership_failed",b

    print("PASS station projection / zero-length / typed trace / native lateral membership")

if __name__=="__main__":
    main()
