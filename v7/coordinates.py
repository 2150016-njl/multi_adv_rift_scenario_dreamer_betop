"""Planar CARLA frame transform: world LH -> ego x-forward/y-right, z-up."""
import math
import numpy as np

WORLD_FRAME = 'carla_world_left_handed'
LOCAL_FRAME = 'ego_local_carla_x_forward_y_right'


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation(ego):
    a=math.radians(ego['yaw_deg']);c,s=math.cos(a),math.sin(a)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def world_to_local(ego,xyz):
    return ((np.asarray(xyz,dtype=float)-[ego[k] for k in ('x','y','z')])@rotation(ego)).tolist()


def local_to_world(ego,xyz):
    return (np.asarray(xyz,dtype=float)@rotation(ego).T+[ego[k] for k in ('x','y','z')]).tolist()


def velocity_to_local(ego,xyz):
    return (np.asarray(xyz,dtype=float)@rotation(ego)).tolist()


def velocity_to_world(ego,xyz):
    return (np.asarray(xyz,dtype=float)@rotation(ego).T).tolist()


def state_to_local(ego,actor):
    p=world_to_local(ego,[actor[k] for k in ('x','y','z')])
    return [p[0],p[1],wrap(math.radians(actor['yaw_deg']-ego['yaw_deg'])),actor.get('speed_mps')]


def in_domain(xy,front=False):
    x,y=xy[:2]
    return (0<x<=32 if front else -32<=x<=32) and -32<=y<=32


def infer_frame_from_map(world_lanes,local_lanes,tolerance=1.e-5):
    """Explicit ledger-only fallback; fit from map pairs, NEVER target poses."""
    local={r['lane_uid']:r for r in local_lanes};pairs=[]
    for lane in world_lanes:
        other=local.get(lane['lane_uid'])
        if other and len(other['centerline'])==len(lane['centerline']):
            pairs.extend(zip(lane['centerline'],other['centerline']))
    if len(pairs)<2:raise ValueError('Insufficient world/local map pairs')
    w=np.array([p[0] for p in pairs]);q=np.array([p[1] for p in pairs])
    i=int(np.argmax(np.linalg.norm(q[:,:2]-q[0,:2],axis=1)))
    if np.linalg.norm(q[i,:2]-q[0,:2])<1:raise ValueError('Degenerate map pairs')
    dw=w[i]-w[0];dq=q[i]-q[0]
    yaw=math.atan2(dw[1],dw[0])-math.atan2(dq[1],dq[0])
    ego={'x':0.,'y':0.,'z':0.,'yaw_deg':math.degrees(wrap(yaw))}
    origin=w[0]-np.asarray(local_to_world(ego,q[0]))
    ego.update(zip(('x','y','z'),origin.tolist()))
    residual=float(np.linalg.norm(np.asarray(world_to_local(ego,w))-q,axis=1).max())
    if residual>tolerance:raise ValueError('Map frame fit residual %.9gm'%residual)
    return ego,residual


def infer_frame_from_observed_actor(actor,token):
    """Fallback using one visible non-withheld observation; no target state used."""
    local=token['state'];yaw=math.radians(actor['yaw_deg'])-local[2]
    ego={'x':0.,'y':0.,'z':0.,'yaw_deg':math.degrees(wrap(yaw))}
    offset=local_to_world(ego,[local[0],local[1],token['z_m']])
    ego.update({k:actor[k]-offset[i] for i,k in enumerate(('x','y','z'))})
    return ego
