"""One online/offline observation path. Accepts observations, not native labels."""
from .coordinates import LOCAL_FRAME,WORLD_FRAME,state_to_local,world_to_local,velocity_to_local
from .directed_map import DirectedMapBuilder,associate_route
from .schema import validate_condition


def actor_token(ego,actor):
    velocity=[actor.get(k) for k in ('vx','vy','vz')]
    return {'actor_token':str(actor['actor_id']),'type_id':actor['type_id'],
            'state_xy_yaw_speed':state_to_local(ego,actor),
            'z_local_m':world_to_local(ego,[actor[k] for k in ('x','y','z')])[2],
            'size_lwh_m':[actor.get(k) for k in ('length','width','height')],
            'velocity_local_xyz_mps':velocity_to_local(ego,velocity) if None not in velocity else None}


def build_condition(record_id,observation,lanes,provenance,event=None):
    ego=observation['ego']
    graph=DirectedMapBuilder().build(lanes,ego,provenance.get('opendrive_sha256'))
    route_world=observation.get('ego_route')
    if route_world:
        points=world_to_local(ego,[[p[k] for k in ('x','y','z')] for p in route_world])
    else:
        points=observation['ego_route_local_xy']
    route=associate_route(graph,points,route_world)
    ego_velocity=[ego.get(k) for k in ('vx','vy','vz')]
    events={'mode':'historical_unknown','event_id':record_id+':historical',
            'trigger_reason':None,'eligible':None,'available_cbv_slots':None,
            'max_generated_slots':2,'ego_route_progress_m':None,
            'field_status':'not_serialized_in_historical_ledger'}
    if event:events.update(event)
    record={'schema_version':'hv_v7_condition_v1','record_id':record_id,
            'provenance':provenance,'event':events,
            'condition':{'coordinate_frame':LOCAL_FRAME,'world_frame':WORLD_FRAME,
                'transform_contract':'yaw-only SE(2), z translation; radians locally, degrees in world pose',
                'ego_world_pose':{k:ego[k] for k in ('x','y','z','yaw_deg')},
                'ego':{'speed_mps':ego.get('speed_mps'),'size_lwh_m':[ego.get(k) for k in ('length','width','height')],
                       'velocity_local_xyz_mps':velocity_to_local(ego,ego_velocity) if None not in ego_velocity else None},
                'output_domain':{'slot0':{'x_min_exclusive':0.,'x_max_inclusive':32.,'y_bounds_m':[-32.,32.]},
                                 'slot1':{'x_bounds_m':[-32.,32.],'y_bounds_m':[-32.,32.]},
                                 'map_query_extent_independent':True},
                'ego_route':route,'directed_map':graph,
                'observed_actor_tokens':[actor_token(ego,p) for p in observation['actors']],
                'existing_cbv_ids':observation.get('existing_cbv_ids'),
                'observation_view':observation['observation_view'],
                'fixed_target_blueprint':'vehicle.tesla.model3',
                'fixed_blueprint_calibration_status':'unverified'}}
    validate_condition(record)
    return record
