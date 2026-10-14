"""Offline adapters for the unchanged RIFT native candidate algorithm.

CARLA imports are deferred. No live CarlaDataProvider object/cache is patched.
The native function's bytecode is reused with an isolated provider binding.
This certifies candidate eligibility, not attention selection or CBV activation.
"""
import copy
import hashlib
import math
import struct
import types


def isolated_method(function, instance, provider):
    scope = dict(function.__globals__)
    scope['CarlaDataProvider'] = provider
    cloned = types.FunctionType(function.__code__, scope, function.__name__,
                                function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    return types.MethodType(cloned, instance)


def waypoint_dict(wp):
    if wp is None:
        return None
    t = wp.transform
    return {'x':float(t.location.x), 'y':float(t.location.y), 'z':float(t.location.z),
            'yaw_deg':float(t.rotation.yaw), 'road_id':int(wp.road_id),
            'lane_id':int(wp.lane_id), 'section_id':int(wp.section_id),
            's':float(wp.s), 'is_junction':bool(wp.is_junction),
            'lane_width_m':float(wp.lane_width)}


def transform_from_pose(carla, pose):
    return carla.Transform(
        carla.Location(x=float(pose['x']), y=float(pose['y']), z=float(pose['z'])),
        carla.Rotation(yaw=float(pose['yaw_deg']), pitch=float(pose.get('pitch_deg',0.)),
                       roll=float(pose.get('roll_deg',0.))))


class SavedActor:
    def __init__(self, carla, pose):
        self.id = int(pose['actor_id'])
        self.type_id = str(pose.get('type_id','vehicle.unknown'))
        self.transform = transform_from_pose(carla, pose)


class SnapshotProvider:
    def __init__(self, actors, reached_ids=()):
        self.actors = {a.id:a for a in actors}
        self.reached = dict.fromkeys(reached_ids)

    def get_actors(self):
        return self.actors

    def get_location(self, actor):
        return actor.transform.location

    def get_transform(self, actor):
        return actor.transform

    def get_CBVs_reach_goal_by_ego(self, ego_id):
        return self.reached


def native_actor(policy_class, cache_factory, carla, world_map, grp, config,
                 mode, ego_pose, pose, route, existing_ids=(), reached_ids=()):
    """Run original get_CBV_candidates, including its envelope checks, for one actor."""
    policy = policy_class(dict(config, desired_speed=0.), logger=None)
    policy.set_mode(mode)
    policy._map = world_map
    policy.global_route_planner = copy.copy(grp)
    # Only these fields mutate during trace_route in this repository. The
    # graph/topology itself is read-only and can be shared between adapters.
    if hasattr(grp, '_previous_decision'):
        policy.global_route_planner._previous_decision = type(grp._previous_decision).VOID
    policy.global_route_planner._intersection_end_node = -1
    cache = cache_factory({})
    policy.set_route_planner(cache)
    ego, actor = SavedActor(carla, ego_pose), SavedActor(carla, pose)
    provider = SnapshotProvider([ego, actor], reached_ids)
    bound = isolated_method(policy_class.get_CBV_candidates, policy, provider)
    candidates = bound(ego, set(existing_ids), route)
    info = cache.inter_info.get(actor.id, {})
    meta = info.get('interaction_meta') or {}
    threshold = float(config[mode]['min_interaction_dis_diff'])
    positive = any(a.id == actor.id for a in candidates)
    trace = info.get('inter_CBV_route_elements') or []
    return {
        'evaluated':True, 'native_candidate':positive,
        'score':max(0., 1.-float(meta['distance_diff'])/threshold) if positive else 0.,
        'interaction_waypoint':waypoint_dict(info.get('inter_waypoint')),
        'interaction_meta':meta,
        'target_route_to_interaction':[waypoint_dict(item[0]) for item in trace],
        'candidate_debug':policy.last_candidate_debug,
    }


class NativeAudit:
    def __init__(self, client, config, mode='eval', resolution=1., load_maps=False):
        from .carla_audit_environment import configure_carla_pythonapi
        configure_carla_pythonapi(validate=True)
        import carla
        from rift.cbv.recognition.base_cbv import BaseCBVRecog
        from rift.cbv.planning.route_planner.base_planner import CBVBasePlanner
        from rift.scenario.tools.global_route_planner import GlobalRoutePlanner
        self.carla, self.client = carla, client
        self.policy_class, self.cache_factory = BaseCBVRecog, CBVBasePlanner
        self.grp_class = GlobalRoutePlanner
        self.config, self.mode = config, mode
        self.resolution, self.load_maps = float(resolution), bool(load_maps)
        self.world = self.world_map = self.grp = None

    def prepare(self, town):
        world = self.client.get_world()
        name = world.get_map().name.rsplit('/',1)[-1]
        if name != town:
            if not self.load_maps:
                raise ValueError('Map mismatch %s != %s; use a dedicated server and --load-maps' % (name,town))
            world = self.client.load_world(town)
        world_map = world.get_map()
        identity = (world_map.name, getattr(world,'id',None))
        if self.world is None or getattr(self,'map_identity',None) != identity:
            self.grp = self.grp_class(world_map, self.resolution)
            self.map_identity = identity
            self.map_sha256 = hashlib.sha256(world_map.to_opendrive().encode('utf-8')).hexdigest()
            self.map_geometry=None
            self.static_geometry=None
        self.world, self.world_map = world, world_map

    def ensure_geometry(self):
        from .audit_physics_geometry import MapGeometry, static_boxes
        if self.map_geometry is None:
            self.map_geometry=MapGeometry(self.world_map,self.resolution)
        if self.static_geometry is None:
            self.static_geometry=static_boxes(self.world,self.carla)

    def enriched_lanes(self,scene):
        self.ensure_geometry()
        return self.map_geometry.local_lanes(scene['ego'])

    def resolve_route(self, points, tolerance=.25):
        if len(points) < 2:
            raise ValueError('Serialized ego route has fewer than two points')
        route, errors = [], []
        for p in points:
            loc = self.carla.Location(x=float(p['x']), y=float(p['y']), z=float(p['z']))
            wp = None
            if all(k in p for k in ('road_id','lane_id','s')):
                wp = self.world_map.get_waypoint_xodr(int(p['road_id']),int(p['lane_id']),float(p['s']))
            if wp is None:
                wp = self.world_map.get_waypoint(loc, project_to_road=False,
                                                lane_type=self.carla.LaneType.Driving)
            if wp is None:
                raise ValueError('Cannot resolve ego route point without projection')
            error = loc.distance(wp.transform.location)
            if error > tolerance or any(int(getattr(wp,k)) != int(p[k])
                                       for k in ('road_id','lane_id','section_id') if k in p):
                raise ValueError('Serialized route identity/position mismatch (%.4fm)' % error)
            route.append(wp)
            errors.append(float(error))
        return route, max(errors)

    def generated_pose(self, ego, local_state, actor_id, blueprint, spawn_request=None):
        x,y,yaw,speed = map(float,local_state[:4])
        angle = math.radians(float(ego['yaw_deg']))
        wx = float(ego['x'])+math.cos(angle)*x-math.sin(angle)*y
        wy = float(ego['y'])+math.sin(angle)*x+math.cos(angle)*y
        wh = float(ego['yaw_deg'])+math.degrees(yaw)
        if spawn_request and 'z' in spawn_request:
            f32 = lambda v:struct.unpack('f',struct.pack('f',float(v)))[0]
            if math.hypot(f32(wx)-float(spawn_request['x']),f32(wy)-float(spawn_request['y'])) > 2.e-6 or \
                    abs((f32(wh)-float(spawn_request['yaw_deg'])+180.)%360.-180.) > 1.e-5:
                raise ValueError('Saved spawn command does not match raw local state')
            z, source = float(spawn_request['z']), 'saved_spawn_request'
        else:
            loc = self.carla.Location(x=wx,y=wy,z=float(ego['z']))
            wp = self.world_map.get_waypoint(loc,project_to_road=True,lane_type=self.carla.LaneType.Driving)
            if wp is None:
                raise ValueError('No terrain waypoint for strict spawn z rule')
            z, source = float(wp.transform.location.z)+.3, 'strict_nearest_road_z_plus_0p3'
        return {'actor_id':actor_id,'type_id':blueprint,'x':wx,'y':wy,'z':z,
                'yaw_deg':wh,'roll_deg':0.,'pitch_deg':0.,'speed_mps':speed,
                'length':4.624,'width':1.909,'z_source':source}

    def geometry(self, pose):
        t = transform_from_pose(self.carla,pose)
        center = self.world_map.get_waypoint(t.location,project_to_road=False,
                                            lane_type=self.carla.LaneType.Driving)
        f = t.get_forward_vector().make_unit_vector()
        loc = self.carla.Location(x=t.location.x+2*f.x,y=t.location.y+2*f.y,z=t.location.z+2*f.z)
        # Match native call, and independently record the explicit bool API.
        preview = self.world_map.get_waypoint(loc,project_to_road=None,lane_type=self.carla.LaneType.Driving)
        explicit = self.world_map.get_waypoint(loc,project_to_road=False,lane_type=self.carla.LaneType.Driving)
        def identity(wp):
            return None if wp is None else (wp.road_id,wp.section_id,wp.lane_id)
        heading = None if preview is None else math.degrees(
            t.get_forward_vector().get_vector_angle(preview.transform.get_forward_vector()))
        return {'center_driving':center is not None,'center_waypoint':waypoint_dict(center),
                'preview_driving':preview is not None,'preview_waypoint':waypoint_dict(preview),
                'preview_none_false_agree':identity(preview)==identity(explicit),
                'preview_heading_error_deg':heading,
                'whole_obb_road_containment_checked':False}

    def actor(self, scene, pose, route):
        debug = ((scene.get('diagnostics') or {}).get('rift_labeler') or {}).get('rift_candidate_debug') or {}
        existing = debug.get('existing_cbv_ids', [])
        reached = debug.get('reach_goal_cbv_ids', [])
        result = native_actor(self.policy_class,self.cache_factory,self.carla,self.world_map,
                              self.grp,self.config,self.mode,scene['ego'],pose,route,existing,reached)
        result['capacity_context'] = 'saved_debug' if 'existing_cbv_ids' in debug else 'empty_slots_assumed'
        comparable = ('search_radius','max_interaction_dis','max_interaction_fov','min_interaction_dis_diff','max_agent_num')
        result['saved_config_mismatches'] = {k:{'saved':debug[k],'audit':self.config[self.mode][k]}
                                             for k in comparable if k in debug and debug[k] != self.config[self.mode][k]}
        return result


def destroy_owned_without_tick(client,carla,actors,progress=None):
    """Delete only our newly spawned IDs via RPC, bypassing Actor's frame cache."""
    ids=[actor.id for actor in reversed(actors)]
    evidence={'method':'batch_destroy_actor_id_no_tick','actor_ids':ids,'responses':[]}
    errors=[]
    if not ids:return errors,evidence
    for actor_id in ids:
        if progress:progress('owned_cleanup',spawned_actor_id=actor_id)
    try:
        responses=client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids],False)
        if len(responses)!=len(ids):errors.append('Owned cleanup response count mismatch')
        for actor_id,response in zip(ids,responses):
            error=str(response.error) if response.has_error() else None
            evidence['responses'].append({'actor_id':actor_id,'error':error})
            if error:errors.append('Owned cleanup %s: %s'%(actor_id,error))
    except Exception as exc:
        errors.append('Owned batch cleanup: %s: %s'%(type(exc).__name__,exc))
    return errors,evidence


def spawn_frozen_cohort(world, carla, ego, context, targets, engine=None, probe_ticks=0, scene=None):
    """One try per saved pose, no tick, snap, fallback, Traffic Manager or retry.

    Restore only serialized vehicles; this is a static acceptance experiment,
    not an exact replay of episode traffic lights, pedestrians or trajectories.
    Caller owns synchronous settings and exclusive empty-world checks.
    """
    created, attempts = [], []
    owned={};geometry_report={};dynamic=None;geometry_errors=[]
    progress=getattr(engine,'progress',None)
    try:
        for role, poses in [('ego',[ego]),('context',context),('target',targets)]:
            for p in poses:
                attempt = {'role':role,'source_actor_id':p['actor_id'],'requested_type_id':p['type_id'],
                           'requested_pose':dict(p),'accepted':False,'status':'pending'}
                attempts.append(attempt)
                try:
                    if progress:progress('raw_spawn',source_actor_id=p['actor_id'],role=role,type_id=p['type_id'])
                    bp = world.get_blueprint_library().find(p['type_id'])
                    if bp.id != p['type_id']:
                        raise ValueError('Blueprint fallback forbidden')
                    t = transform_from_pose(carla,p)
                    command = dict(x=float(t.location.x),y=float(t.location.y),z=float(t.location.z),
                                   yaw_deg=float(t.rotation.yaw),pitch_deg=float(t.rotation.pitch),
                                   roll_deg=float(t.rotation.roll))
                    f32 = lambda v:struct.unpack('f',struct.pack('f',float(v)))[0]
                    attempt['spawn_command'] = command
                    # Real CARLA constructors use float32. CPU stand-ins use
                    # Python floats; both must agree up to that rounding only.
                    attempt['raw_request_fidelity'] = all(
                        abs(command[k]-float(p.get(k,0.))) <=
                        abs(f32(p.get(k,0.))-float(p.get(k,0.)))+2.e-6
                        for k in ('x','y','z','yaw_deg','pitch_deg','roll_deg'))
                    if not attempt['raw_request_fidelity']:
                        raise ValueError('Spawn command differs from requested pose beyond float32 precision')
                    actor = world.try_spawn_actor(bp,t)
                    if actor is None:
                        attempt['status'] = 'spawn_rejected'
                        continue
                    created.append(actor)
                    owned[p['actor_id']]=(actor,t)
                    if actor.type_id != p['type_id']:
                        raise ValueError('Actual spawned blueprint differs')
                    # Synchronous world + zero ticks freezes poses without
                    # changing physics/collision behavior between spawn calls.
                    f = t.get_forward_vector()
                    speed = float(p.get('speed_mps',0.))
                    velocity = (dict(x=float(p['vx']),y=float(p['vy']),z=float(p['vz']))
                                if all(k in p for k in ('vx','vy','vz')) else
                                dict(x=speed*f.x,y=speed*f.y,z=speed*f.z))
                    actor.set_target_velocity(carla.Vector3D(**velocity))
                    box = actor.bounding_box
                    attempt.update({'accepted':True,'status':'accepted','spawned_actor_id':actor.id,
                                    'actual_type_id':actor.type_id,
                                    'requested_velocity':velocity,
                                    'actual_blueprint_box':{'length':2*box.extent.x,'width':2*box.extent.y,
                                                            'height':2*box.extent.z,
                                                            'offset_x':box.location.x,'offset_y':box.location.y,
                                                            'offset_z':box.location.z},
                                    'pose_velocity_observation':'pre_tick_unconfirmed'})
                except Exception as exc:
                    attempt['status'] = 'error'
                    attempt['error'] = '%s: %s' % (type(exc).__name__,exc)
        if engine is not None:
            from .audit_physics_geometry import cohort_geometry
            try:
                if progress:progress('cohort_geometry',spawned_actors=len(owned))
                engine.ensure_geometry()
                geometry_report=cohort_geometry(world,carla,owned,engine.map_geometry,engine.static_geometry,
                                               progress=getattr(engine,'progress',None))
            except Exception as exc:
                geometry_errors.append('%s: %s' % (type(exc).__name__,exc))
        if probe_ticks:
            dynamic=probe_cohort(world,carla,owned,attempts,probe_ticks,engine,scene)
    finally:
        cleanup_errors = []
        cleanup_evidence={'method':'actor_destroy_without_client'}
        if engine is not None:
            cleanup_errors,cleanup_evidence=destroy_owned_without_tick(engine.client,carla,created,progress)
        else:
            for actor in reversed(created):
                try:
                    if progress:progress('owned_cleanup',spawned_actor_id=actor.id)
                    if actor.destroy() is False:
                        cleanup_errors.append('destroy returned false for %s' % actor.id)
                except Exception as exc:
                    cleanup_errors.append('%s: %s' % (actor.id,exc))
    background = [a for a in attempts if a['role'] != 'target']
    target_attempts = [a for a in attempts if a['role'] == 'target']
    return {'attempts':attempts,'context_complete':all(a['accepted'] for a in background),
            'targets_all_accepted':bool(target_attempts) and all(a['accepted'] for a in target_attempts),
            'whole_scene_accepted':bool(attempts) and all(a['accepted'] for a in attempts),
             'cleanup_errors':cleanup_errors,'cleanup_evidence':cleanup_evidence,'world_ticks_requested':probe_ticks,
            'geometry_errors':geometry_errors,
            'geometry_3d':geometry_report,'dynamic_probe':dynamic,
            'dynamic_replay':False,
            'speed_observation_available':dynamic is not None and any(m['observed_frames'] for m in dynamic['measurements']),
            'speed_execution_verified':dynamic is not None and not dynamic['errors'] and
                bool(dynamic['measurements']) and all(m['first_tick_velocity_error_mps'] is not None and
                    m['first_tick_velocity_error_mps']<=.5 for m in dynamic['measurements']),
            'speed_execution_error_tolerance_mps':.5,
            'probe_ticks_requested':probe_ticks}


def probe_cohort(world,carla,owned,attempts,ticks,engine,scene):
    """Explicit physics experiment after the unchanged raw spawn requests.

    Not a historical episode replay or a RIFT controller/attention evaluation.
    No snapping, retries, autopilot, velocity resets or pose correction.
    """
    import time
    sensors=[];events=[];errors=[];frames=[]
    requested={a['source_actor_id']:a.get('requested_velocity') for a in attempts}
    start=world.get_snapshot().frame
    def listen(source,event):
        impulse=event.normal_impulse
        events.append({'source_actor_id':source,'frame':int(event.frame),
                       'other_actor_id':int(event.other_actor.id),'other_type_id':event.other_actor.type_id,
                       'normal_impulse':{'x':float(impulse.x),'y':float(impulse.y),'z':float(impulse.z)}})
    try:
        bp=world.get_blueprint_library().find('sensor.other.collision')
        for source,(actor,_) in owned.items():
            sensor=world.spawn_actor(bp,carla.Transform(),attach_to=actor)
            sensors.append(sensor)
            sensor.listen(lambda event,source=source:listen(source,event))
        for step in range(1,ticks+1):
            frame=int(world.tick())
            snapshot=world.get_snapshot()
            if int(snapshot.frame)!=frame or frame!=start+step:
                raise RuntimeError('Unexpected world advancement in exclusive dynamic probe')
            row={'step':step,'frame':frame,'actors':[]}
            for source,(actor,_) in owned.items():
                state=snapshot.find(actor.id)
                if state is None:
                    errors.append('Missing actor snapshot %s at %s' % (source,frame));continue
                t=state.get_transform();v=state.get_velocity();req=requested[source]
                p=dict(scene['ego'] if source==scene['ego']['actor_id'] else
                       next(a['requested_pose'] for a in attempts if a['source_actor_id']==source))
                p.update(x=float(t.location.x),y=float(t.location.y),z=float(t.location.z),
                         yaw_deg=float(t.rotation.yaw),pitch_deg=float(t.rotation.pitch),roll_deg=float(t.rotation.roll),
                         vx=float(v.x),vy=float(v.y),vz=float(v.z),speed_mps=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z))
                item={'source_actor_id':source,'actual_actor_id':actor.id,'pose':p,
                      'velocity_error_from_initial_request_mps':None if req is None else
                      math.sqrt((v.x-req['x'])**2+(v.y-req['y'])**2+(v.z-req['z'])**2)}
                row['actors'].append(item)
            frames.append(row)
        # Collision callbacks are asynchronous. Keep an explicit finite grace
        # period; no event is not proof of exhaustive mesh collision freedom.
        time.sleep(.25)
    except Exception as exc:
        errors.append('%s: %s' % (type(exc).__name__,exc))
    finally:
        for sensor in reversed(sensors):
            try:
                sensor.stop()
                if sensor.destroy() is False:errors.append('Collision sensor destroy returned false')
            except Exception as exc:errors.append('Collision sensor cleanup: %s' % exc)
    measurements=[]
    for source in owned:
        track=[a for f in frames for a in f['actors'] if a['source_actor_id']==source]
        tail=track[-min(5,len(track)):]
        measurements.append({'source_actor_id':source,'observed_frames':len(track),
             'first_tick_velocity_error_mps':track[0]['velocity_error_from_initial_request_mps'] if track else None,
             'tail_z_range_m':max(a['pose']['z'] for a in tail)-min(a['pose']['z'] for a in tail) if tail else None,
             'max_abs_pitch_roll_deg':max(max(abs(a['pose']['pitch_deg']),abs(a['pose']['roll_deg'])) for a in track) if track else None,
             'collision_event_count':sum(e['source_actor_id']==source for e in events)})
    return {'ticks_requested':ticks,'ticks_observed':len(frames),'start_frame':start,
            'end_frame':int(world.get_snapshot().frame),'frames':frames,'measurements':measurements,
            'collision_events':events,'errors':errors,'collision_sensors_attached':len(sensors),
            'sensor_callback_grace_seconds':.25,'normal_rift_controller':False,
            'historical_episode_replay':False,'post_spawn_corrections':False}
