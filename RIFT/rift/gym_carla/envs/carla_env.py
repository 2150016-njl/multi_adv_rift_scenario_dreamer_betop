#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : carla_env.py
@Date    : 2023/10/4
"""
import weakref
import numpy as np
import gym
import carla

from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.cbv.initialization import (
    HighValueCBVInitializer,
    HighValueGeneratedSceneInitializer,
    HighValueSceneSnapshotter,
)
from rift.cbv.recognition.base_cbv import BaseCBVRecog
from rift.gym_carla.action import ACTION_TYPE
from rift.gym_carla.action.cbv_action import CBVAction
from rift.gym_carla.action.ego_action import EgoAction
from rift.gym_carla.done.cbv_done import CBVDone
from rift.gym_carla.done.ego_done import EgoDone
from rift.gym_carla.observation import OBSERVATION_LIST

from rift.gym_carla.utils.common import get_nearby_agents, \
    get_ego_min_dis, \
    store_agent_state
from rift.scenario.scenario_manager.route_scenario import RouteScenario
from rift.scenario.scenario_manager.scenario_manager import ScenarioManager
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.gym_carla.visualization.visualize import draw_trajectory
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.scenario.tools.timer import GameTime
from rift.util.logger import Logger


class CarlaEnv(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator. 
    """

    def __init__(self, env_params, statistics_manager, world=None, cbv_policy: CBVBasePolicy = None, cbv_recog_policy: BaseCBVRecog = None, logger: Logger=None):
        assert world is not None, "the world passed into CarlaEnv is None"

        self.config = None
        self.world = world
        self.logger = logger

        # Record the time of total steps and resetting steps
        self.total_step = 0
        self.env_id = None
        self.ego_vehicle = None
        self.env_params = env_params
        self.ego_policy_type = env_params['ego_policy_type']
        self.cbv_policy_type = env_params['cbv_policy_type']
        self.mode = env_params['mode']
        self.frame_rate = env_params['frame_rate']

        self.need_video_render = env_params['need_video_render']
        self.need_sensor_render = env_params.get('need_sensor_render', self.need_video_render)
        self.camera_sensor = None
        self.ego_collide = None
        self.CBVs_collision_sensor = {}
        self.CBVs_collision = {}

        self.CBVs = {}
        self.gps_route = None
        self.route = None
        self.search_radius = env_params['search_radius']

        # for CBV
        self.cbv_policy = cbv_policy
        self.cbv_recog_policy = cbv_recog_policy
        
        self.statistics_manager = statistics_manager

        # scenario manager
        self.scenario_manager = ScenarioManager(env_params, self.statistics_manager, self.logger)

        # for actions
        self.EgoAction: EgoAction = ACTION_TYPE['ego_action'](env_params['acc_range'], env_params['steer_range'], policy_type=self.ego_policy_type)
        self.CBVAction: CBVAction = ACTION_TYPE['cbv_action'](env_params['acc_range'], env_params['steer_range'], policy_type=self.cbv_policy_type)
        
        # for observations
        self.EgoObs = OBSERVATION_LIST[env_params['ego_obs_type']](env_params, self.EgoAction)
        self.CBVObs = OBSERVATION_LIST[env_params['cbv_obs_type']](env_params, self.CBVAction)
        self.CollectEgoObs = OBSERVATION_LIST[env_params['collect_ego_obs_type']](env_params, self.EgoAction)

        # for route planners
        self.ego_route_planner = self.EgoObs.ego_route_planner
        self.CBV_route_planner = self.CBVObs.CBV_route_planner

        # for rewards
        self.EgoReward = self.EgoObs.ego_reward
        self.CBVReward = self.CBVObs.CBV_reward

        # for dones
        self.EgoDone = EgoDone(self.scenario_manager, self.logger)
        self.CBVDone = CBVDone(self.scenario_manager, self.cbv_recog_policy, self.logger)
    
        # for env wrapper
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']

        # for scenario
        self.ROOT_DIR = env_params['ROOT_DIR']
        self.warm_up_steps = env_params['warm_up_steps']
        self.img_size = env_params['img_size']

        # transfer the CBV route planner to the CBV planning policy and CBV recognition policy
        self.cbv_policy.set_route_planner(self.CBV_route_planner)
        self.cbv_recog_policy.set_route_planner(self.CBV_route_planner)

        self.high_value_initializer = None
        self.high_value_init_result = None
        self.high_value_generation_initializer = None
        self.high_value_generation_result = None
        self.high_value_generation_rollout_results = []
        self.hv_generation_register_cbv = bool(env_params.get('hv_generation_register_cbv', False))
        self.hv_generation_register_cbv_from_audit = bool(
            env_params.get('hv_generation_register_cbv_from_audit', False)
        )
        register_count = env_params.get('hv_generation_register_cbv_count', None)
        self.hv_generation_register_cbv_count = (
            int(env_params.get('hv_num_generate_agents', 2))
            if register_count is None else int(register_count)
        )
        self.high_value_snapshotter = HighValueSceneSnapshotter(
            actor_radius=float(env_params.get('hv_actor_radius', env_params.get('search_radius', 80.0))),
            map_radius=float(env_params.get('hv_map_radius', env_params.get('hv_init_map_radius', 64.0))),
            max_route_points=int(env_params.get('hv_max_route_points', env_params.get('hv_init_max_route_points', 160))),
            max_map_lanes=int(env_params.get('hv_max_map_lanes', env_params.get('hv_init_max_map_lanes', 120))),
            candidate_fov=float(env_params.get('hv_sd_fov', 64.0)),
            forward_min_x=float(env_params.get('hv_generation_forward_min_x', 0.0)),
            target_fov_margin=float(env_params.get('hv_generation_target_fov_margin', 2.0)),
            logger=logger,
        )
        if env_params.get('high_value_init', False):
            self.high_value_initializer = HighValueCBVInitializer(env_params, self.world, self.logger)
        if env_params.get('high_value_init_generation', False):
            self.high_value_generation_initializer = HighValueGeneratedSceneInitializer(env_params, self.world, self.logger)

    def _create_sensors(self):
        if self.need_sensor_render:
            # camera sensor
            self.camera_img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            # self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))  # for ego view
            self.camera_trans = carla.Transform(carla.Location(x=-6.5, y=0., z=6.5),carla.Rotation(pitch=-30.0))  # for third-person view
            # self.camera_trans = carla.Transform(carla.Location(x=12., y=0., z=20.),carla.Rotation(pitch=-90.0))  # god view
            self.camera_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.camera_bp.set_attribute('image_size_x', str(self.img_size[0]))
            self.camera_bp.set_attribute('image_size_y', str(self.img_size[1]))
            self.camera_bp.set_attribute('fov', '120')
            # self.camera_bp.set_attribute('lens_circle_multiplier', str(3.0))
            self.camera_bp.set_attribute('lens_circle_falloff', str(0.0))
            self.camera_bp.set_attribute('motion_blur_intensity', str(0.1))
            self.camera_bp.set_attribute('motion_blur_max_distortion', str(0.1))
            # self.camera_bp.set_attribute('chromatic_aberration_intensity', str(0.5))
            # self.camera_bp.set_attribute('chromatic_aberration_offset', str(0))
            # Set the time in seconds between sensor captures
            self.camera_bp.set_attribute('sensor_tick', str(1.0 / self.frame_rate))

    def _create_scenario(self, config: RouteScenarioConfiguration, env_id: int):
        self.logger.log(f">> Loading scenario: {config.name}")
        # create scenarios according to different types
        scenario = RouteScenario(
            world=self.world,
            config=config,
            env_id=env_id,
            env_params=self.env_params,
            cbv_recog_policy=self.cbv_recog_policy,
            EgoAction=self.EgoAction,
            CBVAction=self.CBVAction,
            logger=self.logger
        )
        # init scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario, config)  # The scenario manager only controls the RouteScenario

        # init for ego controller
        self.route = self.scenario_manager.route_scenario.route  # the global route
        self.gps_route = self.scenario_manager.route_scenario.gps_route  # the global gps route
        self.global_route_waypoints = self.scenario_manager.route_scenario.global_route_waypoints
        # init for reward
        self.EgoReward.set_ego_vehicle(self.ego_vehicle)
        self.CBVReward.set_ego_vehicle(self.ego_vehicle)
        # init for done
        self.EgoDone.set_ego_vehicle(self.ego_vehicle)
        self.CBVDone.set_ego_vehicle(self.ego_vehicle)

    def _run_scenario(self):
        self.scenario_manager.run_scenario()  # init the background vehicle

    def register_CBV_sensor(self, CBV):
        blueprint = CarlaDataProvider._blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=CBV)

        # use weak reference to avoid memory leak
        self_weakref = weakref.ref(self)

        def count_collisions(event):
            self_strongref = self_weakref()
            if self_strongref is not None:
                self_strongref.CBVs_collision[event.actor.id] = {
                    'other_actor_id': event.other_actor.id,
                    'normal_impulse': [event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z]
                }

        collision_sensor.listen(lambda event: count_collisions(event))
        self.CBVs_collision_sensor[CBV.id] = collision_sensor
        self.CBVs_collision[CBV.id] = None

    def CBVs_recog(self):
        if (
            self.high_value_initializer is not None and
            self.high_value_initializer.should_lock_dynamic_cbv(self.high_value_init_result)
        ):
            return

        # Skip CBV calculations if conditions are not met
        if (
            self.cbv_policy_type == 'rule-based' or
            len(self.CBVs) >= self.cbv_recog_policy.max_agent_num or
            self.time_step % 2 != 0 or
            self.time_step <= 25
        ):
            return

        # Select the candidate CBV
        CBVs = self.cbv_recog_policy.get_CBVs(
            self.ego_vehicle,
            set(self.CBVs.keys()),
            self.local_route_waypoints,
            self.rest_route_waypoints,
            self.red_light_state
        )

        for CBV in CBVs:
            self._activate_CBV(CBV)

    def _activate_CBV(self, CBV):
        assert CBV.id not in self.CBVs, 'The selected CBV should not in the existing CBV list'

        self.CBVs[CBV.id] = CBV
        CBV.set_autopilot(False, CarlaDataProvider.get_traffic_manager_port())  # prepared to be controlled
        self.register_CBV_sensor(CBV)
        CarlaDataProvider.add_CBV(self.ego_vehicle, CBV)

        # update the nearby vehicles around the CBV
        CarlaDataProvider.set_CBV_nearby_agents(self.ego_vehicle, CBV, get_nearby_agents(CBV, self.search_radius))

    def _register_high_value_generated_cbvs(self, result):
        if not isinstance(result, dict):
            return []
        result["cbv_registration_enabled"] = bool(self.hv_generation_register_cbv)
        result.setdefault("cbv_registered_actor_ids", [])
        result.setdefault("cbv_registration_failures", [])
        result.setdefault("cbv_registration_count", 0)
        # Post-spawn RIFT labels are evaluation statistics by default.  They
        # must not decide whether a generated actor remains in the scene or
        # becomes Pluto-controlled.  The legacy audit-driven registration is
        # retained only as an explicit compatibility ablation.
        if self.hv_generation_register_cbv_from_audit:
            candidate_ids = [
                int(actor_id) for actor_id in result.get("generated_rift_positive_actor_ids", [])
            ]
            registration_mode = "legacy_audit"
        else:
            candidate_ids = []
            registration_mode = "normal_rift_recognition_only"
        result["cbv_registration_mode"] = registration_mode
        result["cbv_registration_candidate_actor_ids"] = candidate_ids
        result["cbv_registration_requested_count"] = int(self.hv_generation_register_cbv_count)
        result["cbv_registration_available_slots"] = max(
            0,
            int(self.cbv_recog_policy.max_agent_num) - len(self.CBVs),
        )
        result["cbv_registration_max_new"] = int(max(0, min(
            int(self.hv_generation_register_cbv_count),
            int(result["cbv_registration_available_slots"]),
        )))
        if not self.hv_generation_register_cbv:
            self._update_high_value_generation_dump(result)
            return []

        candidate_ids = list(result["cbv_registration_candidate_actor_ids"])
        max_new = int(result["cbv_registration_max_new"])
        registered = []
        failures = []
        for actor_id in candidate_ids[:max_new]:
            if actor_id in self.CBVs:
                registered.append(actor_id)
                continue
            if actor_id not in self.CBV_route_planner.inter_info:
                failures.append({
                    "actor_id": int(actor_id),
                    "failure_reason": "missing_interaction_info",
                })
                continue
            actor = CarlaDataProvider.get_actor_by_id(actor_id)
            if actor is None or not getattr(actor, "is_alive", False):
                failures.append({
                    "actor_id": int(actor_id),
                    "failure_reason": "actor_unavailable",
                })
                continue
            try:
                self._activate_CBV(actor)
                registered.append(actor_id)
            except Exception as exc:
                failures.append({
                    "actor_id": int(actor_id),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                })

        result["cbv_registered_actor_ids"] = registered
        result["cbv_registration_count"] = len(registered)
        result["cbv_registration_failures"] = failures
        result["cbv_route_update_success"] = False
        result["cbv_route_update_failure"] = None
        if registered:
            try:
                self.CBV_route_planner.update_CBV(self.rest_route_waypoints)
                result["cbv_route_update_success"] = True
            except Exception as exc:
                result["cbv_route_update_failure"] = f"{type(exc).__name__}: {exc}"
        self._update_high_value_generation_dump(result)
        return registered

    def _update_high_value_generation_dump(self, result):
        if self.high_value_generation_initializer is None:
            return
        if not hasattr(self.high_value_generation_initializer, "update_dump_result"):
            return
        try:
            self.high_value_generation_initializer.update_dump_result(result)
        except Exception as exc:
            if self.logger:
                self.logger.log(f">> HV generation dump update failed: {exc}", color="yellow")

    def _remove_high_value_generated_actors_for_update(self, frame_id):
        if self.high_value_generation_initializer is None:
            return []
        if not getattr(self.high_value_generation_initializer, "rolling_force_update", False):
            return []
        if not self.high_value_generation_initializer.should_attempt_rollout(frame_id):
            return []

        removed = []
        actor_ids = self.high_value_generation_initializer.active_generated_actor_ids(self.ego_vehicle)
        for actor_id in actor_ids:
            self.high_value_generation_initializer.mark_generated_actor_event(
                actor_id,
                frame_id=frame_id,
                event='force_update_removal',
                terminal=True,
            )
            actor = CarlaDataProvider.get_actor_by_id(actor_id)
            if actor_id in self.CBVs:
                self.CBVs.pop(actor_id, None)
                self._remove_CBV_sensor(actor_id)
                if actor is not None:
                    try:
                        CarlaDataProvider.CBV_terminate(self.ego_vehicle, actor)
                    except Exception:
                        pass
                    try:
                        CarlaDataProvider.pop_CBV_nearby_agents(self.ego_vehicle, actor)
                    except Exception:
                        pass
            if CarlaDataProvider.actor_id_exists(actor_id):
                CarlaDataProvider.remove_actor_by_id(actor_id)
            removed.append(int(actor_id))
        if removed:
            try:
                self.CBV_route_planner.update_CBV(self.rest_route_waypoints)
            except Exception as exc:
                if self.logger:
                    self.logger.log(
                        f">> HV generation force-update route cleanup failed: {type(exc).__name__}: {exc}",
                        color="yellow",
                    )
        if removed and self.logger:
            self.logger.log(
                f">> HV generation force-update removed generated actors at frame {frame_id}: {removed}",
                color="yellow",
            )
        return removed

    def reset(self, config: RouteScenarioConfiguration, env_id: int):
        self.config = config
        self.index = config.index
        self.env_id = env_id

        # create sensor
        self._create_sensors()

        # create RouteScenario, scenario manager, ego_vehicle etc.
        self._create_scenario(config, env_id)
        
        # generate the initial background vehicles
        self._run_scenario()
        
        # atach sensor
        self._attach_sensor()

        # reset the route planner
        self.ego_route_planner.reset(self.ego_vehicle, self.global_route_waypoints)
        self.CBV_route_planner.reset(self.ego_vehicle, self.global_route_waypoints)
        self.local_route_waypoints, self.rest_route_waypoints, self.red_light_state = self.ego_route_planner.run_step()
        self.CBV_route_planner.run_step(self.local_route_waypoints, self.rest_route_waypoints)
        # Update time_steps
        self.time_step = 0
        self.high_value_generation_rollout_results = []

        # applying setting can tick the world and get data from sensors
        # removing this block will cause error: AttributeError: 'NoneType' object has no attribute 'raw_data'
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()

        # update the info in the CarlaDataProvider
        CarlaDataProvider.on_carla_tick()

        if self.high_value_generation_initializer is not None:
            self.high_value_generation_result = self.high_value_generation_initializer.initialize(
                ego_vehicle=self.ego_vehicle,
                route_config=config,
                cbv_recog_policy=self.cbv_recog_policy,
                cbv_route_planner=self.CBV_route_planner,
                ego_rest_route_waypoints=self.rest_route_waypoints,
                existing_cbv_ids=set(self.CBVs.keys()),
            )
            self._register_high_value_generated_cbvs(self.high_value_generation_result)
            self._prune_dead_unactivated_actors()

        high_value_init_attempted = False
        high_value_initialized = False
        if self.high_value_initializer is not None:
            high_value_init_attempted = True
            self.high_value_init_result = self.high_value_initializer.initialize(
                ego_vehicle=self.ego_vehicle,
                route_config=config,
                cbv_recog_policy=self.cbv_recog_policy,
                cbv_route_planner=self.CBV_route_planner,
                ego_rest_route_waypoints=self.rest_route_waypoints,
                existing_cbv_ids=set(self.CBVs.keys()),
            )
            if self.high_value_init_result.get('success', False):
                cbv_actors = self.high_value_init_result.get(
                    'cbv_actors',
                    self.high_value_init_result.get('selected_actors', []),
                )
                for CBV in cbv_actors:
                    self._activate_CBV(CBV)
                high_value_initialized = True
            elif (
                self.high_value_initializer.fail_on_invalid and
                not self.high_value_init_result.get('collect_only', False)
            ):
                raise RuntimeError(self.high_value_init_result.get('failure_reason', 'high-value init failed'))

        # recognize the CBV and update CBV global routes
        if not high_value_init_attempted and not high_value_initialized:
            self.CBVs_recog()
        self.CBV_route_planner.update_CBV(self.rest_route_waypoints)

        # update agent nearby vehicles
        self._update_agent_nearby_vehicle()

        # update the self.ego_collide status
        self.update_ego_collision()

        # store all actor state
        store_agent_state(self.ego_vehicle)

        # get obs
        ego_obs = self.EgoObs.get_obs(self.ego_vehicle)
        CBVs_obs = self.CBVObs.get_obs(self.ego_vehicle)

        info = self.get_info()
        info.update({
            'route_waypoints': self.global_route_waypoints,  # the global route waypoints
            'gps_route': self.gps_route,  # the global gps route
            'route': self.route,  # the global route
        })
        if self.high_value_initializer is not None:
            info['high_value_init'] = self.high_value_initializer.result_for_info(self.high_value_init_result)
        if self.high_value_generation_initializer is not None:
            info['high_value_init_generation'] = dict(self.high_value_generation_result or {})
            info['high_value_generation_rollout_count'] = len(self.high_value_generation_rollout_results)

        return ego_obs, CBVs_obs, info

    def build_high_value_generation_snapshot(self, include_rift_labels=True):
        return self.high_value_snapshotter.build(
            ego_vehicle=self.ego_vehicle,
            route_config=self.config,
            ego_rest_route_waypoints=self.rest_route_waypoints,
            cbv_recog_policy=self.cbv_recog_policy,
            cbv_route_planner=self.CBV_route_planner,
            existing_cbv_ids=[],
            include_rift_labels=include_rift_labels,
            frame_id=self.time_step,
        )

    def dump_high_value_generation_snapshot(self, output_dir, prefix=None, include_rift_labels=True):
        snapshot = self.build_high_value_generation_snapshot(include_rift_labels=include_rift_labels)
        path = self.high_value_snapshotter.dump(snapshot, output_dir=output_dir, prefix=prefix)
        return snapshot, path

    def _attach_sensor(self):
        if self.need_sensor_render:
            self_weakref = weakref.ref(self)  # weak reference of self

            # Add camera sensor
            self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
            self.camera_sensor.listen(lambda data, self_ref=self_weakref: get_camera_img(self_ref(), data) if self_ref() else None)

            def get_camera_img(ego_self, data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                ego_self.camera_img = array

    def visualize_actors(self):
        # visualize the past trajectory of all the actor on the map
        if self.need_video_render:
            draw_trajectory(self.world, self.ego_vehicle, self.time_step)
            self.ego_route_planner.vis_route()
            self.CBV_route_planner.vis_route()

    def step_before_tick(self, ego_action, cbv_actions):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp

                # update the actions
                self.scenario_manager.get_update(timestamp, ego_action, cbv_actions)

            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    def step_after_tick(self):
        if (
            self.high_value_generation_initializer is not None
            and self.high_value_generation_initializer.sd_background_model_only_traffic
        ):
            self.high_value_generation_initializer.enforce_model_only_traffic(self.ego_vehicle)

        # route planner
        self.local_route_waypoints, self.rest_route_waypoints, self.red_light_state = self.ego_route_planner.run_step()
        self.CBV_route_planner.run_step(self.local_route_waypoints, self.rest_route_waypoints)

        if self.high_value_generation_initializer is not None:
            removed_for_force_update = self._remove_high_value_generated_actors_for_update(self.time_step)
            rollout_result = self.high_value_generation_initializer.maybe_generate_rollout(
                ego_vehicle=self.ego_vehicle,
                route_config=self.config,
                cbv_recog_policy=self.cbv_recog_policy,
                cbv_route_planner=self.CBV_route_planner,
                ego_rest_route_waypoints=self.rest_route_waypoints,
                frame_id=self.time_step,
            )
            if rollout_result is not None:
                if removed_for_force_update:
                    rollout_result["force_update_removed_actor_ids"] = removed_for_force_update
                self._register_high_value_generated_cbvs(rollout_result)
                self.high_value_generation_rollout_results.append(rollout_result)
                self._prune_dead_unactivated_actors()

        # update agent nearby vehicles
        self._update_agent_nearby_vehicle()

        # store all actor state
        store_agent_state(self.ego_vehicle)

        # update the running status and check whether terminate or not
        self.scenario_manager.update_running_status(self.CBVs_collision)

        # update the self.ego_collide status
        self.update_ego_collision()

        # data before CBV recog
        data = self.get_data_before_CBV_recog()

        # if CBV collided, then remove it
        self._remove_and_clean_CBV(data)

        # recognize the updated CBV and update CBV global routes
        if self.scenario_manager.running:
            self.CBVs_recog()
            self.CBV_route_planner.update_CBV(self.rest_route_waypoints)
        if self.high_value_generation_initializer is not None:
            self.high_value_generation_initializer.observe_generated_actor_lifecycle(
                frame_id=self.time_step,
                ego_vehicle=self.ego_vehicle,
                active_cbv_ids=self.CBVs.keys(),
            )

        # data after CBV recog
        data = self.get_data_after_CBV_recog(data)

        # self.visualize_actors()  # visualize the controlled bv and the waypoints in clients side after tick

        # Update time steps
        self.time_step += 1
        self.total_step += 1

        return data

    def get_data_before_CBV_recog(self):
        return {
            'CBVs_next_obs': self.CBVObs.get_obs(self.ego_vehicle),
            'CBVs_reward': self.CBVReward.get_reward(self.CBVs_collision),
            'next_info': self.get_info(),
            'CBVs_terminated': self.CBVDone.terminated(self.CBVs_collision),
            'CBVs_truncated': self.CBVDone.truncated(),
            'CBVs_done': self.CBVDone.done(),
            'CBV_ids': list(self.CBVs.keys())
        }

    def get_data_after_CBV_recog(self, data):
        # get ego training data
        data['ego_next_obs'] = self.EgoObs.get_obs(self.ego_vehicle)
        data['ego_reward'] = self.EgoReward.get_reward(self.local_route_waypoints, self.ego_collide)
        data['ego_terminal'] = self.EgoDone.terminated()
        data['ego_id'] = self.ego_vehicle.id

        # get update transition data
        data['ego_transition_obs'] = self.EgoObs.get_tran_obs(self.ego_vehicle, data['ego_next_obs'])
        data['CBVs_transition_obs'] = self.CBVObs.get_tran_obs(self.ego_vehicle, data['CBVs_next_obs'])
        data['transition_info'] = self.get_info()  # info of updated CBV

        return data

    def get_info(self):
        info = {'env_id': self.env_id}
        # collect data
        info.update(self.get_collection_data()) if self.mode == 'collect_data' else None
        if self.high_value_generation_initializer is not None:
            info['high_value_init_generation'] = dict(self.high_value_generation_result or {})
            info['high_value_generation_rollout_count'] = len(self.high_value_generation_rollout_results)
            if self.high_value_generation_rollout_results:
                info['high_value_generation_latest_rollout'] = dict(self.high_value_generation_rollout_results[-1])
            generated_by_source = self.high_value_generation_initializer.active_generated_actor_ids_by_source(
                self.ego_vehicle
            )
            background_ids = generated_by_source.get('scenario_dreamer_background', [])
            high_value_ids = generated_by_source.get('high_value', [])
            info['scenario_dreamer_generation_actor_ids'] = background_ids
            info['scenario_dreamer_generation_cbv_actor_ids'] = [
                int(actor_id) for actor_id in background_ids if actor_id in self.CBVs
            ]
            info['high_value_generation_actor_ids'] = high_value_ids
            info['high_value_generation_cbv_actor_ids'] = [
                int(actor_id) for actor_id in high_value_ids if actor_id in self.CBVs
            ]

        return info

    def get_collection_data(self):
        collect_data = {
            'collect_ego_obs': self.CollectEgoObs.get_obs(self.ego_vehicle),
            'collect_ego_min_dis': get_ego_min_dis(self.ego_vehicle, self.search_radius),
            'collect_ego_collide': float(self.ego_collide),
            'camera_image': self.camera_img
        }
        return collect_data

    def _update_agent_nearby_vehicle(self):
        # update ego nearby vehicles
        CarlaDataProvider.set_ego_nearby_agents(self.ego_vehicle, get_nearby_agents(self.ego_vehicle, self.search_radius))
        # update the nearby vehicles around the CBV
        for CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).values():
            CarlaDataProvider.set_CBV_nearby_agents(self.ego_vehicle, CBV, get_nearby_agents(CBV, self.search_radius))

    def _prune_dead_unactivated_actors(self):
        route_scenario = getattr(self.scenario_manager, 'route_scenario', None)
        if route_scenario is None or not hasattr(route_scenario, 'unactivated_actors'):
            return
        route_scenario.unactivated_actors = [
            actor for actor in route_scenario.unactivated_actors
            if actor is not None and getattr(actor, 'is_alive', False)
        ]

    def update_ego_collision(self):
        self.ego_collide = self.scenario_manager.ego_collision or any(
            collision_event
            and collision_event['other_actor_id'] == self.ego_vehicle.id
            for collision_event in self.CBVs_collision.values()
        )

        self.logger.log(f'>> Ego collide', color='yellow') if self.ego_collide else None

    def _remove_sensor(self):
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.CBVs_collision_sensor:
            # remove the collision sensor that has not been destroyed
            for sensor in self.CBVs_collision_sensor.values():
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            self.CBVs_collision_sensor = {}

    def _remove_CBV_sensor(self, CBV_id):
        sensor = self.CBVs_collision_sensor.pop(CBV_id, None)
        if sensor is not None and sensor.is_alive:
            sensor.stop()
            sensor.destroy()
        self.CBVs_collision.pop(CBV_id, None)

    def _remove_ego(self):
        if self.ego_vehicle is not None and CarlaDataProvider.actor_id_exists(self.ego_vehicle.id):
            CarlaDataProvider.remove_actor_by_id(self.ego_vehicle.id)
        self.ego_vehicle = None

    def _remove_and_clean_CBV(self, data):
        # remove the truncated CBV from the CBV list and set them free to normal bvs
        CBVs_truncated = data['CBVs_truncated']
        for CBV_id, truncated in CBVs_truncated.items():
            if truncated:
                CBV = self.CBVs.pop(CBV_id, None)
                if CBV is not None:
                    # remove the CBV collision sensor
                    self._remove_CBV_sensor(CBV_id)
                    # remove the truncated CBV from existing CBV lists
                    CBV.set_autopilot(True, CarlaDataProvider.get_traffic_manager_port())  # set the original CBV to normal bvs
                    CarlaDataProvider.pop_CBV_nearby_agents(self.ego_vehicle, CBV)
                    CarlaDataProvider.CBV_back_to_BV(self.ego_vehicle, CBV)
                    if self.high_value_generation_initializer is not None:
                        self.high_value_generation_initializer.mark_generated_actor_event(
                            CBV_id,
                            frame_id=self.time_step,
                            event='rift_cbv_returned_to_tm',
                        )

        # clean the terminated CBV
        CBVs_terminated = data['CBVs_terminated']
        for CBV_id, terminated in CBVs_terminated.items():
            if terminated:
                CBV = self.CBVs.pop(CBV_id, None)
                if CBV is not None:
                    terminated_after_collision = bool(self.CBVs_collision.get(CBV_id))
                    # remove sensor
                    self._remove_CBV_sensor(CBV_id)
                    # remove the CBV from the CBV dict
                    CarlaDataProvider.CBV_terminate(self.ego_vehicle, CBV)
                    if CBV_id in CarlaDataProvider.get_CBVs_reach_goal_by_ego(self.ego_vehicle.id).keys():
                        # set the goal reaching CBV free
                        CBV.set_autopilot(True, CarlaDataProvider.get_traffic_manager_port())
                    else:
                        # clean the CBV from the environment
                        if CarlaDataProvider.actor_id_exists(CBV_id):
                            CarlaDataProvider.remove_actor_by_id(CBV_id)
                    CarlaDataProvider.pop_CBV_nearby_agents(self.ego_vehicle, CBV)
                    if self.high_value_generation_initializer is not None:
                        event = 'rift_cbv_terminated_after_collision' if terminated_after_collision else 'rift_cbv_terminated'
                        self.high_value_generation_initializer.mark_generated_actor_event(
                            CBV_id,
                            frame_id=self.time_step,
                            event=event,
                            terminal=CBV_id not in CarlaDataProvider.get_CBVs_reach_goal_by_ego(self.ego_vehicle.id),
                        )

    def _reset_variables(self):
        self.CBVs = {}
        self.gps_route = None
        self.route = None
        self.global_route_waypoints = None
        self.local_route_waypoints = None
        self.CBVs_collision = {}

    def clean_up(self):
        if self.high_value_generation_initializer is not None:
            try:
                self.high_value_generation_initializer.finalize_generated_actor_lifecycle(self.time_step)
            except Exception as exc:
                if self.logger:
                    self.logger.log(f">> HV lifecycle finalization failed: {exc}", color="yellow")

        # remove temp variables
        self._reset_variables()

        # remove the sensor only when evaluating
        self._remove_sensor()

        # destroy criterion sensors on the ego vehicle
        self.scenario_manager.clean_up()

        # remove the ego vehicle after removing all the sensors
        self._remove_ego()
