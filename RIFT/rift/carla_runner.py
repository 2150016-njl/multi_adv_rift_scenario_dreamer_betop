#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : carla_runner.py
@Date    : 2023/10/4
"""

import copy
import math
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
from rift.scenario.tools.global_route_planner import GlobalRoutePlanner
from rift.cbv.recognition.base_cbv import BaseCBVRecog
from rift.ego.base_policy import EgoBasePolicy
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.gym_carla.buffer.collect_buffer import CollectBuffer
from rift.gym_carla.buffer.ego_rollout_buffer import EgoRolloutBuffer
from rift.gym_carla.envs.env_wrapper import VectorWrapper

from rift.ego import EGO_POLICY_LIST
from rift.cbv.planning import CBV_POLICY_LIST
from rift.cbv.recognition import CBV_RECOGNITION_LIST

from rift.scenario.scenario_data_parser import ScenarioDataParser
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.scenario.scenario_data_loader import EvalDataLoader, TrainDataLoader

from rift.scenario.statistics_manager import StatisticsManager
from rift.scenario.tools.route_manipulation import get_latlon_ref
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.util.logger import Logger, setup_logger_dir


MODE_SEED = {
    'train_ego': 1000,
    'train_cbv': 2000,
    'collect_data': 3000,
    'eval': 4000,
    'collect_hv_init': 5000,
    'collect_hv_generation': 6000,
    'collect_hv_future_outcome': 7000,
}


class CarlaRunner:
    def __init__(self, client, traffic_manager, configs: List[Dict]):
        config, ego_config, cbv_config, cbv_recog_config, collect_data_config = configs
        self.config = config
        self.cbv_config = cbv_config
        self.ego_config = ego_config
        self.collect_data_config = collect_data_config
        self.current_map = None

        self.sampling_resolution = 1.0  # used for global route planner
        self.seed = config['seed']
        self.output_dir = config['output_dir']
        self.mode = config['mode']
        self.num_scenario = config['num_scenario']  # default 2
        self.frame_rate = config['frame_rate']
        self.resume = config['resume']
        self.cbv_policy_name = cbv_config['policy_name']
        self.ego_policy_name = ego_config['policy_name']

        # apply settings to carla
        self.client = client
        self.traffic_manager = traffic_manager
        self.world = None
        self.env = None
        self.need_sensor_render = config['render'] or self.mode == 'collect_data'
        self.need_video_render = self.need_sensor_render or config.get('save_bev', False)
        self.rift_hd_map_dir = config.get('rift_hd_map_dir')
        self.rift_speed_limits_dir = config.get('rift_speed_limits_dir')
        if self.rift_hd_map_dir:
            os.environ["RIFT_HD_MAP_DIR"] = self.rift_hd_map_dir
        if self.rift_speed_limits_dir:
            os.environ["RIFT_SPEED_LIMITS_DIR"] = self.rift_speed_limits_dir

        self.env_params = {
            'mode': self.mode,  # the mode of the script
            'num_scenario': self.num_scenario,  # the number of scenarios
            'search_radius': 60,  # the default search radius
            'ego_policy_type': ego_config['policy_type'],  # ego policy type
            'cbv_policy_type': cbv_config['policy_type'],  # cbv policy type
            'ego_obs_type': ego_config['obs_type'],  # ego obs type for ego navigation
            'cbv_obs_type': cbv_config['obs_type'],  # cbv obs type for cbv navigation
            'collect_ego_obs_type': collect_data_config['ego_obs_type'],  # the ego obs type used in collecting data
            'ROOT_DIR': config['ROOT_DIR'],
            'need_video_render': self.need_video_render,  # whether to activate the render
            'need_sensor_render': self.need_sensor_render,  # whether to spawn CARLA camera sensors
            'signalized_junction': False,  # whether the signal controls the junction
            'warm_up_steps': 4,  # number of ticks after spawning the vehicles
            'img_size': [1280, 720],  # screen size of the camera image
            'acc_range': [-3.0, 3.0],  # continuous acceleration range
            'steer_range': [-0.3, 0.3],  # continuous steering angle range
            'out_lane_thres': 4,  # threshold for out of lane (meter)
            'desired_speed': 8,  # desired speed (m/s)
            'frame_rate': self.frame_rate,  # frame rate of the simulation
            'high_value_init': config.get('high_value_init', False),
            'hv_init_generator_entry': config.get('hv_init_generator_entry'),
            'hv_init_checkpoint': config.get('hv_init_checkpoint'),
            'hv_init_device': config.get('hv_init_device', 'cuda'),
            'hv_init_target_count': config.get('hv_init_target_count', 2),
            'hv_init_apply_top_k': config.get('hv_init_apply_top_k', 2),
            'hv_init_max_attempts': config.get('hv_init_max_attempts', 1),
            'hv_init_require_target_mask': config.get('hv_init_require_target_mask', True),
            'hv_init_fail_on_invalid': config.get('hv_init_fail_on_invalid', True),
            'hv_init_validation_mode': config.get('hv_init_validation_mode', 'audit'),
            'hv_init_lock_selected_cbvs': config.get('hv_init_lock_selected_cbvs', True),
            'hv_init_reject_extra_rift_positive': config.get('hv_init_reject_extra_rift_positive', False),
            'hv_init_tick_after_apply': config.get('hv_init_tick_after_apply', True),
            'hv_init_dump_dir': config.get('hv_init_dump_dir'),
            'hv_init_save_viz': config.get('hv_init_save_viz', True),
            'hv_init_debug_generator': config.get('hv_init_debug_generator', False),
            'hv_init_collect_only': config.get('hv_init_collect_only', False),
            'hv_init_allow_untrained_generator': config.get('hv_init_allow_untrained_generator', False),
            'hv_init_min_selected_pair_distance': config.get('hv_init_min_selected_pair_distance', 0.0),
            'hv_init_max_abs_xy_delta': config.get('hv_init_max_abs_xy_delta', 30.0),
            'hv_init_max_abs_yaw_delta': config.get('hv_init_max_abs_yaw_delta', 180.0),
            'hv_init_max_speed_mps': config.get('hv_init_max_speed_mps', 40.0),
            'hv_init_max_route_points': config.get('hv_init_max_route_points', 160),
            'hv_init_map_radius': config.get('hv_init_map_radius', 64.0),
            'hv_init_max_map_lanes': config.get('hv_init_max_map_lanes', 120),
            'hv_model_backbone': config.get('hv_model_backbone', 'autoencoder'),
            'hv_model_autoencoder_checkpoint': config.get('hv_model_autoencoder_checkpoint'),
            'hv_model_cfg_path': config.get('hv_model_cfg_path'),
            'hv_model_ldm_checkpoint': config.get('hv_model_ldm_checkpoint'),
            'hv_model_ldm_cfg_path': config.get('hv_model_ldm_cfg_path'),
            'hv_model_state_source': config.get('hv_model_state_source', 'ldm_inpainting'),
            'hv_model_num_generation_candidates': config.get('hv_model_num_generation_candidates', 8),
            'hv_model_hidden_dim': config.get('hv_model_hidden_dim', 256),
            'hv_model_num_layers': config.get('hv_model_num_layers', 3),
            'hv_model_num_heads': config.get('hv_model_num_heads', 8),
            'hv_model_dropout': config.get('hv_model_dropout', 0.1),
            'hv_sd_fov': config.get('hv_sd_fov', 64.0),
            'hv_sd_max_num_lanes': config.get('hv_sd_max_num_lanes', 100),
            'hv_sd_num_points_per_lane': config.get('hv_sd_num_points_per_lane', 20),
            'hv_sd_map_id': config.get('hv_sd_map_id', 0),
            'high_value_init_generation': config.get('high_value_init_generation', False),
            'hv_generation_generator_entry': config.get(
                'hv_generation_generator_entry',
                'high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator',
            ),
            'hv_generation_checkpoint': config.get('hv_generation_checkpoint'),
            'hv_generation_device': config.get('hv_generation_device', config.get('hv_init_device', 'cuda')),
            'hv_generation_mode': config.get('hv_generation_mode', 'inpainting'),
            'hv_num_generate_agents': config.get('hv_num_generate_agents', 2),
            'hv_generation_forward_min_x': config.get('hv_generation_forward_min_x', 0.0),
            'hv_generation_target_fov_margin': config.get('hv_generation_target_fov_margin', 2.0),
            'hv_actor_radius': config.get('hv_actor_radius', 80.0),
            'hv_map_radius': config.get('hv_map_radius', 64.0),
            'hv_max_route_points': config.get('hv_max_route_points', 160),
            'hv_max_map_lanes': config.get('hv_max_map_lanes', 120),
            'hv_snapshot_map_coordinate_mode': config.get('hv_snapshot_map_coordinate_mode', 'legacy'),
            'hv_generation_dump_dir': config.get('hv_generation_dump_dir') or config.get('hv_dump_dir'),
            'hv_autoencoder_checkpoint': config.get('hv_autoencoder_checkpoint'),
            'hv_ldm_checkpoint': config.get('hv_ldm_checkpoint'),
            'hv_generation_actor_model': config.get('hv_generation_actor_model', 'vehicle.*'),
            'hv_generation_actor_role': config.get('hv_generation_actor_role', 'generated_hv'),
            'sd_background_checkpoint': config.get('sd_background_checkpoint'),
            'sd_background_source': config.get('sd_background_source', 'carla'),
            'sd_background_num_agents': config.get('sd_background_num_agents', 6),
            'sd_background_replace_existing': config.get('sd_background_replace_existing', False),
            'sd_background_replace_radius': config.get('sd_background_replace_radius', 64.0),
            'sd_background_model_only_traffic': config.get('sd_background_model_only_traffic', False),
            'hv_generation_autopilot': config.get('hv_generation_autopilot', True),
            'hv_generation_register_cbv': config.get('hv_generation_register_cbv', False),
            'hv_generation_register_cbv_from_audit': config.get('hv_generation_register_cbv_from_audit', False),
            'hv_generation_register_cbv_count': config.get('hv_generation_register_cbv_count'),
            'hv_generation_clamp_generated_states': config.get('hv_generation_clamp_generated_states', False),
            'hv_min_generated_speed': config.get('hv_min_generated_speed', 0.0),
            'hv_max_generated_speed': config.get('hv_max_generated_speed', 30.0),
            'hv_min_generated_length': config.get('hv_min_generated_length', 2.0),
            'hv_max_generated_length': config.get('hv_max_generated_length', 8.0),
            'hv_min_generated_width': config.get('hv_min_generated_width', 1.4),
            'hv_max_generated_width': config.get('hv_max_generated_width', 3.2),
            'hv_generation_fail_on_invalid': config.get('hv_generation_fail_on_invalid', True),
            'hv_generation_max_attempts': config.get('hv_generation_max_attempts', 8),
            'hv_generation_partial_context': config.get('hv_generation_partial_context', True),
            'hv_snapshot_native_map_export':config.get('hv_snapshot_native_map_export',False),
            'hv_collection_traffic_intensity':config.get('hv_collection_traffic_intensity'),
            'hv_generation_require_hidden_slots': config.get('hv_generation_require_hidden_slots', True),
            'hv_generation_hidden_slots': config.get('hv_generation_hidden_slots'),
            'hv_generation_hidden_forward_min': config.get('hv_generation_hidden_forward_min', 0.0),
            'hv_generation_hidden_forward_max': config.get('hv_generation_hidden_forward_max', 64.0),
            'hv_generation_hidden_lateral_abs': config.get('hv_generation_hidden_lateral_abs', 32.0),
            'hv_generation_rolling': config.get('hv_generation_rolling', False),
            'hv_generation_rolling_interval_steps': config.get('hv_generation_rolling_interval_steps', 50),
            'hv_generation_rolling_min_step': config.get('hv_generation_rolling_min_step', 50),
            'hv_generation_rolling_min_rift_positive': config.get('hv_generation_rolling_min_rift_positive'),
            'hv_generation_gate_on_existing_rift_positive': config.get('hv_generation_gate_on_existing_rift_positive', True),
            'hv_generation_existing_rift_positive_threshold': config.get('hv_generation_existing_rift_positive_threshold', 2),
            'video_render_bounds': config.get('video_render_bounds', 32.0),
            'video_render_map_radius': config.get('video_render_map_radius', 64.0),
            'video_render_offset': config.get('video_render_offset', 0.0),
            'hv_generation_rolling_force_update': config.get('hv_generation_rolling_force_update', False),
            'hv_generation_lifetime_radius': config.get('hv_generation_lifetime_radius'),
            'hv_generation_lifecycle_telemetry': config.get('hv_generation_lifecycle_telemetry', False),
            'hv_future_outcome_dir': config.get('hv_future_outcome_dir'),
            'hv_future_outcome_horizon_s': config.get('hv_future_outcome_horizon_s', 5.0),
            'hv_future_outcome_anchor_interval_s': config.get('hv_future_outcome_anchor_interval_s', 1.0),
            'hv_future_outcome_observe_interval_steps': config.get('hv_future_outcome_observe_interval_steps', 1),
            'hv_future_outcome_min_forward_rift_positive': config.get('hv_future_outcome_min_forward_rift_positive', 2),
        }

        # pass config from scenario to ego
        ego_config['mode'] = self.mode
        ego_config['desired_speed'] = self.env_params['desired_speed']
        ego_config['num_scenario'] = config['num_scenario']
        ego_config['cbv_policy_name'] = cbv_config['policy_name']

        # pass config to cbv recog
        cbv_recog_config['desired_speed'] = self.env_params['desired_speed']

        # pass config from ego to cbv
        cbv_config['ego_policy'] = ego_config['policy_name']
        cbv_config['ego_obs_type'] = ego_config['obs_type']
        cbv_config['need_video_render'] = self.env_params['need_video_render']
        cbv_config['desired_speed'] = self.env_params['desired_speed']

        # pass config to collect_data
        collect_data_config['search_radius'] = self.env_params['search_radius']
        collect_data_config['acc_range'] = self.env_params['acc_range']
        collect_data_config['steer_range'] = self.env_params['steer_range']
        collect_data_config['ego_policy_type'] = ego_config['policy_type']

        CarlaDataProvider.set_desired_speed(self.env_params['desired_speed'])
        CarlaDataProvider.set_frame_rate(self.frame_rate)

        # define logger
        logger_dir = setup_logger_dir(
            self.output_dir,
            self.seed,
            self.mode,
            ego=ego_config['policy_name'],
            cbv=cbv_config['policy_name'],
            cbv_recog=cbv_recog_config['type'],
        )
        self.logger = Logger(logger_dir)

        # prepare parameters
        if self.mode == 'train_ego':
            self.save_freq = ego_config['save_freq']
        elif self.mode == 'train_cbv':
            self.save_freq = cbv_config['save_freq']
        elif self.mode == 'collect_data':
            self.data_path = self.collect_data_config['data_path']
        elif self.mode == 'eval':
            self.logger.log('>> Evaluation Mode, analyzing result', 'yellow')
        elif self.mode == 'collect_hv_init':
            self.logger.log('>> High-value initial-state label collection mode', 'yellow')
        elif self.mode == 'collect_hv_generation':
            self.logger.log('>> High-value scene generation snapshot collection mode', 'yellow')
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define agent and scenario
        self.logger.log('>> Mode: ' + self.mode, color="yellow")
        self.logger.log('>> Ego planning method: ' + ego_config['policy_name'], color="yellow")
        self.logger.log('>> CBV planning method: ' + cbv_config['policy_name'], color="yellow")
        self.logger.log('>> CBV recognition method: ' + cbv_recog_config['type'], color="yellow")
        self.logger.log('>> ' + '-' * 40)

        # define ego policy, scenario policy
        self.ego_policy: EgoBasePolicy = EGO_POLICY_LIST[ego_config['policy_name']](ego_config, logger=self.logger)
        self.cbv_policy: CBVBasePolicy = CBV_POLICY_LIST[cbv_config['policy_name']](cbv_config, logger=self.logger)
        self.cbv_recog_policy: BaseCBVRecog = CBV_RECOGNITION_LIST[cbv_recog_config['type']](cbv_recog_config, logger=self.logger)

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town, reset_settings=False)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.no_rendering_mode = not self.need_sensor_render
        settings.tile_stream_distance = 800   # used for large map (currently don't support multi-scenareio)
        settings.actor_active_distance = 800  # used for large map (currently don't support multi-scenareio)
        self.world.apply_settings(settings)
        
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world, town)
        CarlaDataProvider.set_traffic_manager_port(self.config['tm_port'])

        # Wait for the world to be ready
        self.world.tick()

        # deterministic mode for reproducing
        self.traffic_manager.set_random_device_seed(self.seed)
        # set the global distance to leading vehicle
        self.traffic_manager.set_global_distance_to_leading_vehicle(8)
        # change the percentage of speed difference, negative value means exceed the speed limit
        self.traffic_manager.global_percentage_speed_difference(-20)
        # Carla Global Route Planner
        CarlaDataProvider.set_global_route_planner(GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution))
        # Carla Map API init
        CarlaDataProvider.set_map_api(CarlaMap(carla_town_name=town, map_data_dir=self.rift_hd_map_dir))
        # Carla GPS info
        CarlaDataProvider.set_gps_info(get_latlon_ref(self.world))
        # set for cbv recognition
        self.cbv_recog_policy.set_world()

    def train_cbv(self, data_loader: TrainDataLoader, buffer: CBVRolloutBuffer):

        while len(data_loader) > 0:
            
            # simulate multiple scenarios in parallel (usually 2 scenarios)
            sampled_scenario_configs = data_loader.sampler()

            # build the env
            self.build_env(sampled_scenario_configs)

            start_time = time.time()

            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            # start loop
            CBV_episode_reward = []
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=False)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=False)

                # apply action to env and get obs
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)

                # store to the replay buffer
                buffer.store(data_dict)

                # for transition
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])

                scenario_reward = np.concatenate([np.array(list(CBVs_reward.values())) for CBVs_reward in data_dict['CBVs_reward']])
                CBV_episode_reward.append(np.mean(scenario_reward)) if scenario_reward.size != 0 else None

            self.logger.log_route_info(sampled_scenario_configs, np.sum(CBV_episode_reward))
            self.logger.log(f'>> buffer length: {buffer.buffer_pos}')
            self.logger.log(f'>> Scenario Progress: [{data_loader.index}/{data_loader.total}] Time: {time.time() - start_time:.2f}s', 'yellow')
            
            # end up environment
            self.env.clean_up()
            # tick to ensure that all destroy commands are executed
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')

            # train cbv
            self.cbv_policy.log_episode_reward(np.sum(CBV_episode_reward), episode=data_loader.index)
            self.cbv_policy.train(data_loader.index) if buffer.buffer_full else None

            # save checkpoints
            if data_loader.index % self.save_freq == 0 or len(data_loader) == 0:
                self.cbv_policy.save_model(data_loader.index)

            # save video
            if self.need_video_render:
                self.logger.video_render.save_video(self.current_map)
            
            self.logger.log('>> ' + '-' * 40)
        
        self.cbv_policy.finish()

    def train_ego(self, data_loader: TrainDataLoader, buffer: EgoRolloutBuffer):

        while len(data_loader) > 0:
            # simulate multiple scenarios in parallel (usually 2 scenarios)
            sampled_scenario_configs = data_loader.sampler()

            # build the env
            self.build_env(sampled_scenario_configs)

            start_time = time.time()

            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            # start loop
            ego_episode_reward = []
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=False)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=False)

                # apply action to env and get obs
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)

                # store to the replay buffer
                buffer.store(data_dict)

                # for transition
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])

                ego_episode_reward.append(np.mean(data_dict['ego_reward']))

            self.logger.log_route_info(sampled_scenario_configs, np.sum(ego_episode_reward))
            self.logger.log(f'>> buffer length: {buffer.buffer_pos}')
            self.logger.log(f'>> Scenario Progress: [{data_loader.index}/{data_loader.total}] Time: {time.time() - start_time:.2f}s', 'yellow')
            
            # end up environment
            self.env.clean_up()
            # tick to ensure that all destroy commands are executed
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')

            # train on-policy ego
            self.ego_policy.log_episode_reward(np.sum(ego_episode_reward), episode=data_loader.index)
            self.ego_policy.train(data_loader.index) if buffer.buffer_full else None

            # save checkpoints
            if data_loader.index % self.save_freq == 0 or len(data_loader) == 0:
                self.ego_policy.save_model(data_loader.index)

            # save video
            if self.need_video_render:
                self.logger.video_render.save_video(self.current_map)
            
            self.logger.log('>> ' + '-' * 40)

        self.ego_policy.finish()

    def eval(self, data_loader: EvalDataLoader):
        
        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs = data_loader.sampler()

            # build the env
            self.build_env(sampled_scenario_configs)
            
            start_time = time.time()
            
            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=True)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=True)

                # apply action to env and get obs
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)

                # for transition
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])
            
            # Save the progress and write the route statistics
            self.statistics_manager.save_progress(data_loader.index, data_loader.total)
            self.statistics_manager.write_statistics()

            # clean up all things
            self.logger.log(">> All scenarios are completed. Cleaning up all actors")
            self.logger.log(f'>> Scenario Evaluation Progress: [{data_loader.index}/{data_loader.total}] Time: {time.time() - start_time:.2f}s', 'yellow')

            self.env.clean_up()
            # tick to ensure that all destroy commands are executed
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')

            # save video
            if self.need_video_render:
                self.logger.video_render.save_video(self.current_map)
            
            self.logger.log('>> ' + '-' * 40)

        # save global statistics
        self.statistics_manager.compute_global_statistics()
        self.statistics_manager.validate_and_write_statistics()
        self.logger.log(">> Finishing all the evaluation, saving the statistics", color='yellow')

    def collect_hv_init(self, data_loader: EvalDataLoader):
        while len(data_loader) > 0:
            sampled_scenario_configs = data_loader.sampler()
            self.build_env(sampled_scenario_configs)

            start_time = time.time()
            _ego_obs_list, _CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            collected = 0
            two_positive = 0
            for info in info_list:
                hv_info = info.get('high_value_init') or {}
                if hv_info.get('collect_only', False):
                    collected += 1
                    two_positive += int(hv_info.get('num_interactive_labels', 0) >= hv_info.get('target_count', 2))

            self.logger.log(
                f'>> HV init collect progress: [{data_loader.index}/{data_loader.total}] '
                f'collected={collected} two_positive={two_positive} time={time.time() - start_time:.2f}s',
                'yellow',
            )

            self.env.clean_up()
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')
            self.logger.log('>> ' + '-' * 40)

    def collect_hv_generation(self, data_loader: EvalDataLoader):
        dump_dir = self.config.get('hv_generation_dump_dir') or self.config.get('hv_dump_dir')
        if not dump_dir:
            raise ValueError("collect_hv_generation requires --hv_generation_dump_dir or --hv_dump_dir")
        sample_interval_s = float(self.config.get('hv_generation_sample_interval_s', 1.0))
        sample_steps = max(1, int(round(sample_interval_s * self.frame_rate / max(int(self.config.get('frame_skip', 1)), 1))))
        max_frames_per_episode = int(self.config.get('hv_generation_max_frames_per_episode', 200))
        max_snapshots_per_episode = int(self.config.get('hv_generation_max_snapshots_per_episode', 1000000))
        min_forward_positive = int(self.config.get('hv_generation_collect_min_forward_rift_positive', 0))
        if min_forward_positive < 0:
            raise ValueError("--hv_generation_collect_min_forward_rift_positive must be non-negative")

        while len(data_loader) > 0:
            sampled_scenario_configs = data_loader.sampler()
            self.build_env(sampled_scenario_configs)

            start_time = time.time()
            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            num_dumped = 0
            num_positive = 0
            num_forward_positive = 0
            num_forward_rejected = 0
            while not self.env.all_scenario_done():
                for env_id, current_env in enumerate(self.env.env_list):
                    if self.env.finished_env[env_id]:
                        continue
                    if current_env.time_step % sample_steps != 0:
                        continue
                    if current_env.time_step > max_frames_per_episode:
                        continue
                    if num_dumped >= max_snapshots_per_episode:
                        continue
                    snapshot = current_env.build_high_value_generation_snapshot(include_rift_labels=True)
                    snapshot['collection_conditions']={
                        'seed':self.config.get('seed'),'traffic_intensity':
                            self.config.get('hv_collection_traffic_intensity') if self.config.get('hv_collection_traffic_intensity') is not None
                            else float(self.cbv_recog_policy.traffic_intensity),
                        'traffic_intensity_override':self.config.get('hv_collection_traffic_intensity'),
                        'sample_interval_s':sample_interval_s,
                        'zero_saved_forward_positive_is_not_proven_infeasible':True,
                    }
                    forward_count = int((snapshot.get("diagnostics") or {}).get("num_forward_rift_positive_agents", 0))
                    if forward_count < min_forward_positive:
                        num_forward_rejected += 1
                        continue
                    path = current_env.high_value_snapshotter.dump(
                        snapshot,
                        output_dir=dump_dir,
                        prefix=f"route{current_env.index}_env{env_id}",
                    )
                    num_dumped += 1
                    num_positive += int(snapshot["diagnostics"]["num_rift_positive_agents"] >= 2)
                    num_forward_positive += int(forward_count >= 2)
                    self.logger.log(f">> HV generation snapshot dumped: {path}", color="green")

                if max_frames_per_episode >= 0:
                    active_steps = [
                        current_env.time_step
                        for env_id, current_env in enumerate(self.env.env_list)
                        if not self.env.finished_env[env_id]
                    ]
                    if active_steps and min(active_steps) > max_frames_per_episode:
                        break

                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=True)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=True)
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])

            self.logger.log(
                f'>> HV generation collect progress: [{data_loader.index}/{data_loader.total}] '
                f'dumped={num_dumped} two_positive={num_positive} '
                f'two_forward_positive={num_forward_positive} forward_rejected={num_forward_rejected} '
                f'time={time.time() - start_time:.2f}s',
                'yellow',
            )

            self.env.clean_up()
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')
            self.logger.log('>> ' + '-' * 40)

    def collect_hv_future_outcome(self, data_loader: EvalDataLoader):
        """Collect current-time forward-HV anchors and observation-only futures.

        The sidecar is intentionally separate from V4-A JSONL construction:
        an anchor stores the current raw snapshot, while its future outcomes
        remain actor-ID keyed supervision metadata for a later critic study.
        """
        from rift.cbv.initialization.hv_future_outcome import FutureOutcomeCollector

        output_dir = self.config.get('hv_future_outcome_dir')
        if not output_dir:
            raise ValueError("collect_hv_future_outcome requires --hv_future_outcome_dir")
        horizon_s = float(self.config.get('hv_future_outcome_horizon_s', 5.0))
        anchor_interval_s = float(self.config.get('hv_future_outcome_anchor_interval_s', 1.0))
        observe_interval = max(1, int(self.config.get('hv_future_outcome_observe_interval_steps', 1)))
        min_forward = int(self.config.get('hv_future_outcome_min_forward_rift_positive', 2))
        max_anchor_frames = int(self.config.get('hv_generation_max_frames_per_episode', 200))
        max_anchors = int(self.config.get('hv_generation_max_snapshots_per_episode', 1000000))
        if horizon_s <= 0.0 or anchor_interval_s <= 0.0 or min_forward < 2:
            raise ValueError("future-outcome collection requires positive horizon/interval and at least two forward positives")
        anchor_steps = max(1, int(round(anchor_interval_s * self.frame_rate / max(int(self.config.get('frame_skip', 1)), 1))))
        horizon_steps = int(math.ceil(horizon_s * self.frame_rate / max(int(self.config.get('frame_skip', 1)), 1)))

        while len(data_loader) > 0:
            sampled_scenario_configs = data_loader.sampler()
            self.build_env(sampled_scenario_configs)
            started = time.time()
            ego_obs_list, cbv_obs_list, info_list = self.env.reset(sampled_scenario_configs)
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)
            collectors = {}
            anchors_dumped = 0
            anchors_rejected = 0
            anchored_actor_labels = 0
            while not self.env.all_scenario_done():
                for env_id, current_env in enumerate(self.env.env_list):
                    if self.env.finished_env[env_id]:
                        continue
                    if current_env.time_step > max_anchor_frames or current_env.time_step % anchor_steps != 0:
                        continue
                    if anchors_dumped >= max_anchors:
                        continue
                    snapshot = current_env.build_high_value_generation_snapshot(include_rift_labels=True)
                    forward_count = int((snapshot.get('diagnostics') or {}).get('num_forward_rift_positive_agents', 0))
                    if forward_count < min_forward:
                        anchors_rejected += 1
                        continue
                    key = (int(current_env.index), int(env_id), int(data_loader.index))
                    collector = collectors.get(key)
                    if collector is None:
                        collector = FutureOutcomeCollector(
                            output_dir=Path(output_dir) / f"route{current_env.index}_env{env_id}_episode{data_loader.index}",
                            horizon_s=horizon_s, frame_rate=self.frame_rate,
                            fov=float(self.config.get('hv_sd_fov', 64.0)),
                            forward_min_x=float(self.config.get('hv_generation_forward_min_x', 0.0)),
                            target_fov_margin=float(self.config.get('hv_generation_target_fov_margin', 2.0)),
                        )
                        collectors[key] = collector
                    anchor_path = current_env.high_value_snapshotter.dump(
                        snapshot, output_dir=str(collector.output_dir / 'anchors'),
                        prefix=f"route{current_env.index}_env{env_id}",
                    )
                    anchored_actor_labels += collector.anchor(snapshot, anchor_path)
                    anchors_dumped += 1

                ego_actions = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=True)
                cbv_actions = self.cbv_policy.get_action(cbv_obs_list, info_list, deterministic=True)
                data = self.env.step(ego_actions, cbv_actions, ego_obs_list, cbv_obs_list, info_list)
                info_list = copy.deepcopy(data['transition_info'])
                ego_obs_list = copy.deepcopy(data['ego_transition_obs'])
                cbv_obs_list = copy.deepcopy(data['CBVs_transition_obs'])

                for env_id, current_env in enumerate(self.env.env_list):
                    key = (int(current_env.index), int(env_id), int(data_loader.index))
                    collector = collectors.get(key)
                    if collector is None or self.env.finished_env[env_id]:
                        continue
                    if current_env.time_step % observe_interval != 0:
                        continue
                    observation = current_env.build_high_value_generation_snapshot(include_rift_labels=True)
                    collector.observe(observation, active_cbv_ids=current_env.CBVs.keys(), cbv_collisions=current_env.CBVs_collision)

                # No new anchors are taken beyond max_anchor_frames, but we
                # continue for one full horizon so every final anchor can be
                # closed unless the scenario itself terminates.
                active_steps = [env.time_step for index, env in enumerate(self.env.env_list) if not self.env.finished_env[index]]
                if active_steps and min(active_steps) > max_anchor_frames + horizon_steps:
                    break
            for collector in collectors.values():
                collector.finalize_episode()
            self.logger.log(
                f'>> HV future-outcome progress: [{data_loader.index}/{data_loader.total}] '
                f'anchors={anchors_dumped} anchor_actor_labels={anchored_actor_labels} '
                f'forward_rejected={anchors_rejected} collectors={len(collectors)} '
                f'time={time.time() - started:.2f}s', 'yellow',
            )
            self.env.clean_up()
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')
            self.logger.log('>> ' + '-' * 40)

    def collect_data(self, data_loader: TrainDataLoader, buffer: CollectBuffer):

        while len(data_loader) > 0:
            # simulate multiple scenarios in parallel (usually 2 scenarios)
            sampled_scenario_configs = data_loader.sampler()
            
            # build the env
            self.build_env(sampled_scenario_configs)

            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            # start loop
            while not self.env.all_scenario_done():
                # get action from AV policy and CBV policy (assume using one batch)
                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=False)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=False)

                # apply action to env and get obs
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)

                # store to the replay buffer
                buffer.store(data_dict)

                # for transition
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])

            self.logger.log(f'>> dataset length: {len(buffer)}')
            
            # end up environment
            self.env.clean_up()
            # tick to ensure that all destroy commands are executed
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')
            self.logger.log('>> ' + '-' * 40)

        # save the data
        self.logger.log(">> Start saving the offline data", 'yellow')
        buffer.save_data()
        self.logger.log(">> Successfully saved the offline data", 'yellow')
        self.logger.log('>> ' + '-' * 40)

    def run(self):
        # get scenario data of different maps, and cluster config according to the town
        config_by_map, total_routes = ScenarioDataParser.scenario_parse(self.config, self.logger)
        # init statistics manager
        self.statistics_manager = StatisticsManager(self.logger.output_dir) if self.mode == 'eval' else None
        # init render
        self.logger.init_video_render(self.env_params) if self.need_video_render else None

        # run with different modes
        if self.mode == 'eval':
            # build eval data loader
            data_loader = EvalDataLoader(config_by_map, self.num_scenario, self.logger)
            # evaluation resume
            self.eval_resume(data_loader)  
            # load the model
            self.ego_policy.load_model()
            self.cbv_policy.load_model()
            self.cbv_recog_policy.load_model()
            self.ego_policy.set_mode('eval')
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.set_mode('eval')
            # start evaluation
            self.eval(data_loader)

        elif self.mode == 'train_ego':
            # build ego data loader
            data_loader = TrainDataLoader(config_by_map, self.num_scenario, self.logger)
            # ego training resume
            self.train_resume(self.ego_policy, data_loader)
            # build ego buffer
            buffer = EgoRolloutBuffer(self.num_scenario, self.mode, self.ego_config, self.logger)
            self.ego_policy.set_buffer(buffer, total_routes)
            # load model
            self.cbv_policy.load_model()
            self.cbv_recog_policy.load_model()
            self.ego_policy.set_mode('train')
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.set_mode('eval')
            # start training ego
            self.train_ego(data_loader, buffer)

        elif self.mode == 'train_cbv':
            # cbv data loader
            data_loader = TrainDataLoader(config_by_map, self.num_scenario, self.logger)
            # cbv training resume
            self.train_resume(self.cbv_policy, data_loader)
            # cbv buffer
            buffer = CBVRolloutBuffer(self.num_scenario, self.mode, self.cbv_config, self.logger)
            self.cbv_policy.set_buffer(buffer, total_routes)
            # load model
            self.ego_policy.load_model()
            self.cbv_recog_policy.load_model()
            self.ego_policy.set_mode('eval')
            self.cbv_policy.set_mode('train')
            self.cbv_recog_policy.set_mode('train')
            # start training the CBV
            self.train_cbv(data_loader, buffer)

        elif self.mode == 'collect_data':
            # collect data loader
            data_loader = TrainDataLoader(config_by_map, self.num_scenario, self.logger)
            # data collecting resume
            exist, file_name = self.data_resume()
            # general buffer for both agent and scenario
            buffer = CollectBuffer(self.num_scenario, self.mode, self.collect_data_config, self.logger)
            buffer.set_file_name(file_name)
            # load model
            self.ego_policy.load_model()
            self.ego_policy.set_mode('eval')
            self.cbv_policy.load_model()
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.load_model()
            self.cbv_recog_policy.set_mode('eval')
            # start collecting the dataset
            self.collect_data(data_loader, buffer) if not exist else None

        elif self.mode == 'collect_hv_init':
            data_loader = EvalDataLoader(config_by_map, self.num_scenario, self.logger)
            self.ego_policy.set_mode('eval')
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.load_model()
            self.cbv_recog_policy.set_mode('eval')
            self.collect_hv_init(data_loader)

        elif self.mode == 'collect_hv_generation':
            data_loader = EvalDataLoader(config_by_map, self.num_scenario, self.logger)
            self.ego_policy.load_model()
            self.cbv_policy.load_model()
            self.cbv_recog_policy.load_model()
            self.ego_policy.set_mode('eval')
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.set_mode('eval')
            self.collect_hv_generation(data_loader)

        elif self.mode == 'collect_hv_future_outcome':
            # This is an observational sidecar collection mode.  It uses the
            # normal recognizer to record later lifecycle outcomes, but it
            # does not enable any high-value generator or edit actor states.
            data_loader = EvalDataLoader(config_by_map, self.num_scenario, self.logger)
            self.ego_policy.load_model()
            self.cbv_policy.load_model()
            self.cbv_recog_policy.load_model()
            self.ego_policy.set_mode('eval')
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.set_mode('eval')
            self.collect_hv_future_outcome(data_loader)

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def build_env(self, sampled_configs: List[RouteScenarioConfiguration]):
        sampled_maps = {config.town for config in sampled_configs if config.town is not None}
        assert len(sampled_maps) == 1, 'Maps across sampled configs should be the same'
        sampled_map = sampled_maps.pop()
        # set the traffic flow related seed
        self._set_traffic_random_seed(sampled_configs)

        if self.current_map != sampled_map:
            self.current_map = sampled_map  # record the current running map name
            self._init_world(self.current_map)

            # create scenarios within the vectorized wrapper
            self.env = VectorWrapper(
                self.env_params,
                self.statistics_manager,
                self.config,
                self.world,
                self.ego_policy,
                self.cbv_policy,
                self.cbv_recog_policy,
                self.logger,
            )
            self.logger.log(">> Finish scenario initialization")

    def eval_resume(self, data_loader: EvalDataLoader):
        if self.resume:
            resume = data_loader.validate_and_resume()
        else:
            resume = False

        if resume:
            self.logger.log(f'>> Evaluating from {data_loader.index}', color='yellow')
            self.statistics_manager.add_file_records()
        else:
            self.logger.log('>> Evaluating from scratch', color='yellow')
            self.statistics_manager.clear_records()
        
        self.statistics_manager.save_progress(data_loader.index, data_loader.total)
        self.statistics_manager.write_statistics()

    def train_resume(self, policy, data_loader: TrainDataLoader):
        # load policy model
        policy.load_model(self.resume)
        continue_episode = policy.continue_episode
        data_loader.validate_and_resume(continue_episode)

    def data_resume(self) -> Tuple[bool, str]:
        """
        Check if the data file exists, and create the directory if it doesn't exist.

        Returns:
            tuple[bool, str]: A tuple containing a boolean indicating existence
                            and the file path as a string.
        """
        # Construct the file path using pathlib for cleaner syntax
        file_path = Path(self.data_path)
        file_name = file_path / f"{self.ego_policy_name}_{self.cbv_policy_name}.hdf5"

        # Check existence and create directory if necessary
        if file_name.is_file():
            self.logger.log(f'>> exist data on {file_path}', color='red')
            return True, str(file_name)
        else:
            file_path.mkdir(parents=True, exist_ok=True)
            return False, str(file_name)

    def _set_traffic_random_seed(self, sampled_configs: List[RouteScenarioConfiguration]):
        sampled_index = sampled_configs[0].index
        # the traffic random seed related to the sampled config index, the mode
        traffic_random_seed = MODE_SEED[self.mode] + sampled_index
        CarlaDataProvider.set_traffic_random_seed(traffic_random_seed)

    def close(self):
        # save the rest video data of the error running
        if (
            self.need_video_render
            and getattr(self, "logger", None) is not None
            and getattr(self.logger, "video_render", None) is not None
            and self.current_map is not None
        ):
            self.logger.video_render.save_video(self.current_map)
        # close the unfinished env
        if self.env:
            self.env.clean_up()
