#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : run.py
@Date    : 2023/10/4
"""

import os
import sys
import socket
import time
import traceback
import os.path as osp
import atexit
import torch
import subprocess
import signal

import carla
from rift.scenario.tools.exception import SpawnRuntimeError
from rift.util.run_util import load_config
from rift.util.torch_util import set_seed, set_torch_variable
from rift.carla_runner import CarlaRunner

MODE_TO_ROUTES = {
    'eval': 'rift/scenario/route/drivetransformer_bench2drive_dev10.xml',
    # 'eval': 'rift/scenario/route/bench2drive220.xml',
    'train_ego': 'rift/scenario/route/bench2drive220.xml',
    'train_cbv': 'rift/scenario/route/bench2drive220.xml',
    'collect_data': 'rift/scenario/route/bench2drive220.xml',
    'collect_hv_init': 'rift/scenario/route/bench2drive220.xml',
    'collect_hv_generation': 'rift/scenario/route/bench2drive220.xml',
    'collect_hv_future_outcome': 'rift/scenario/route/bench2drive220.xml',
}


def cleanup(server_pid):
    try:
        os.killpg(os.getpgid(server_pid), signal.SIGKILL)
        print(f">> Successfully sent SIGKILL to process group {server_pid}")
        time.sleep(1)
        os.kill(server_pid, 0)
    except ProcessLookupError:
        print(f">> CARLA server with PID {server_pid} has been successfully terminated.")
    except Exception as e:
        print(f">> Failed to terminate CARLA server with PID {server_pid}: {e}")


def setup_simulation(args, carla_device_id):
    """
    Prepares the simulation by getting the client, and setting up the world and traffic manager settings
    """
    carla_path = os.environ["CARLA_ROOT"]
    args.port = find_free_port(args.base_port + 150 * carla_device_id)

    # the command to start the carla server
    cmd1 = f"{os.path.join(carla_path, 'CarlaUE4.sh')} -nosound -carla-rpc-port={args.port}"
    if not args.render:
        cmd1 += " -nullrhi"
        print(">> Running Carla without GPU!", flush=True)
    else:
        cmd1 += f" -RenderOffScreen -graphicsadapter={carla_device_id}"
    # user the subprocess.Popen to start the server in the background
    server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
    print(">> Starting Carla:", cmd1, flush=True)
    # Register cleanup operations when exiting
    atexit.register(cleanup, server.pid)
    time.sleep(5)
        
    attempts = 0
    num_max_restarts = 20
    while attempts < num_max_restarts:
        try:
            client = carla.Client(args.host, args.port)
            if args.timeout:
                client_timeout = args.timeout
            client.set_timeout(client_timeout)

            settings = carla.WorldSettings(
                synchronous_mode = True,
                fixed_delta_seconds = 1.0 / args.frame_rate,
                deterministic_ragdolls = True,
                spectator_as_ego = False,
                no_rendering_mode = not args.render,  # only render the scene when needed
            )
            client.get_world().apply_settings(settings)
            print(f">> load world success, attempts={attempts}", flush=True)
            break
        except Exception as e:
            print(f">> load world failed , attempts={attempts}", flush=True)
            print(e, flush=True)
            attempts += 1
            time.sleep(2)

    attempts = 0
    num_max_restarts = 40
    while attempts < num_max_restarts:
        try:
            args.tm_port = find_free_port(args.base_tm_port + 150 * carla_device_id)
            traffic_manager = client.get_trafficmanager(args.tm_port)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_hybrid_physics_mode(True)
            print(f">> traffic_manager port:{args.tm_port} init success, try_time={attempts}", flush=True)
            print('>> ' + '-' * 40, flush=True)
            break
        except Exception as e:
            print(f">> traffic_manager init fail, try_time={attempts}", flush=True)
            print(e, flush=True)
            attempts += 1
            time.sleep(5)
    return client, traffic_manager, args


def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_ego', 'train_cbv', 'eval', 'collect_data', 'collect_hv_init', 'collect_hv_generation', 'collect_hv_future_outcome'])
    # simulation setup
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--repetitions', '-rep', type=int, default=2, help='Number of repetitions per route.')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--pretrain_seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--routes', type=str, default=None,
                        help='Optional route XML override. Defaults to the route file associated with --mode.')
    parser.add_argument('--routes_subset', type=str, default='',
                        help='Optional route ids/subsets accepted by RouteParser, e.g. "0", "0-9", "0,4,8".')
    # agent-related
    parser.add_argument('--cbv_recog', '-recog', type=str, default='rule', choices=['rule', 'attention'])
    parser.add_argument('--ego_cfg', type=str, default='expert.yaml')
    parser.add_argument('--pretrain_ego', '-pre_ego', type=str, default='pdm_lite', choices=['pdm_lite', 'plant', 'expert'])
    parser.add_argument('--pretrain_cbv', '-pre_cbv', type=str, default='standard', choices=['ppo', 'standard', 'frea', 'fppo_rs'])
    parser.add_argument('--cbv_cfg', type=str, default='standard_eval.yaml')
    parser.add_argument('--collect_data_cfg', type=str, default='collect_data.yaml')
    # simulation-related
    parser.add_argument('--spectator', '-sp', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--save_bev', action='store_true', default=False, help='Save offline BEV/trajectory videos without CARLA camera rendering.')
    parser.add_argument('--no_resume', action='store_false', dest='resume', help='Do not resume execution (default: resume=True)')
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--host', default='localhost', help='IP of the host server (default: localhost)')
    parser.add_argument('--base_port', type=int, default=30000, help='port to communicate with carla')
    parser.add_argument('--base_tm_port', type=int, default=50000, help='traffic manager port')
    parser.add_argument('--frame_rate', '-fps', type=float, default=10.0)
    parser.add_argument('--timeout', default=300.0, type=float, help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--rift_hd_map_dir', type=str, default=None,
                        help='Directory containing RIFT *_HD_map.npz files.')
    parser.add_argument('--rift_speed_limits_dir', type=str, default=None,
                        help='Directory containing RIFT *_speed_limits.npy files.')
    # high-value initial-state generation
    parser.add_argument('--high_value_init', action='store_true', default=False)
    parser.add_argument('--hv_init_generator_entry', type=str, default=None)
    parser.add_argument('--hv_init_checkpoint', type=str, default=None)
    parser.add_argument('--hv_init_device', type=str, default='cuda')
    parser.add_argument('--hv_init_target_count', type=int, default=2)
    parser.add_argument('--hv_init_apply_top_k', type=int, default=2,
                        help='Apply generated initial states to top-k actors; first target_count become CBVs.')
    parser.add_argument('--hv_init_max_attempts', type=int, default=1)
    parser.add_argument('--hv_init_validation_mode', type=str, default='audit', choices=['audit', 'strict'])
    parser.add_argument('--no_hv_init_require_target_mask', action='store_false', dest='hv_init_require_target_mask')
    parser.set_defaults(hv_init_require_target_mask=True)
    parser.add_argument('--no_hv_init_fail_on_invalid', action='store_false', dest='hv_init_fail_on_invalid')
    parser.set_defaults(hv_init_fail_on_invalid=True)
    parser.add_argument('--no_hv_init_lock_selected_cbvs', action='store_false', dest='hv_init_lock_selected_cbvs')
    parser.set_defaults(hv_init_lock_selected_cbvs=True)
    parser.add_argument('--hv_init_reject_extra_rift_positive', action='store_true', default=False)
    parser.add_argument('--no_hv_init_tick_after_apply', action='store_false', dest='hv_init_tick_after_apply')
    parser.set_defaults(hv_init_tick_after_apply=True)
    parser.add_argument('--hv_init_dump_dir', type=str, default=None)
    parser.add_argument('--no_hv_init_save_viz', action='store_false', dest='hv_init_save_viz')
    parser.set_defaults(hv_init_save_viz=True)
    parser.add_argument('--hv_init_debug_generator', action='store_true', default=False,
                        help='Smoke-test only: selects actors from existing RIFT labels and reapplies their current states.')
    parser.add_argument('--hv_init_collect_only', action='store_true', default=False,
                        help='Collect RIFT-native high-value labels without generating, modifying actors, or registering CBVs.')
    parser.add_argument('--hv_init_allow_untrained_generator', action='store_true', default=False)
    parser.add_argument('--hv_init_min_selected_pair_distance', type=float, default=0.0)
    parser.add_argument('--hv_init_max_abs_xy_delta', type=float, default=30.0)
    parser.add_argument('--hv_init_max_abs_yaw_delta', type=float, default=180.0)
    parser.add_argument('--hv_init_max_speed_mps', type=float, default=40.0)
    parser.add_argument('--hv_init_max_route_points', type=int, default=160)
    parser.add_argument('--hv_init_map_radius', type=float, default=64.0)
    parser.add_argument('--hv_init_max_map_lanes', type=int, default=120)
    parser.add_argument('--hv_model_backbone', type=str, default='autoencoder',
                        choices=['autoencoder', 'adapter_only'],
                        help='Use Scenario Dreamer autoencoder latents by default; adapter_only is for ablations/debug.')
    parser.add_argument('--hv_model_autoencoder_checkpoint', type=str, default=None)
    parser.add_argument('--hv_model_cfg_path', type=str, default=None)
    parser.add_argument('--hv_model_ldm_checkpoint', type=str, default=None)
    parser.add_argument('--hv_model_ldm_cfg_path', type=str, default=None)
    parser.add_argument('--hv_model_state_source', type=str, default='ldm_inpainting',
                        choices=['ldm_inpainting', 'ldm_lane_conditioned', 'ldm', 'identity'])
    parser.add_argument('--hv_model_num_generation_candidates', type=int, default=8)
    parser.add_argument('--hv_model_hidden_dim', type=int, default=256)
    parser.add_argument('--hv_model_num_layers', type=int, default=3)
    parser.add_argument('--hv_model_num_heads', type=int, default=8)
    parser.add_argument('--hv_model_dropout', type=float, default=0.1)
    parser.add_argument('--hv_sd_fov', type=float, default=64.0)
    parser.add_argument('--hv_sd_max_num_lanes', type=int, default=100)
    parser.add_argument('--hv_sd_num_points_per_lane', type=int, default=20)
    parser.add_argument('--hv_sd_map_id', type=int, default=0)
    # high-value scene generation: model outputs anonymous agent states.
    parser.add_argument('--high_value_init_generation', action='store_true', default=False)
    parser.add_argument('--hv_generation_generator_entry', type=str,
                        default='high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator')
    parser.add_argument('--hv_generation_checkpoint', type=str, default=None)
    parser.add_argument('--sd_background_checkpoint', type=str, default=None,
                        help='Optional Scenario Dreamer background checkpoint for the composed online generator.')
    parser.add_argument('--sd_background_source', choices=['carla', 'scenario_dreamer'], default='carla',
                        help='Use snapshot CARLA BVs or Scenario Dreamer BVs as context for the two HV actors.')
    parser.add_argument('--sd_background_num_agents', type=int, default=6,
                        help='Number of ordinary background actors generated by Scenario Dreamer in composed mode.')
    parser.add_argument('--sd_background_replace_existing', action='store_true', default=False,
                        help='Replace nearby CARLA background actors before composed Scenario Dreamer generation.')
    parser.add_argument('--sd_background_replace_radius', type=float, default=64.0,
                        help='Ego-relative radius used by --sd_background_replace_existing.')
    parser.add_argument('--sd_background_model_only_traffic', action='store_true', default=False,
                        help='Remove every non-model vehicle at reset and after each tick; use only with composed generation.')
    parser.add_argument('--hv_generation_device', type=str, default='cuda')
    parser.add_argument('--hv_generation_mode', type=str, default='inpainting',
                        choices=['inpainting', 'sparse_inpainting', 'full_fov_inpainting', 'full_generation'])
    parser.add_argument('--hv_num_generate_agents', type=int, default=2)
    parser.add_argument('--hv_generation_forward_min_x', type=float, default=0.0,
                        help='V4-A forward target-sector lower x bound in the ego-local 64 m FOV.')
    parser.add_argument('--hv_generation_target_fov_margin', type=float, default=2.0,
                        help='V4-A forward target-sector boundary margin; must match the checkpoint contract.')
    parser.add_argument('--hv_actor_radius', type=float, default=80.0)
    parser.add_argument('--hv_map_radius', type=float, default=64.0)
    parser.add_argument('--hv_max_route_points', type=int, default=160)
    parser.add_argument('--hv_max_map_lanes', type=int, default=120)
    parser.add_argument('--hv_snapshot_map_coordinate_mode', choices=['legacy', 'right_handed'], default='legacy',
                        help='Snapshot HD-map coordinate convention. Use right_handed only with a matching V3 model.')
    parser.add_argument('--hv_generation_dump_dir', '--hv_dump_dir', dest='hv_generation_dump_dir', type=str, default=None)
    parser.add_argument('--hv_generation_lifecycle_telemetry', action='store_true', default=False,
                        help='Append observation-only generated-actor lifecycle events to the HV dump directory.')
    parser.add_argument('--hv_generation_sample_interval_s', type=float, default=1.0)
    parser.add_argument('--hv_generation_max_frames_per_episode', type=int, default=200)
    parser.add_argument('--hv_generation_max_snapshots_per_episode', type=int, default=1000000)
    parser.add_argument('--hv_generation_collect_min_forward_rift_positive', type=int, default=0,
                        help='Collection-only filter: dump a snapshot only when this many RIFT-positive actors are in the V4-A forward sector. 0 keeps all snapshots.')
    parser.add_argument('--hv_future_outcome_dir', type=str, default=None,
                        help='Sidecar output for current-anchor / future-outcome collection; required by collect_hv_future_outcome.')
    parser.add_argument('--hv_future_outcome_horizon_s', type=float, default=5.0,
                        help='Observation horizon only; future positions are never generation/spawn states.')
    parser.add_argument('--hv_future_outcome_anchor_interval_s', type=float, default=1.0)
    parser.add_argument('--hv_future_outcome_observe_interval_steps', type=int, default=1)
    parser.add_argument('--hv_future_outcome_min_forward_rift_positive', type=int, default=2,
                        help='Anchor only current snapshots with this many forward RIFT-positive actors.')
    parser.add_argument('--hv_generation_max_attempts', type=int, default=8,
                        help='Maximum LDM samples tried at reset before declaring online generation invalid.')
    parser.add_argument('--no_hv_generation_partial_context', action='store_false', dest='hv_generation_partial_context',
                        help='Disable partial/inpainting context preparation before online generation.')
    parser.set_defaults(hv_generation_partial_context=True)
    parser.add_argument('--no_hv_generation_require_hidden_slots', action='store_false', dest='hv_generation_require_hidden_slots',
                        help='Allow online generation even if fewer background actors were hidden than requested.')
    parser.set_defaults(hv_generation_require_hidden_slots=True)
    parser.add_argument('--hv_generation_hidden_slots', type=int, default=None,
                        help='Number of background actors to hide before inpainting. Defaults to hv_num_generate_agents.')
    parser.add_argument('--hv_generation_hidden_forward_min', type=float, default=0.0)
    parser.add_argument('--hv_generation_hidden_forward_max', type=float, default=64.0)
    parser.add_argument('--hv_generation_hidden_lateral_abs', type=float, default=32.0)
    parser.add_argument('--hv_generation_rolling', action='store_true', default=False,
                        help='Enable rolling high-value inpainting during route execution.')
    parser.add_argument('--hv_generation_rolling_interval_steps', type=int, default=50)
    parser.add_argument('--hv_generation_rolling_min_step', type=int, default=50)
    parser.add_argument('--hv_generation_rolling_min_rift_positive', type=int, default=None,
                        help='Skip rolling generation when current RIFT-positive actor count is already at least this value. Defaults to hv_num_generate_agents.')
    parser.add_argument('--no_hv_generation_gate_on_existing_rift_positive', action='store_false',
                        dest='hv_generation_gate_on_existing_rift_positive',
                        help='Generate even when the pre-generation RIFT recognizer already finds an interactive actor.')
    parser.set_defaults(hv_generation_gate_on_existing_rift_positive=True)
    parser.add_argument('--hv_generation_existing_rift_positive_threshold', type=int, default=2,
                        help='Skip model generation when at least this many pre-existing RIFT-positive actors are present.')
    parser.add_argument('--video_render_bounds', type=float, default=32.0,
                        help='Half-width in meters shown by saved BEV videos. Defaults to the 64m HV model FOV.')
    parser.add_argument('--video_render_map_radius', type=float, default=64.0,
                        help='Map query radius in meters for saved BEV videos.')
    parser.add_argument('--video_render_offset', type=float, default=0.0,
                        help='Forward offset in meters for saved BEV videos.')
    parser.add_argument('--hv_generation_rolling_force_update', action='store_true', default=False,
                        help='Experimental: at each rolling interval, remove active generated HV actors and generate/register fresh HV-CBVs.')
    parser.add_argument('--hv_generation_lifetime_radius', type=float, default=None,
                        help='Remove generated agents once they are farther than this radius from ego. Defaults to hidden_forward_max + 30.')
    parser.add_argument('--hv_autoencoder_checkpoint', type=str, default=None)
    parser.add_argument('--hv_ldm_checkpoint', type=str, default=None)
    parser.add_argument('--hv_generation_actor_model', type=str, default='vehicle.*')
    parser.add_argument('--hv_generation_actor_role', type=str, default='generated_hv')
    parser.add_argument('--no_hv_generation_autopilot', action='store_false', dest='hv_generation_autopilot')
    parser.set_defaults(hv_generation_autopilot=True)
    parser.add_argument('--hv_generation_register_cbv', action='store_true', default=False,
                        help='Enable generated-actor CBV handling. By default only normal RIFT recognition may activate them.')
    parser.add_argument('--hv_generation_register_cbv_from_audit', action='store_true', default=False,
                        help='Legacy ablation: immediately register post-spawn RIFT-positive generated actors as CBVs.')
    parser.add_argument('--hv_generation_register_cbv_count', type=int, default=None,
                        help='Maximum generated actors to register as CBVs. Defaults to hv_num_generate_agents.')
    parser.add_argument('--hv_generation_clamp_generated_states', action='store_true', default=False,
                        help='Clamp generated speed/length/width before spawning online generated actors.')
    parser.add_argument('--hv_min_generated_speed', type=float, default=0.0)
    parser.add_argument('--hv_max_generated_speed', type=float, default=30.0)
    parser.add_argument('--hv_min_generated_length', type=float, default=2.0)
    parser.add_argument('--hv_max_generated_length', type=float, default=8.0)
    parser.add_argument('--hv_min_generated_width', type=float, default=1.4)
    parser.add_argument('--hv_max_generated_width', type=float, default=3.2)
    parser.add_argument('--no_hv_generation_fail_on_invalid', action='store_false', dest='hv_generation_fail_on_invalid')
    parser.set_defaults(hv_generation_fail_on_invalid=True)
    args = parser.parse_args()
    if args.mode == 'collect_hv_init':
        args.high_value_init = True
        args.hv_init_collect_only = True
        args.hv_init_fail_on_invalid = False
        if not args.hv_init_dump_dir:
            raise ValueError("--mode collect_hv_init requires --hv_init_dump_dir")
    if args.mode == 'collect_hv_generation':
        args.high_value_init_generation = False
        if not args.hv_generation_dump_dir:
            raise ValueError("--mode collect_hv_generation requires --hv_generation_dump_dir/--hv_dump_dir")
    if args.mode == 'collect_hv_future_outcome' and not args.hv_future_outcome_dir:
        raise ValueError("--mode collect_hv_future_outcome requires --hv_future_outcome_dir")
    if args.high_value_init and args.high_value_init_generation:
        raise ValueError("--high_value_init and --high_value_init_generation are separate pipelines; enable only one")

    # set the seed and device
    carla_device_id = set_torch_variable()
    
    # setup the simulation
    client, traffic_manager, args = setup_simulation(args, carla_device_id)

    args_dict = vars(args)
    
    # get the routes file
    args_dict['routes'] = args.routes or MODE_TO_ROUTES[args.mode]
    
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    err_list = []

    configs = [args_dict]

    # load ego config
    ego_config_path = osp.join(args.ROOT_DIR, 'rift/ego/config', args.ego_cfg)
    ego_config = load_config(ego_config_path)
    ego_config.update(args_dict)
    configs.append(ego_config)

    # load cbv planning config
    cbv_config_path = osp.join(args.ROOT_DIR, 'rift/cbv/planning/config', args.cbv_cfg)
    cbv_config = load_config(cbv_config_path)
    cbv_config.update(args_dict)
    configs.append(cbv_config)

    # load CBV config
    cbv_recog_config_path = osp.join(args.ROOT_DIR, 'rift/cbv/recognition/config', f'{args.cbv_recog}.yaml')
    cbv_recog_config = load_config(cbv_recog_config_path)
    cbv_recog_config.update(args_dict)
    configs.append(cbv_recog_config)

    # load collect data config
    collect_data_config_path = osp.join(args.ROOT_DIR, 'data/config', args.collect_data_cfg)
    collect_data_config = load_config(collect_data_config_path)
    collect_data_config.update(args_dict)
    configs.append(collect_data_config)

    # create the main runner
    runner = CarlaRunner(client, traffic_manager, configs)

    # start running
    try:
        runner.run()
    except (SpawnRuntimeError, RuntimeError):
        runner.close()
        print(">> An runtime error occurred during runner.run():")
        traceback.print_exc()
        err_list.append(traceback.format_exc())
        sys.exit(99)
    except KeyboardInterrupt:
        runner.close()
        runner.world.tick()  # normal exit need to tick the world to ensure that all destroy commands are executed
        print(">> KeyboardInterrupt: exiting")
        err_list.append(traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        if "TimeoutException" in str(e):
            runner.close()
            print(">> A TimeoutException occurred during runner.run():")
            print("Caught exception", e, flush=True)
            traceback.print_exc()
            err_list.append(traceback.format_exc())
            sys.exit(99)
        else:
            runner.close()
            print(">> An error occurred during runner.run():")
            print("Caught exception", e, flush=True)
            traceback.print_exc()
            err_list.append(traceback.format_exc())
            sys.exit(1)
    
    for err in err_list:
        print(err)
    
    sys.exit(0)
