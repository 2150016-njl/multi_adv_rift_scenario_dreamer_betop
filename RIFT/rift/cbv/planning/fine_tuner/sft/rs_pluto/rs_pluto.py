#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : reinforce_pluto.py
@Date    : 2025/01/27
'''

import gc
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from rift.cbv.planning.fine_tuner.sft.sft_pluto import SFTPluto
from rift.cbv.planning.fine_tuner.training_builder import TrainingEngine, build_training_engine
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.cbv.planning.pluto.planners.ml_planner_utils import global_trajectory_to_states
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from nuplan_plugin.trajectory.interpolated_trajectory import InterpolatedTrajectory


class RewardShapingPluto(SFTPluto):
    name = 'rs_pluto'
    type = 'learnable'

    def __init__(self, config, logger):
        super().__init__(config, logger)

    def train(self, e_i):
        self.logger.log('>> Starting fine-tuning...', color='yellow')
        # rebuild the training engine for each training
        dir_path = self.model_path / self.load_agent_info

        # update the training cfg learning rate
        self.cfg.lr = max(self.initial_lr * (self.cfg.cl_lr_decay ** self.current_epoch), self.cfg.min_lr)  # decay the learning rate through close loop training

        # build the training engine
        training_engine: TrainingEngine = build_training_engine(
            cfg=self.cfg,
            dir_path=dir_path,
            carla_episode=e_i,
            torch_module_wrapper=self.train_model,  # copy the training pluto model
            buffer=self.buffer,
        )
        
        # load the current training model ckpt
        training_engine.model.load_state_dict(self.load_train_checkpoint(self.checkpoint, device_name=self.device))

        # process_buffer
        training_engine.datamodule.preprocess_buffer()

        # starting training
        training_engine.trainer.fit(model=training_engine.model, datamodule=training_engine.datamodule)

        if wandb.run is not None:
            wandb.finish()  # finish the wandb run

        # update the latest ckpt
        self.update_training_ckpt()
        
        # update inference model parameters
        self.pluto_model.load_state_dict(self.load_infer_checkpoint(self.checkpoint, device_name=self.device))

        # clean the GPU memory
        del training_engine
        gc.collect()
        torch.cuda.empty_cache()
        # reset buffer
        self.buffer.reset_buffer()
        self.logger.log('>> Finishing fine-tuning...', color='yellow')

    def get_action(self, CBVs_obs_list, infos, deterministic=False) -> Dict[str, List[Dict[Any, Any]]]:
        CBVs_action = [{} for _ in range(self.num_scenario)]
        CBVs_target_speed = [{} for _ in range(self.num_scenario)]

        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            if not CBVs_obs:
                continue

            env_id = info['env_id']
            pluto_feature_data = PlutoFeature.collate([CBV_obs['raw_pluto_feature'] for CBV_obs in CBVs_obs.values()]).to_device(self.device).data
            with torch.no_grad():
                pluto_output = self.pluto_model(pluto_feature_data)
            for index, (CBV_id, CBV_obs) in enumerate(CBVs_obs.items()):
                # get the action for the inference
                CBVs_action[env_id][CBV_id] = self._get_action(pluto_output, CBV_obs, env_id, CBV_id, index)
                CBVs_target_speed[env_id][CBV_id] = self.controllers[env_id][CBV_id].desired_speed
            
        # render part
        if self._render:
            for info in infos:
                env_id = info['env_id']
                ego = CarlaDataProvider.get_ego_vehicle_by_env_id(env_id)
                ego_state = CarlaDataProvider.get_current_state(ego)
                nearby_agents = CarlaDataProvider.get_ego_nearby_agents(ego.id)
                nearby_agents_states = {agent.id: CarlaDataProvider.get_current_state(agent) for agent in nearby_agents}

                self._render_data[env_id].update({
                    'ego_states': {ego.id: ego_state},
                    'nearby_agents_states': nearby_agents_states
                })
        
        # clean useless CBVs
        self._clean_CBVs(infos, CBVs_obs_list)

        data = {
            'CBVs_actions': CBVs_action,
        }
            
        # get the teacher info for the training
        if self.mode == 'train':
            CBVs_teacher_rewards = [{} for _ in range(self.num_scenario)]
            CBV_id_set = set()
            for info, CBVs_obs in zip(infos, CBVs_obs_list):
                env_id = info['env_id']

                CBV_id_set.update(CBVs_obs.keys())
                teacher_infos = {
                    CBV_id: self._get_teacher_infos(CBV_id)
                    for CBV_id in CBVs_obs.keys()
                }
                CBVs_teacher_rewards[env_id] = {
                    CBV_id: self._get_teacher_rewards(teacher_infos[CBV_id], CBVs_target_speed[env_id][CBV_id])
                    for CBV_id in CBVs_obs.keys()
                }
                if self._render:
                    self._render_data[env_id]["CBV_teacher_infos"] = teacher_infos
            # clean unused teacher model
            self._clean_teacher_model(CBV_id_set)
            data['CBVs_teacher_rewards'] = CBVs_teacher_rewards

        return data

    def _get_teacher_rewards(self, teacher_infos, desired_speed):
        # the delta speed between teacher target_speed and best_traj desired_speed
        teacher_reward = - abs(teacher_infos[0] - desired_speed)
        return teacher_reward
