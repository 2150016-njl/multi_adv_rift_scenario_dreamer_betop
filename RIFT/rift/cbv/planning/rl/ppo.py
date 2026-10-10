#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : ppo.py
@Date    : 2023/10/4
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import wandb
import numpy as np
import torch
import torch.nn as nn
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.util.logger import Logger
from rift.util.torch_util import CUDA, CPU
from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.gym_carla.utils.net import ActorPPO, CriticPPO


class PPO(CBVBasePolicy):
    name = 'ppo'
    type = 'rl'

    def __init__(self, config, logger: Logger):
        super(PPO, self).__init__(config, logger)

        self.continue_episode = 0
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.gamma = config['gamma']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.seed = config['seed']
        self.pretrain_seed = config['pretrain_seed']
        self.train_repeat_times = config['train_repeat_times']

        self.state_dim = config['CBV_obs_dim']
        self.action_dim = config['CBV_action_dim']
        self.clip_epsilon = config['clip_epsilon']
        self.batch_size = config['batch_size']
        self.lambda_gae_adv = config['lambda_gae_adv']
        self.lambda_entropy = config['lambda_entropy']
        self.lambda_entropy = CUDA(torch.tensor(self.lambda_entropy, dtype=torch.float32))
        self.dims = config['dims']

        self.model_type = config['model_type']
        self.cbv_recog = config['cbv_recog']
        self.model_path = Path(config['ROOT_DIR']) / config['model_path']

        self.wandb_config = self.config['wandb']
        self.wandb_initialized = False

        if config['mode'] == 'train_cbv':
            self.load_agent_info = config['ego_policy'] + '-' + str(self.cbv_recog) + '-seed' + str(self.seed)
        else:
            self.load_agent_info = config['pretrain_ego'] + '-' + str(self.cbv_recog) + '-seed' + str(self.pretrain_seed)

        self.save_agent_info = config['ego_policy'] + '-' + str(self.cbv_recog) + '-seed' + str(self.seed)

        self.policy = CUDA(ActorPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)  # trick about eps
        self.value = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)  # trick about eps
        self.value_criterion = nn.SmoothL1Loss()
        # render
        self._render = config['need_video_render']
        if self._render:
            self.reset_render_data()

    def reset_render_data(self):
        self._render_data = {env_id: {
            "ego_states": {},
            "CBV_states": {},
            "nearby_agents_states": {},
        } for env_id in range(self.num_scenario)}

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.value.train()
        elif mode == 'eval':
            self.policy.eval()
            self.value.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def lr_decay(self, e_i):
        lr_policy_now = self.policy_lr * (1 - e_i / self.total_routes)
        lr_value_now = self.value_lr * (1 - e_i / self.total_routes)
        for p in self.policy_optim.param_groups:
            p['lr'] = lr_policy_now
        for p in self.value_optim.param_groups:
            p['lr'] = lr_value_now

    @staticmethod
    def CBVs_obs_process(CBVs_obs_list, infos):
        CBVs_obs_batch = []
        CBVs_id = []
        env_index = []
        for CBVs_obs, info in zip(CBVs_obs_list, infos):
            for CBV_id, CBV_obs in CBVs_obs.items():
                CBVs_obs_batch.append(CBV_obs)
                CBVs_id.append(CBV_id)
                env_index.append(info['env_id'])
        if CBVs_obs_batch:
            state_batch = np.concatenate([x.reshape(1, -1) for x in CBVs_obs_batch], axis=0)
        else:
            state_batch = None

        return state_batch, CBVs_id, env_index

    def get_action(self, CBVs_obs_list, infos, deterministic=False) -> Dict[str, List[Dict[Any, Any]]]:
        state, CBVs_id, env_index = self.CBVs_obs_process(CBVs_obs_list, infos)
        CBVs_action = {env_id:{} for env_id in range(self.num_scenario)}
        CBVs_action_log_prob = {env_id:{} for env_id in range(self.num_scenario)}
        if state is not None:
            state_tensor = CUDA(torch.FloatTensor(state))
            if deterministic:
                action = self.policy(state_tensor)
                for i, (CBV_id, env_id) in enumerate(zip(CBVs_id, env_index)):
                    CBVs_action[env_id][CBV_id] = CPU(action[i])
                    CBVs_action_log_prob[env_id][CBV_id] = None
            else:
                action, log_prob = self.policy.get_action(state_tensor)
                for i, (CBV_id, env_id) in enumerate(zip(CBVs_id, env_index)):
                    CBVs_action[env_id][CBV_id] = CPU(action[i])
                    CBVs_action_log_prob[env_id][CBV_id] = CPU(log_prob[i])
        data = {
            'CBVs_actions': CBVs_action,
            'CBVs_actions_log_prob': CBVs_action_log_prob
        }

        if self._render:
            for CBVs_obs, info in zip(CBVs_obs_list, infos):
                env_id = info['env_id']
                # ego states
                ego = CarlaDataProvider.get_ego_vehicle_by_env_id(env_id)
                ego_state = CarlaDataProvider.get_current_state(ego)
                self._render_data[env_id]['ego_states'][ego.id] = ego_state
                # nearby agent states
                for agent in CarlaDataProvider.get_ego_nearby_agents(ego.id):
                    agent_state = CarlaDataProvider.get_current_state(agent)
                    self._render_data[env_id]['nearby_agents_states'][agent.id] = agent_state      
                # CBV_ids
                for CBV_id in CBVs_obs.keys():
                    CBV_state = CarlaDataProvider.get_current_state(CarlaDataProvider.get_actor_by_id(CBV_id))
                    self._render_data[env_id]['CBV_states'][CBV_id] = CBV_state

        return data
    
    def set_buffer(self, buffer: CBVRolloutBuffer, total_routes):
        self.buffer: CBVRolloutBuffer = buffer
        self.total_routes = total_routes
        resume = 'auto' if self.config['resume'] else 'never'
        # init the wandb logger
        if not self.wandb_initialized:
            wandb.init(
                project=self.wandb_config['project'],
                group=self.wandb_config['base_group'] + '-' + self.logger.output_dir.name,
                name=self.logger.output_dir.name,
                config=self.config,
                resume=resume,
                dir=self.logger.output_dir,
                mode=self.wandb_config['mode']
            )
            self.wandb_initialized = True

    def log_episode_reward(self, episode_reward, episode):
        wandb.log({'CBV episode reward': episode_reward}, step=episode)
    
    def get_render_data(self, env_id):
        return self._render_data[env_id]

    def get_advantages_GAE(self, rewards, undones, values, next_values, unterminated):
        """
            unterminated: if the CBV collide with an object, then it is terminated
            undone: if the CBV is stuck or collide or max step will cause 'done'
            https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentPPO.py
        """
        advantages = torch.empty_like(values)  # advantage value

        horizon_len = rewards.shape[0]

        advantage = torch.zeros_like(values[0])  # last advantage value by GAE (Generalized Advantage Estimate)

        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + unterminated[t] * self.gamma * next_values[t] - values[t]
            advantages[t] = advantage = delta + undones[t] * self.gamma * self.lambda_gae_adv * advantage
        return advantages

    def train(self, e_i):
        """
            from https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentPPO.py
        """

        with torch.no_grad():
            # learning rate decay
            self.lr_decay(e_i)  # add the learning rate decay

            batch = self.buffer.get_all_np_data()

            states = CUDA(torch.FloatTensor(batch['CBVs_obs']))
            next_states = CUDA(torch.FloatTensor(batch['CBVs_next_obs']))
            actions = CUDA(torch.FloatTensor(batch['CBVs_actions']))
            log_probs = CUDA(torch.FloatTensor(batch['CBVs_actions_log_prob']))
            rewards = CUDA(torch.FloatTensor(batch['CBVs_reward']))
            undones = CUDA(torch.FloatTensor(1.0-batch['CBVs_done']))
            unterminated = CUDA(torch.FloatTensor(1.0-batch['CBVs_terminated']))

            buffer_size = states.shape[0]

            values = self.value(states)
            next_values = self.value(next_states)

            advantages = self.get_advantages_GAE(rewards, undones, values, next_values, unterminated)
            reward_sums = advantages + values
            del rewards, undones, values, next_values, unterminated

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)

        # start to train, use gradient descent without batch size
        update_times = int(buffer_size * self.train_repeat_times / self.batch_size)
        assert update_times >= 1

        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            log_prob = log_probs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            # update value function
            value = self.value(state)
            value_loss = self.value_criterion(value, reward_sum)  # the value criterion is SmoothL1Loss() instead of MSE
            self.value_optim.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_optim.step()

            # update policy
            new_log_prob, entropy = self.policy.get_logprob_entropy(state, action)

            ratio = (new_log_prob - log_prob.detach()).exp()
            L1 = advantage * ratio
            L2 = advantage * torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon)
            surrogate = torch.min(L1, L2).mean()
            actor_loss = -(surrogate + entropy.mean() * self.lambda_entropy)
            self.policy_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optim.step()

        # logging
        wandb.log({
            "CBV value loss": value_loss.item(),
            "CBV actor entropy":entropy.mean().item(),
            "CBV actor loss": actor_loss.item()
        }, step=e_i)

        # reset buffer
        self.buffer.reset_buffer()

    def save_model(self, episode):
        save_dir = self.model_path / self.save_agent_info
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"model_{self.name}_{self.model_type}_{episode:04}.torch"

        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'value_optim': self.value_optim.state_dict(),
            'episode': episode
        }, filepath)
        
        self.logger.log(f'>> Saving CBV policy {self.name} model to {os.path.basename(filepath)}', 'yellow')


    def load_model(self, resume=True):
        load_dir = self.model_path / self.load_agent_info
        episode_files = list(load_dir.glob(f"model_{self.name}_{self.model_type}_*.torch"))
        
        if resume and episode_files:
            episode = max(int(file.stem.split('.')[-1]) for file in episode_files)
            filepath = load_dir / f"model_{self.name}_{self.model_type}_{episode:04}.torch"
            self.logger.log(f">> Loading {self.name} model from {filepath}", 'yellow')

            checkpoint = torch.load(filepath)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.policy_optim.load_state_dict(checkpoint['policy_optim'])
            self.value_optim.load_state_dict(checkpoint['value_optim'])
            self.continue_episode = episode
        else:
            if not resume:
                self.logger.log(f">> Clearning all the data in {load_dir}, training from scratch", 'red')
                for file in episode_files:
                    file.unlink()  # remove all the model files
                self._initialize_model()
            else:
                filepath = self.model_path / f"model_{self.name}_cbv_pretrain.torch"
                self.logger.log(f">> Loading {self.name} pretrain model from {filepath}", 'yellow')
                checkpoint = torch.load(filepath)
                # load the pretrain model
                self.policy.load_state_dict(checkpoint['policy'])
                self.value.load_state_dict(checkpoint['value'])
                # reinitialize the optimizer
                self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)
                self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)
            
            self.continue_episode = 0

    def _initialize_model(self):
        self.policy = CUDA(ActorPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)
        self.value = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)

    def finish(self):
        if self.wandb_initialized:
            wandb.finish()