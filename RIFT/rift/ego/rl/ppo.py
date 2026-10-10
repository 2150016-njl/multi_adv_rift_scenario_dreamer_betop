"""
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-01 16:00:55
Description: 
    Copyright (c) 2022-2023 Safebench Team

    Modified from <https://github.com/gouxiangchen/ac-ppo>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
"""

import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from fnmatch import fnmatch

import wandb

from rift.gym_carla.buffer.ego_rollout_buffer import EgoRolloutBuffer
from rift.util.logger import Logger
from rift.util.torch_util import CUDA, CPU, hidden_init
from rift.ego.base_policy import EgoBasePolicy
from rift.gym_carla.utils.net import ActorPPO, CriticPPO


class PPO(EgoBasePolicy):
    name = 'ppo'

    def __init__(self, config, logger: Logger):
        super(PPO, self).__init__(config, logger)

        self.continue_episode = 0
        self.logger = logger
        self.gamma = config['gamma']
        self.num_scenario = config['num_scenario']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.train_repeat_times = config['train_repeat_times']

        self.state_dim = config['ego_obs_dim']
        self.action_dim = config['ego_action_dim']
        self.clip_epsilon = config['clip_epsilon']
        self.batch_size = config['batch_size']
        self.lambda_gae_adv = config['lambda_gae_adv']
        self.lambda_entropy = config['lambda_entropy']
        self.dims = config['dims']

        self.model_type = config['model_type']
        self.model_path = Path(config['ROOT_DIR']) / config['model_path']
        self.obs_type = config['obs_type']
        self.save_cbv_policy_name = config['cbv_policy_name']

        self.wandb_config = self.config['wandb']
        self.wandb_initialized = False

        if config['mode'] == 'eval':
            self.load_cbv_policy_name = config['pretrain_cbv']
        else:
            self.load_cbv_policy_name = config['cbv_policy_name']

        self.policy = CUDA(ActorPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)  # trick about eps
        self.value = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)  # trick about eps
        self.value_criterion = nn.SmoothL1Loss()

        self.mode = 'train'

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
        
    def set_buffer(self, buffer: EgoRolloutBuffer, total_routes):
        self.buffer: EgoRolloutBuffer = buffer
        self.total_routes = total_routes
        resume = 'auto' if self.config['resume'] else 'never'
        # init the wandb logger
        if not self.wandb_initialized:
            wandb.init(
                project=self.wandb_config['project'],
                group=self.wandb_config['base_group'] + '_' + self.logger.output_dir.name,
                name=self.logger.output_dir.name,
                config=self.config,
                resume=resume,
                dir=self.logger.output_dir,
                mode=self.wandb_config['mode']
            )
            self.wandb_initialized = True

    def log_episode_reward(self, episode_reward, episode):
        wandb.log({'Ego episode reward': episode_reward}, step=episode)

    def lr_decay(self, e_i):
        lr_policy_now = self.policy_lr * (1 - e_i / self.total_routes)
        lr_value_now = self.value_lr * (1 - e_i / self.total_routes)
        for p in self.policy_optim.param_groups:
            p['lr'] = lr_policy_now
        for p in self.value_optim.param_groups:
            p['lr'] = lr_value_now

    def get_action(self, obs, infos, deterministic=False) -> Dict[str, np.ndarray]:
        env_id_list = [info['env_id'] for info in infos]
        flatten_obs = np.concatenate([x.reshape(1, -1) for x in obs], axis=0)
        obs_tensor = CUDA(torch.FloatTensor(flatten_obs))
        ego_actions = {env_id:{} for env_id in range(self.num_scenario)}
        ego_actions_log_prob = {env_id:{} for env_id in range(self.num_scenario)}
        if deterministic:
            actions = self.policy(obs_tensor)
            for action, env_id in zip(actions, env_id_list):
                ego_actions[env_id] = CPU(action)
                ego_actions_log_prob[env_id] = None   
        else:
            actions, log_probs = self.policy.get_action(obs_tensor)
            for action, log_prob, env_id in zip(actions, log_probs, env_id_list):
                ego_actions[env_id] = CPU(action)
                ego_actions_log_prob[env_id] = CPU(log_prob)

        data = {
            'ego_actions': ego_actions,
            'ego_actions_log_prob': ego_actions_log_prob
        }
        return data

    def get_advantages_GAE(self, rewards, undones, values, next_values, unterminated):
        """
            unterminated: if the CBV collide with object, then it is terminated
            undone: if the CBV is stuck or collide or max step will done
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
            self.lr_decay(e_i)  # add the learning rate decay

            batch = self.buffer.get_all_np_data()

            states = CUDA(torch.FloatTensor(batch['ego_obs']))
            next_states = CUDA(torch.FloatTensor(batch['ego_next_obs']))
            actions = CUDA(torch.FloatTensor(batch['ego_actions']))
            log_probs = CUDA(torch.FloatTensor(batch['ego_actions_log_prob']))
            rewards = CUDA(torch.FloatTensor(batch['ego_reward']))
            undones = CUDA(torch.FloatTensor(1-batch['ego_terminal']))
            unterminated = CUDA(torch.FloatTensor(1-batch['ego_terminal']))

            buffer_size = states.shape[0]

            values = self.value(states)
            next_values = self.value(next_states)

            advantages = self.get_advantages_GAE(rewards, undones, values, next_values, unterminated)
            reward_sums = advantages + values
            del rewards, undones, values, unterminated

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
            L2 = advantage * torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            surrogate = torch.min(L1, L2).mean()
            actor_loss = -(surrogate + entropy.mean() * self.lambda_entropy)
            self.policy_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optim.step()
        
        # logging
        wandb.log({
            "Ego value loss": value_loss.item(),
            "Ego actor entropy":entropy.mean().item(),
            "Ego actor loss": actor_loss.item()
        }, step=e_i)

        # reset buffer
        self.buffer.reset_buffer()

    def save_model(self, episode):
        save_dir = self.model_path / self.save_cbv_policy_name
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"model_{self.name}_{self.model_type}_{episode:04}.torch"

        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'value_optim': self.value_optim.state_dict(),
            'episode': episode
        }, filepath)

        self.logger.log(f'>> Saving {self.name} model to {filepath}', 'yellow')

    def load_model(self, resume=True):
        load_dir = self.model_path / self.load_cbv_policy_name
        episode_files = list(load_dir.glob(f"model_{self.name}_{self.model_type}_*.torch"))

        if resume and episode_files:
            episode = max([int(file.stem.split('.')[-1]) for file in episode_files])
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
                self.logger.log(f">> Clearning all the data in {load_dir}", 'red')
                for file in episode_files:
                    file.unlink()  # remove all the model files
            self._initialize_model()
            self.continue_episode = 0

    def _initialize_model(self):
        self.policy = CUDA(ActorPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)
        self.value = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)

    def finish(self):
        if self.wandb_initialized:
            wandb.finish()