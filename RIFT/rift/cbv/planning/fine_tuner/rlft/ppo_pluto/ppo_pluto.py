#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : ppo_pluto.py
@Date    : 2025/01/27
'''
import gc
import hydra
import torch
import hydra._internal.instantiate._instantiate2
import hydra.types

import wandb
from rift.cbv.planning.fine_tuner.rlft.rlft_pluto import RLFTPluto
from rift.cbv.planning.fine_tuner.training_builder import TrainingEngine, build_training_engine
from rift.cbv.planning.pluto.model.pluto_model import PlanningModel
from rift.gym_carla.utils.net import CriticPPO
from rift.util.torch_util import CUDA

# Instantiation related symbols
instantiate = hydra._internal.instantiate._instantiate2.instantiate


class PPOPlutoModel(PlanningModel):
    def __init__(
            self,
            radius: float,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            clip_epsilon: float = 0.2,
            lambda_entropy: float = 0.01
        ):
        super().__init__(radius=radius)
        self.clip_epsilon = clip_epsilon
        self.lambda_entropy = lambda_entropy
        self.value_net = CriticPPO(dims=hidden_dim, state_dim=state_dim, action_dim=action_dim)


class PPOPluto(RLFTPluto):
    name = 'ppo_pluto'
    type = 'learnable'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.ppo_config = config['ppo']
        self.hidden_dim = self.ppo_config['hidden_dim']
        self.state_dim = self.ppo_config['state_dim']
        self.action_dim = self.ppo_config['action_dim']
        self.clip_epsilon = self.ppo_config['clip_epsilon']
        self.lambda_entropy = self.ppo_config['lambda_entropy']

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            # build the train model
            self.train_model = CUDA(PPOPlutoModel(
                radius=self.radius, state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim,
                clip_epsilon=self.clip_epsilon, lambda_entropy=self.lambda_entropy)
            )  # PPO fine-tuning pluto need an extra value net
            self.train_model.train()
            # inference model
            self.pluto_model.eval()
        elif mode == 'eval':
            self.pluto_model.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def train(self, e_i):
        self.logger.log('>> Starting fine-tuning...', color='yellow')
        # rebuild the training engine for each training
        dir_path = self.model_path / self.load_agent_info
        
        # update the training cfg learning rate
        self.cfg.lr = max(self.initial_lr* (self.cfg.cl_lr_decay ** self.current_epoch), self.cfg.min_lr)  # decay the learning rate through close loop training

        # build the training engine
        training_engine: TrainingEngine = build_training_engine(
            cfg=self.cfg,
            dir_path=dir_path,
            carla_episode=e_i,
            torch_module_wrapper=self.train_model,  # copy the training pluto model
            buffer=self.buffer,
        )
        
        # load the current training model ckpt
        training_engine.model.load_state_dict(self.load_train_checkpoint(self.checkpoint, device_name=self.device), strict=False)

        # process_buffer
        training_engine.datamodule.preprocess_buffer(training_engine.model.model)

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
    
    def load_infer_checkpoint(self, checkpoint: str, device_name: str):
        ckpt = torch.load(checkpoint, map_location=device_name)
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        
        # inference model don't need the value net
        value_net_keys = [k for k in state_dict.keys() if k.startswith('value_net')]
        if value_net_keys:
            for k in value_net_keys:
                del state_dict[k]
        
        return state_dict
