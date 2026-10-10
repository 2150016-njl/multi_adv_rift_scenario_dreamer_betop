#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : agent_state_encoder.py
@Date    : 2023/10/22
"""
import torch
from lightning.pytorch import LightningModule
from torch.nn.utils.rnn import pad_sequence
from rift.ego.utils.explainability_utils import *
from rift.gym_carla.utils.utils import get_bev_boxes, get_input_batch
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from einops import rearrange
from rift.util.torch_util import CUDA

import numpy as np
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
)


class AttnModel(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.obs_type = config['obs_type']
        self.viz_attn_map = config['viz_attn_map']
        self.model_path = config['pretrained_model_path']
        self.frame_rate = config['frame_rate']

    def load_ckpt(self, strict=True):
        updated_model_path = self.config['pretrained_model_path']
        self.net = CUDA(EncoderModel.load_from_checkpoint(updated_model_path, strict=strict, config=self.config))
        self.net.eval()

    def get_most_relevant_vehicles(self, data_car_ids, data_car, attn_vector, topk=1):
        # get topk indices of attn_vector
        if topk > len(attn_vector):
            topk = len(attn_vector)
        else:
            topk = topk

        attn_indices = np.argpartition(attn_vector, -topk)[-topk:]

        # get carla vehicles ids of topk vehicles
        keep_vehicle_ids = []
        keep_vehicle_attn = []
        for indice in attn_indices:
            if indice < len(data_car_ids):
                keep_vehicle_ids.append(data_car_ids[indice])
                keep_vehicle_attn.append(attn_vector[indice])

        # if we don't have any detected vehicle we should not have any ids here
        # otherwise we want #topk vehicles
        if len(data_car) > 0:
            assert len(keep_vehicle_ids) == topk
        else:
            assert len(keep_vehicle_ids) == 0

        # get topk vehicles indices
        if len(keep_vehicle_ids) >= 1:
            most_relevant_vehicles = [CarlaDataProvider.get_actor_by_id(vehicle_id) for vehicle_id in keep_vehicle_ids]
        else:
            most_relevant_vehicles = []

        return most_relevant_vehicles

    @torch.no_grad()
    def get_CBVs(self, ego_vehicle, CBV_candidates, local_route_waypoints, traffic_light_hazard, sample_length):
        if len(local_route_waypoints) > 1:
            target_point = local_route_waypoints[1]  # the preview waypoint
        else:
            target_point = local_route_waypoints[0]  # the first waypoint
        label_raw = get_bev_boxes(ego_vehicle, CBV_candidates, local_route_waypoints)
        input_batch, data_car_ids, data_car = get_input_batch(label_raw, target_point, traffic_light_hazard)
        x, y, _, tp, light = input_batch
        
        with torch.no_grad():
            _, attn_map = self.net(x, y, target_point=tp, light_hazard=light)

        # get the most relevant vehicle as the controlled background vehicle
        attn_vector = get_attn_norm_vehicles(self.config['attention_score'], data_car, attn_map)
        most_relevant_vehicles = self.get_most_relevant_vehicles(data_car_ids, data_car, attn_vector, topk=sample_length)

        if self.viz_attn_map:
            keep_vehicle_ids, _, keep_vehicle_attn = get_vehicleID_from_attn_scores(data_car_ids, data_car, self.config['topk'], attn_vector)
            draw_attention_bb_in_carla(keep_vehicle_ids, keep_vehicle_attn, self.frame_rate)
        return most_relevant_vehicles


class EncoderModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config_all = config
        self.config_net = config['network']

        self.object_types = 2 + 1  # vehicles, route +1 for padding and wp embedding
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y

        # model
        config = AutoConfig.from_pretrained(
            self.config_net['hf_checkpoint']
        )  # load config from hugging face model
        n_embd = config.hidden_size
        self.model = AutoModel.from_config(config=config)

        # sequence padding for batching
        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # +1 because at this step we still have the type indicator
        self.eos_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # unnecessary TODO: remove

        # token embedding
        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
        # object type embedding
        self.obj_token = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, self.num_attributes))
                for _ in range(self.object_types)
            ]
        )
        self.obj_emb = nn.ModuleList(
            [nn.Linear(self.num_attributes, n_embd) for _ in range(self.object_types)]
        )
        self.drop = nn.Dropout(self.config_net['embd_pdrop'])

    def pad_sequence_batch(self, x_batched):
        """
        Pads a batch of sequences to the longest sequence in the batch.
        """
        # split input into components
        x_batch_ids = x_batched[:, 0]

        x_tokens = x_batched[:, 1:]

        B = int(x_batch_ids[-1].item()) + 1
        input_batch = []
        for batch_id in range(B):
            # get the batch of elements
            x_batch_id_mask = x_batch_ids == batch_id

            # get the batch of types
            x_tokens_batch = x_tokens[x_batch_id_mask]

            x_seq = torch.cat([self.cls_emb, x_tokens_batch, self.eos_emb], dim=0)

            input_batch.append(x_seq)

        padded = torch.swapaxes(pad_sequence(input_batch), 0, 1)
        input_batch = padded[:B]

        return input_batch

    def forward(self, idx, target=None, target_point=None, light_hazard=None):

        # create batch of same size as input
        x_batched = torch.cat(idx, dim=0)
        input_batch = self.pad_sequence_batch(x_batched)
        input_batch_type = input_batch[:, :, 0]  # car or map
        input_batch_data = input_batch[:, :, 1:]

        # create same for output in case of multitask training to use this as ground truth
        if target is not None:
            y_batched = torch.cat(target, dim=0)
            output_batch = self.pad_sequence_batch(y_batched)
            output_batch_type = output_batch[:, :, 0]  # car or map
            output_batch_data = output_batch[:, :, 1:]

        # create masks by object type
        car_mask = (input_batch_type == 1).unsqueeze(-1)
        route_mask = (input_batch_type == 2).unsqueeze(-1)
        other_mask = torch.logical_and(route_mask.logical_not(), car_mask.logical_not())
        masks = [car_mask, route_mask, other_mask]

        # get size of input
        (B, O, A) = (input_batch_data.shape)  # batch size, number of objects, number of attributes

        # embed tokens object wise (one object -> one token embedding)
        input_batch_data = rearrange(
            input_batch_data, "b objects attributes -> (b objects) attributes"
        )
        embedding = self.tok_emb(input_batch_data)
        embedding = rearrange(embedding, "(b o) features -> b o features", b=B, o=O)

        # create object type embedding
        obj_embeddings = [
            self.obj_emb[i](self.obj_token[i]) for i in range(self.object_types)
        ]  # list of a tensors of size 1 x features

        # add object type embedding to embedding (mask needed to only add to the correct tokens)
        embedding = [
            (embedding + obj_embeddings[i]) * masks[i] for i in range(self.object_types)
        ]
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)

        # embedding dropout
        x = self.drop(embedding)

        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions

        return x, attn_map