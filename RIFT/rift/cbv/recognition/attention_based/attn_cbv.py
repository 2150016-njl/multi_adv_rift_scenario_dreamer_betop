#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : att_cbv.py
@Date    : 2024/7/7
"""
from rift.cbv.recognition.attention_based.attn_model import AttnModel
from rift.cbv.recognition.base_cbv import BaseCBVRecog


class AttnCBVRecog(BaseCBVRecog):
    name = 'Attention'

    """ This is the template for implementing the CBV candidate selection for a scenario. """

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.AttnModel = AttnModel(config, self.logger)

    def get_CBVs(self, ego_vehicle, CBVs_id, local_route_waypoints, rest_route_waypoints, red_light_state=None):
        CBV_candidates = self.get_CBV_candidates(ego_vehicle, CBVs_id, rest_route_waypoints)
        sample_length = self.max_agent_num - len(CBVs_id)

        CBVs = self.AttnModel.get_CBVs(ego_vehicle, CBV_candidates, local_route_waypoints, red_light_state, sample_length)
        return CBVs

    def load_model(self):
        # load the pretrained model of the AttnModel, the model is fixed, so no need for fine-tuning
        self.AttnModel.load_ckpt(strict=True)
        self.logger.log(f'>> Successfully loading the Attention-based Recognition model', color='yellow')
