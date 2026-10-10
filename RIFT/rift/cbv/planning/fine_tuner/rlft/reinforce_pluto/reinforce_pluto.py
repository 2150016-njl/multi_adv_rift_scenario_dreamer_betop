#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : reinforce_pluto.py
@Date    : 2025/01/27
'''
from rift.cbv.planning.fine_tuner.rlft.rlft_pluto import RLFTPluto


class ReinforcePluto(RLFTPluto):
    name = 'reinforce_pluto'
    type = 'learnable'

    def __init__(self, config, logger):
        super().__init__(config, logger)