#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : frea.py
@Date    : 2025/02/11
'''

from rift.cbv.planning.rl.ppo import PPO
from rift.util.logger import Logger


class FREA(PPO):
    name = 'frea'
    type = 'rl'
    def __init__(self, config, logger: Logger):
        super(FREA, self).__init__(config, logger)

    def train(self):
        """
        Train the FREA model
        """
        raise NotImplementedError(">> FREA train method is not provided here, please refer to https://github.com/CurryChen77/FREA")
    

class FPPORs(PPO):
    name = 'fppo_rs'
    type = 'rl'
    def __init__(self, config, logger: Logger):
        super(FPPORs, self).__init__(config, logger)

    def train(self):
        """
        Train the FPPORs model
        """
        raise NotImplementedError(">> FPPORs train method is not provided here, please refer to https://github.com/CurryChen77/FREA")
