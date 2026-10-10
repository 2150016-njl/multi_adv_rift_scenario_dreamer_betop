#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2024/8/15
"""
from importlib import import_module


class LazyPolicyRegistry(dict):
    def __init__(self, entries):
        super().__init__(entries)
        self._cache = {}

    def __getitem__(self, name):
        if name in self._cache:
            return self._cache[name]
        module_name, class_name = super().__getitem__(name)
        policy_cls = getattr(import_module(module_name), class_name)
        self._cache[name] = policy_cls
        return policy_cls

    def get(self, name, default=None):
        if name not in self:
            return default
        return self[name]


CBV_POLICY_LIST = LazyPolicyRegistry({
    'standard': ('rift.cbv.planning.dummy_policy', 'DummyPolicy'),
    'ppo': ('rift.cbv.planning.rl.ppo', 'PPO'),
    'frea': ('rift.cbv.planning.rl.frea', 'FREA'),
    'fppo_rs': ('rift.cbv.planning.rl.frea', 'FPPORs'),
    'pluto': ('rift.cbv.planning.pluto.pluto', 'PLUTO'),
    'sft_pluto': ('rift.cbv.planning.fine_tuner.sft.sft_pluto', 'SFTPluto'),
    'rtr_pluto': ('rift.cbv.planning.fine_tuner.sft.rtr_pluto.rtr_pluto', 'RTRPluto'),
    'rs_pluto': ('rift.cbv.planning.fine_tuner.sft.rs_pluto.rs_pluto', 'RewardShapingPluto'),
    'reinforce_pluto': ('rift.cbv.planning.fine_tuner.rlft.reinforce_pluto.reinforce_pluto', 'ReinforcePluto'),
    'ppo_pluto': ('rift.cbv.planning.fine_tuner.rlft.ppo_pluto.ppo_pluto', 'PPOPluto'),
    'grpo_pluto': ('rift.cbv.planning.fine_tuner.rlft.grpo_pluto.grpo_pluto', 'GRPOPluto'),
    'rift_pluto': ('rift.cbv.planning.fine_tuner.rlft.rift_pluto.rift_pluto', 'RIFTPluto'),
})
