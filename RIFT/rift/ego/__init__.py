#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2023/10/4
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


EGO_POLICY_LIST = LazyPolicyRegistry({
    'behavior': ('rift.ego.behavior', 'Behavior'),
    'ppo': ('rift.ego.rl.ppo', 'PPO'),
    'expert': ('rift.ego.expert.expert', 'Expert'),
    'plant': ('rift.ego.plant.plant', 'PlanT'),
    'expert_disturb': ('rift.ego.expert_disturb', 'ExpertDisturb'),
    'pdm_lite': ('rift.ego.pdm_lite.pdm_lite', 'PDM_LITE'),
    'vad': ('rift.ego.b2d.e2e_agent', 'VAD'),
    'uniad': ('rift.ego.b2d.e2e_agent', 'UniAD'),
    'sparsedrive': ('rift.ego.b2d.e2e_agent', 'SparseDrive'),
})
