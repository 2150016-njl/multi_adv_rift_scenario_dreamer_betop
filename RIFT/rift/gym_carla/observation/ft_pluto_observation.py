#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : sft_pluto_observation.py
@Date    : 2024/09/25
"""

from rift.gym_carla.observation.pluto_observation import CBVPlutoObservation


class CBVSFTPlutoObservation(CBVPlutoObservation):
    type = 'cbv_sft_pluto_obs'

    def __init__(self, env_params, CBVAction, config_path='rift/cbv/planning/config/sft_pluto.yaml'):
        super().__init__(env_params, CBVAction, config_path)


class CBVRTRPlutoObservation(CBVPlutoObservation):
    type = 'cbv_rtr_pluto_obs'

    def __init__(self, env_params, CBVAction, config_path='rift/cbv/planning/config/rtr_pluto.yaml'):
        super().__init__(env_params, CBVAction, config_path)


class CBVRSPlutoObservation(CBVPlutoObservation):
    type = 'cbv_rs_pluto_obs'

    def __init__(self, env_params, CBVAction, config_path='rift/cbv/planning/config/rs_pluto.yaml'):
        super().__init__(env_params, CBVAction, config_path)


class CBVReinforcePlutoObservation(CBVPlutoObservation):
    type = 'cbv_reinforce_pluto_obs'

    def __init__(self, env_params, CBVAction, config_path='rift/cbv/planning/config/reinforce_pluto.yaml'):
        super().__init__(env_params, CBVAction, config_path=config_path)


class CBVPPOPlutoObservation(CBVPlutoObservation):
    type = 'cbv_ppo_pluto_obs'

    def __init__(self, env_params, CBVAction, config_path='rift/cbv/planning/config/ppo_pluto.yaml'):
        super().__init__(env_params, CBVAction, config_path)


class CBVGRPOPlutoObservation(CBVPlutoObservation):
    type = 'cbv_grpo_pluto_obs'

    def __init__(self, env_params, CBVAction, config_path='rift/cbv/planning/config/grpo_pluto.yaml'):
        super().__init__(env_params, CBVAction, config_path)

class CBVRIFTPlutoObservation(CBVPlutoObservation):
    type = 'cbv_rift_pluto_obs'

    def __init__(self, env_params, CBVAction, config_path='rift/cbv/planning/config/rift_pluto.yaml'):
        super().__init__(env_params, CBVAction, config_path)