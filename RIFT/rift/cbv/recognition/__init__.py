#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2024/7/7
"""

from rift.cbv.recognition.rule_based.rule_cbv import RuleCBVRecog
from rift.cbv.recognition.attention_based.attn_cbv import AttnCBVRecog


CBV_RECOGNITION_LIST = {
    'attention': AttnCBVRecog,
    'rule': RuleCBVRecog,
}