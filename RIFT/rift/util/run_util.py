#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : run_util.py
@Date    : 2023/10/4
"""

import os
import os.path as osp
from fnmatch import fnmatch

import yaml


def load_config(config_path="default_config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)



