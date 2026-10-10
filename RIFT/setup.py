#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : setup.py
@Date    : 2023/10/4
"""

from setuptools import setup

setup(name='rift',
      packages=["rift"],
      include_package_data=True,
      version='1.0.0',
      install_requires=['gym'])
