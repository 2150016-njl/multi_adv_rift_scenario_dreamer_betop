#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : exception.py
@Date    : 2025/03/25
'''

class SpawnRuntimeError(Exception):
    def __init__(self, message):
        super().__init__(message)