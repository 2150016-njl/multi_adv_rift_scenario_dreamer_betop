#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : logger.py
@Date    : 2023/10/4
"""
import json
import os
import os.path as osp
from pathlib import Path
from typing import List

import joblib
import numpy as np

from rift.gym_carla.visualization.video_render import VideoRender
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration


# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__))))


def setup_logger_dir(output_dir, seed, mode, ego=None, cbv=None, cbv_recog=None):

    # Make a base path
    relpath = mode

    # specify agent policy and scenario policy in the experiment directory.
    exp_name = ego + '-' + cbv + '-' + cbv_recog

    # Make a seed-specific subfolder in the experiment directory.
    subfolder = ''.join([exp_name, '-seed', str(seed)])
    relpath = osp.join(relpath, subfolder)

    logger_dir = os.path.join(DEFAULT_DATA_DIR, output_dir, relpath)
    return logger_dir

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)
        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]
        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)
        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x)
    std = np.std(x)  # compute global std
    if with_min_and_max:
        return mean, std, np.min(x), np.max(x)
    return mean, std


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def colorprint(string, color):
    print(colorize(string, color, bold=True))


class Logger:
    """
        A general-purpose logger.
        Makes it easy to save diagnostics, hyperparameter configurations, the state of a training run, and the trained model.
    """
    def __init__(self, output_dir=None):
        """
            Initialize a Logger.

            Args:
                output_dir (string): A directory for saving results to.
        """
        self.video_render = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log('>> ' + '-' * 40)
        self.log(">> Logging to %s" % self.output_dir, 'green')
        self.log('>> ' + '-' * 40)

    def log(self, msg, color='green'):
        # print with color
        print(colorize(msg, color, bold=True))

    def log_route_info(self, routes: List[RouteScenarioConfiguration], episode_reward: float):
        file_path = self.output_dir / 'route_info.txt'
        with open(file_path, 'a') as file:
            file.writelines(f"episode: {route.index}, route_data_id: {route.data_id}, town: {route.town}, rep_index: {route.repetition_index}, episode_reward: {round(episode_reward, 2)}\n" for route in routes)
    
    def init_video_render(self, env_params):
        self.video_render = VideoRender(self.output_dir, env_params, logger=self)
