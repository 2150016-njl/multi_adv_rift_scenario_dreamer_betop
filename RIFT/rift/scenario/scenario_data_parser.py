#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : scenario_data_parser.py
@Date    : 2024/11/28
'''

import copy
import re
from collections import defaultdict
from typing import List
from rift.scenario.tools.route_parser import RouteParser
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.util.logger import Logger


def parse_index_config(config, routes_subset=0) -> List[RouteScenarioConfiguration]:
    """
    Returns a list of route configuration elements.
    """
    routes_file = config['routes']
    repetitions = config['repetitions']
    num_scenario = config['num_scenario']
    configs_list = []
    # parse all the routes
    route_configurations: List[RouteScenarioConfiguration] = RouteParser.parse_routes_file(routes_file, routes_subset)
    total = len(route_configurations) * repetitions

    for i, config in enumerate(route_configurations):
        for repetition in range(repetitions):
            config_copy = copy.copy(config)
            config_copy.data_id = i * repetitions + repetition  # repeats
            config_copy.repetition_index = repetition
            config_copy.num_scenario = num_scenario
            configs_list.append(config_copy)

    return configs_list, total


def parse_map_key(map_name):
    """
    Parse map_name to generate a tuple (index, town_number) for sorting.
    
    Args:
        map_name (str): Key in the format "TownXX-YYx" (e.g., "Town03-2x")
    
    Returns:
        tuple: (index, town_number) where:
            - index: The numeric value from "-YYx" (e.g., 2 from "2x")
            - town_number: The numeric value from "TownXX" (e.g., 3 from "Town03")
    """
    # Extract town number (e.g., "Town03" → 3, "Town10HD" → 10)
    town_part = map_name.split('-')[0]  # Get part before "-"
    town_num = 0
    if town_match := re.match(r'^Town(\d+)', town_part):
        town_num = int(town_match.group(1))  # Get numeric part
        
    # Extract index (e.g., "2x" → 2)
    index_part = map_name.split('-')[1].rstrip('x')  # Remove trailing "x"
    index = int(index_part)
    
    return (index, town_num)


class ScenarioDataParser(object):

    """
    Pure static class used to parse all the scenario data configuration parameters.
    """

    @staticmethod
    def scenario_parse(config, logger: Logger):
        """
            Data file should also come from args
        """
        logger.log(">> Parsing scenario route and data")
        parsed_configs, total_routes = parse_index_config(config, config.get('routes_subset', ''))

        config_by_map = defaultdict(list)
        for parsed_config in parsed_configs:            
            # cluster config according to the town and repetition index
            map_name = f'{parsed_config.town}-{parsed_config.repetition_index + 1}x'  # Town12-1x
            config_by_map[map_name].append(parsed_config)

        # sort the config by town number (from 01 to 13)
        sorted_config_by_map = {k: config_by_map[k] for k in sorted(config_by_map, key=parse_map_key)}

        return sorted_config_by_map, total_routes
    

