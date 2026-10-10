#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : scenario_data_loader.py
@Date    : 2023/10/4
"""

from collections import defaultdict
from typing import Dict, List
import numpy as np
import os.path as osp
from dictor import dictor
from scipy.spatial import cKDTree
from rift.scenario.tools.checkpoint_tools import fetch_dict
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.util.logger import Logger


def calculate_interpolate_trajectory(config: RouteScenarioConfiguration):
    # get [x, y] along the route
    waypoint_xy = []
    for loc in config.keypoints:
        waypoint_xy.append([loc.x, loc.y])

    return waypoint_xy


def check_route_overlap(current_routes, route, distance_threshold=10):
    if not current_routes:
        return False
    
    current_waypoints = np.vstack([np.array(r) for r in current_routes])
    # build KD tree
    tree = cKDTree(current_waypoints)

    route_waypoints = np.array(route)

    indices = tree.query_ball_point(route_waypoints, distance_threshold)

    return any(len(i) > 0 for i in indices)


class EvalDataLoader:
    def __init__(self, config_dict: Dict[str, List[RouteScenarioConfiguration]], num_scenario: int, logger: Logger):
        self.logger = logger
        self.num_scenario = num_scenario
        self.maps = list(config_dict.keys())
        self.all_configs = []
        self.maps_boundaries = []

        start_idx = 0
        for map_name in self.maps:
            cfgs = config_dict[map_name]
            self.all_configs.extend(cfgs)
            end_idx = start_idx + len(cfgs)
            self.maps_boundaries.append((map_name, start_idx, end_idx))
            start_idx = end_idx
        
        self.total = len(self.all_configs)
        self.remain_map_scenario_idx = {
            map_name: list(range(start, end)) for map_name, start, end in self.maps_boundaries
        }
        
        self.current_map = self.maps[0]  # Use map name directly
        self.index = 0
        # Keep track of how many scenarios remain globally
        self.remain_count = self.total

        self.logger.log(">> Finish Eval data loader preparation")

    def __len__(self):
        return self.remain_count

    def sampler(self):
        """
        Sample scenarios without route overlap from the current map.
        Moves to the next map when the current one is exhausted.
        Returns an empty list if all maps are exhausted.
        
        Logic:
          - Skip maps that have no remaining scenarios.
          - From the current map, select up to `num_scenario` scenarios.
          - Ensure chosen scenarios don't overlap with each other.
          - Update remain lists and index.
          - Move to next map if the current one is exhausted.
        """
        # Move forward if the current map is empty
        while self.current_map and not self.remain_map_scenario_idx[self.current_map]:
            # Find the next map with remaining scenarios
            map_idx = self.maps.index(self.current_map)
            if map_idx + 1 < len(self.maps):
                self.current_map = self.maps[map_idx + 1]
            else:
                self.current_map = None  # No more maps left
                break

        # If all maps are done
        if not self.current_map:
            return []

        current_map_list = self.remain_map_scenario_idx[self.current_map]
        sample_num = min(self.num_scenario, len(current_map_list))

        # If no scenarios can be sampled (e.g., current map empty), move on
        if sample_num == 0:
            self.current_map = None
            return self.sampler()

        # Select non-overlapping scenarios
        selected_idx = []
        selected_routes = []
        for s_i in current_map_list:
            traj = calculate_interpolate_trajectory(self.all_configs[s_i])
            if not check_route_overlap(selected_routes, traj):
                selected_idx.append(s_i)
                selected_routes.append(traj)
            if len(selected_idx) == sample_num:
                break

        # Update remaining and selected scenarios
        selected_set = set(selected_idx)
        new_map_list = []
        selected_scenario = []
        for s_i in current_map_list:
            if s_i in selected_set:
                config = self.all_configs[s_i]
                config.index = self.index
                self.index += 1
                selected_scenario.append(config)
                self.remain_count -= 1
            else:
                new_map_list.append(s_i)

        # Update the remain list for the current map
        self.remain_map_scenario_idx[self.current_map] = new_map_list

        # If the current map is exhausted now, the next call will move to the next map
        return selected_scenario
    
    def validate_and_resume(self, endpoint_file='simulation_results.json'):
        """
        Validates the endpoint by comparing several of its values with the current running routes.
        If all checks pass, the simulation starts from the last route.
        Otherwise, the resume is canceled, and the leaderboard goes back to normal behavior
        """
        resume = True
        endpoint = self.logger.output_dir / endpoint_file
        
        if not endpoint.exists():
            self.logger.log(">> No endpoint file found. Starting from scratch", color='yellow')
            return False
        
        data = fetch_dict(endpoint.as_posix())
        if not data:
            return False

        entry_status = dictor(data, 'entry_status')
        if not entry_status:
            self.logger.log(">> Problem reading checkpoint. Given checkpoint is malformed", color='red')
            resume = False
        
        if entry_status == "Invalid":
            self.logger.log(">> Problem reading checkpoint. The 'entry_status' is 'Invalid'", color='red')
            resume = False

        checkpoint_dict = dictor(data, '_checkpoint')
        if not checkpoint_dict or 'progress' not in checkpoint_dict:
            self.logger.log(">> Problem reading checkpoint. Given endpoint is malformed", color='red')
            resume = False

        progress = checkpoint_dict['progress']
        if progress[1] != self.total:
            self.logger.log(">> Problem reading checkpoint. Endpoint's amount of routes does not match the given one", color='red')
            resume = False

        route_data = dictor(checkpoint_dict, 'records')

        route_data_ids = defaultdict(set)
        for route_record in route_data:
            route_id = route_record['route_id']
            route_rep_index = int(route_id.split('_rep')[-1]) + 1
            map_name = f"{route_record['town_name']}-{route_rep_index}x"
            route_data_ids[map_name].add(route_id)

        # Filter remaining scenarios by removing completed ones
        for map_name, completed_route_ids in route_data_ids.items():
            # Use set to efficiently check for completed route_data_ids
            self.remain_map_scenario_idx[map_name] = [
                idx for idx in self.remain_map_scenario_idx[map_name]
                if f"{self.all_configs[idx].name}_rep{self.all_configs[idx].repetition_index}" not in completed_route_ids
            ]

        # If the current map has no remaining scenarios, find the next available map
        if not self.remain_map_scenario_idx[self.current_map]:
            # Find the index of the current map in self.maps
            current_map_idx = self.maps.index(self.current_map)
            
            # Search for the next available map after current_map
            for next_map in self.maps[current_map_idx + 1:]:
                if self.remain_map_scenario_idx[next_map]:
                    self.current_map = next_map
                    break
            else:
                self.current_map = None  # No more maps left with remaining scenarios

        # Validate progress consistency
        finished_scenarios_num = self.total - sum(len(value) for value in self.remain_map_scenario_idx.values())

        if progress[0] != finished_scenarios_num:
            self.logger.log(f">> Mismatch in progress. Expected: {finished_scenarios_num}, Found: {progress[0]}", color='red')
            resume = False

        if resume:
            self.index = progress[0]
            self.remain_count -= progress[0]
            self.logger.log(f">> Resumed from episode {self.index}, remaining {self.remain_count} scenarios", color='yellow')
        else:
            self.logger.log(">> Resuming from checkpoint failed", color='red')

        return resume


class TrainDataLoader:
    def __init__(self, config_dict: Dict[str, List[RouteScenarioConfiguration]], num_scenario: int, logger: Logger):
        """
        A data loader for multiple maps, each with a list of route scenario configurations.
        
        config_dict: Dict of {map_name: [RouteScenarioConfiguration, ...]}
        num_scenario: Number of scenarios to attempt per sample
        logger: Logger instance
        """
        self.logger = logger
        self.num_scenario = num_scenario
        self.maps = list(config_dict.keys())
        self.all_configs = []
        self.map_boundaries = []

        start_idx = 0
        for map_name in self.maps:
            cfgs = config_dict[map_name]
            self.all_configs.extend(cfgs)
            end_idx = start_idx + len(cfgs)
            self.map_boundaries.append((map_name, start_idx, end_idx))
            start_idx = end_idx

        self.total = len(self.all_configs)
        self.remain_map_scenario_idx = {
            map_name: list(range(start, end)) for map_name, start, end in self.map_boundaries
        }
        
        self.current_map = self.maps[0]  # Use map name directly
        self.index = 0  # Global index for indexing each config (starting from 0)

        # Keep track of how many scenarios remain globally
        self.remain_count = self.total

        self.logger.log(f">> Finished Train data loader preparation, loading {self.total} scenarios", color='yellow')

    def __len__(self):
        return self.remain_count

    def sampler(self):
        """
        Sample scenarios without route overlap from the current map.
        Moves to the next map when the current one is exhausted.
        Returns an empty list if all maps are exhausted.
        
        Logic:
          - Skip maps that have no remaining scenarios.
          - From the current map, select up to `num_scenario` scenarios.
          - Ensure chosen scenarios don't overlap with each other.
          - Update remain lists and index.
          - Move to next map if the current one is exhausted.
        """
        # Move forward if the current map is empty
        while self.current_map and not self.remain_map_scenario_idx[self.current_map]:
            # Find the next map with remaining scenarios
            map_idx = self.maps.index(self.current_map)
            if map_idx + 1 < len(self.maps):
                self.current_map = self.maps[map_idx + 1]
            else:
                self.current_map = None  # No more maps left
                break

        # If all maps are done
        if not self.current_map:
            return []

        current_map_list = self.remain_map_scenario_idx[self.current_map]
        sample_num = min(self.num_scenario, len(current_map_list))

        # If no scenarios can be sampled (e.g., current map empty), move on
        if sample_num == 0:
            self.current_map = None
            return self.sampler()

        # Select non-overlapping scenarios
        selected_idx = []
        selected_routes = []
        for s_i in current_map_list:
            traj = calculate_interpolate_trajectory(self.all_configs[s_i])
            if not check_route_overlap(selected_routes, traj):
                selected_idx.append(s_i)
                selected_routes.append(traj)
            if len(selected_idx) == sample_num:
                break

        # Update remaining and selected scenarios
        selected_set = set(selected_idx)
        new_map_list = []
        selected_scenario = []
        for s_i in current_map_list:
            if s_i in selected_set:
                config = self.all_configs[s_i]
                config.index = self.index
                self.index += 1
                selected_scenario.append(config)
                self.remain_count -= 1
            else:
                new_map_list.append(s_i)
        
        # Update the remain list for the current map
        self.remain_map_scenario_idx[self.current_map] = new_map_list

        # If the current map is exhausted now, the next call will move to the next map
        return selected_scenario
    
    def validate_and_resume(self, start_episode, endpoint_file='route_info.txt'):
        """
        Validates and resumes training by checking the previously completed episodes
        from 'route_info.txt', filtering out completed routes from future episodes.
        
        start_episode: The episode to start resuming from
        endpoint_file: The file containing completed route data (default: 'route_info.txt')
        """
        route_data_ids = defaultdict(set)  # Use set for O(1) lookups
        file_path = self.logger.output_dir / endpoint_file
        if start_episode == 0:
            self.logger.log(f">> Training from scratch", color='yellow')
            if file_path.exists():
                self.logger.log(f">> Cleaning the route_info.txt", color='red')
                file_path.unlink()
            return
        else:
            if file_path.exists():
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                filtered_lines = []
                for line in lines:
                    parts = line.strip().split(", ")

                    try:
                        episode = int(parts[0].split(": ")[1])
                        route_data_id = int(parts[1].split(": ")[1])
                        town = parts[2].split(": ")[1]
                        rep_index = int(parts[3].split(": ")[1])
                        episode_reward = float(parts[4].split(": ")[1])
                        map_name = f"{town}-{rep_index + 1}x"
                    except (IndexError, ValueError):
                        self.logger.log(f"Skipping malformed line: {line}")
                        continue

                    if episode < start_episode:
                        route_data_ids[map_name].add(route_data_id)  # Add to set for faster lookups
                        filtered_lines.append(line)
                
                with open(file_path, 'w') as file:
                    file.writelines(filtered_lines)

                # Filter remaining scenarios by removing completed ones
                for map_name, completed_route_ids in route_data_ids.items():
                    # Use set to efficiently check for completed route_data_ids
                    self.remain_map_scenario_idx[map_name] = [
                        idx for idx in self.remain_map_scenario_idx[map_name]
                        if self.all_configs[idx].data_id not in completed_route_ids
                    ]

                # If the current map has no remaining scenarios, find the next available map
                if not self.remain_map_scenario_idx[self.current_map]:
                    # Find the index of the current map in self.maps
                    current_map_idx = self.maps.index(self.current_map)
                    
                    # Search for the next available map after current_map
                    for next_map in self.maps[current_map_idx + 1:]:
                        if self.remain_map_scenario_idx[next_map]:
                            self.current_map = next_map
                            break
                    else:
                        self.current_map = None  # No more maps left with remaining scenarios

                # Update the index to resume from the next episode
                self.index = start_episode
                self.remain_count -= start_episode
                self.logger.log(f"Resumed from episode {self.index}, remaining {self.remain_count} scenarios", color='yellow')
            else:
                self.logger.log(f"No corresponding route_info.txt file", color='red')
                raise FileNotFoundError(f"File not found: {file_path}")
        


