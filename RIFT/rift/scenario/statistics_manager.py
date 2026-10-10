#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD leaderboard
"""

from __future__ import print_function
from pathlib import Path
from typing import Dict

from dictor import dictor
import math
import os.path as osp
from bisect import bisect_left

import numpy as np

from rift.scenario.tools.traffic_events import TrafficEventType

from rift.scenario.tools.checkpoint_tools import fetch_dict, save_dict

PENALTY_VALUE_DICT = {
    # Traffic events that substract a set amount of points.
    TrafficEventType.COLLISION_PEDESTRIAN: 0.5,
    TrafficEventType.COLLISION_VEHICLE: 0.6,
    TrafficEventType.COLLISION_STATIC: 0.65,
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 0.7,
    TrafficEventType.STOP_INFRACTION: 0.8,
    TrafficEventType.SCENARIO_TIMEOUT: 0.7,
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 0.7
}
PENALTY_PERC_DICT = {
    # Traffic events that substract a varying amount of points. This is the per unit value.
    # 'increases' means that the higher the value, the higher the penalty.
    # 'decreases' means that the ideal value is 100 and the lower the value, the higher the penalty.
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: [0, 'increases'],  # All route traversed through outside lanes is ignored
    # TrafficEventType.MIN_SPEED_INFRACTION: [0.7, 'decreases'],
    TrafficEventType.MIN_SPEED_INFRACTION: [0.7, 'unused'],
}

PENALTY_NAME_DICT = {
    TrafficEventType.COLLISION_STATIC: 'collisions_layout',
    TrafficEventType.COLLISION_PEDESTRIAN: 'collisions_pedestrian',
    TrafficEventType.COLLISION_VEHICLE: 'collisions_vehicle',
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 'red_light',
    TrafficEventType.STOP_INFRACTION: 'stop_infraction',
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: 'outside_route_lanes',
    TrafficEventType.MIN_SPEED_INFRACTION: 'min_speed_infractions',
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 'yield_emergency_vehicle_infractions',
    TrafficEventType.SCENARIO_TIMEOUT: 'scenario_timeouts',
    TrafficEventType.ROUTE_DEVIATION: 'route_dev',
    TrafficEventType.VEHICLE_BLOCKED: 'vehicle_blocked',
}

# Limit the entry status to some values. Eligible should always be gotten from this table
ENTRY_STATUS_VALUES = ['Started', 'Finished', 'Rejected', 'Crashed', 'Invalid']
ELIGIBLE_VALUES = {'Started': False, 'Finished': True, 'Rejected': False, 'Crashed': False, 'Invalid': False}

# Dictionary mapping a route failure with the 'entry status' and 'status'
FAILURE_MESSAGES = {
    "Simulation" : ["Crashed", "Simulation crashed"],
    "Sensors": ["Rejected", "Agent's sensors were invalid"],
    "Agent_init": ["Started", "Agent couldn't be set up"],
    "Agent_runtime": ["Started", "Agent crashed"]
}

ROUND_DIGITS = 3
ROUND_DIGITS_SCORE = 6

CBV_DATA_BINS = {
    'speed': [0.0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10, 12, 14],  # speed data bin (unit: m/s)
    'delta_speed': [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.0, 6.5, 7.5, 8.0, 8.5, 9.0],  # delta_speed data bin (unit: m/s)
    'target_speed': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], # target speed bin of all CBVs (unit: m/s)
    'acc': [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],  # acceleration data bin (unit: m/s^2)
    'jerk': [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # jerk data bin (unit: m/s^3)
}

EGO_DATA_BINS = {
    'RTTC': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'ACT': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'EI': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
}
EGO_SPEED_BINS = [0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8, 10]  # speed data bin (unit: m/s)


class RouteRecord():
    def __init__(self):
        self.index = -1
        self.route_id = None
        self.scenario_name = None
        self.weather_id = None
        self.save_name = None
        self.status = 'Started'
        self.num_infractions = 0
        self.infractions = {}
        for event_name in PENALTY_NAME_DICT.values():
            self.infractions[event_name] = []
        self.infractions['route_timeout'] = []

        self.scores = {
            'score_route': 0,
            'score_penalty': 0,
            'score_composed': 0
        }

        self.meta = {
            'route_length': 0,
            'duration_game': 0,
            'duration_system': 0,
            'cbv_total_game_time': 0,
            'cbv_off_road_game_time': 0,
            'cbv_uncomfortable_game_time': 0,
            'cbv_progress': 0,
            'cbv_collision_count': 0,
            'cbv_count': 0,
            'cbv_reach_goal_count': 0
        }
        self.meta.update({
            k: v
            for key, BINs in CBV_DATA_BINS.items()
            for k, v in {
                f'cbv_{key}_distribution': {f"{BINs[i]}~{BINs[i+1]}": 0 for i in range(len(BINs)-1)},

            }.items()
        })

        self.meta.update({
            f'ego_{key}_distribution': {
                f"speed{EGO_SPEED_BINS[j]}~{EGO_SPEED_BINS[j+1]}_{key}{BINs[i]}~{BINs[i+1]}": 0
                for j in range(len(EGO_SPEED_BINS) - 1)
                for i in range(len(BINs) - 1)
            }
            for key, BINs in EGO_DATA_BINS.items()
        })

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)


class GlobalRecord():
    def __init__(self):
        self.index = -1
        self.route_id = -1
        self.status = 'Perfect'
        self.infractions = {}
        for event_name in PENALTY_NAME_DICT.values():
            self.infractions[event_name] = 0
        self.infractions['route_timeout'] = 0

        self.scores_mean = {
            'score_composed': 0,
            'score_route': 0,
            'score_penalty': 0
        }
        self.scores_std_dev = self.scores_mean.copy()

        self.meta = {
            'total_length': 0,
            'duration_game': 0,
            'duration_system': 0,
            'exceptions': [],
            'cbv_total_game_time': 0,
            'cbv_off_road_game_time': 0,
            'cbv_uncomfortable_game_time': 0,
            'cbv_progress': 0,
            'cbv_collision_count': 0,
            'cbv_count': 0,
            'cbv_reach_goal_count': 0
        }
        self.meta.update({
            k: v
            for key, BINs in CBV_DATA_BINS.items()
            for k, v in {
                f'cbv_{key}_distribution': {f"{BINs[i]}~{BINs[i+1]}": 0 for i in range(len(BINs)-1)},
                f'cbv_{key}_mean': 0,
                f'cbv_{key}_std': 0
            }.items()
        })
        self.meta.update({
            f'ego_{key}_distribution': {
                f"speed{EGO_SPEED_BINS[j]}~{EGO_SPEED_BINS[j+1]}_{key}{BINs[i]}~{BINs[i+1]}": 0
                for j in range(len(EGO_SPEED_BINS) - 1)
                for i in range(len(BINs) - 1)
            }
            for key, BINs in EGO_DATA_BINS.items()
        })

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)

class Checkpoint():

    def __init__(self):
        self.global_record = {}
        self.progress = []
        self.records = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d['global_record'] = self.global_record.to_json() if self.global_record else {}
        d['progress'] = self.progress
        d['records'] = []
        d['records'] = [x.to_json() for x in self.records if x.index != -1]  # Index -1 = Route in progress

        return d


class Results():

    def __init__(self):
        self.checkpoint = Checkpoint()
        self.entry_status = "Started"
        self.eligible = ELIGIBLE_VALUES[self.entry_status]
        self.sensors = []
        self.values = []
        self.labels = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d['_checkpoint'] = self.checkpoint.to_json()
        d['entry_status'] = self.entry_status
        d['eligible'] = self.eligible
        d['sensors'] = self.sensors
        d['values'] = self.values
        d['labels'] = self.labels

        return d


def to_route_record(record_dict):
    record = RouteRecord()
    for key, value in record_dict.items():
        setattr(record, key, value)

    return record


def compute_route_length(route):
    route_length = 0.0
    previous_location = None

    for transform, _ in route:
        location = transform.location
        if previous_location:
            dist_vec = location - previous_location
            route_length += dist_vec.length()
        previous_location = location

    return route_length



class StatisticsManager(object):

    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """

    def __init__(self, base_dir, endpoint='simulation_results.json', debug_endpoint='live_results.txt'):
        self._scenario = None
        self._route_length = 0
        self._total_routes = 0
        self._results = Results()
        # create the statistics output dir
        Path(base_dir).mkdir(exist_ok=True, parents=True)
        self._endpoint = osp.join(base_dir, endpoint)
        self._debug_endpoint = osp.join(base_dir, debug_endpoint)

    def add_file_records(self, endpoint=None):
        """Reads a file and saves its records onto the statistics manager"""
        file_path = self._endpoint if endpoint is None else endpoint
        data = fetch_dict(file_path)

        if data:
            route_records = dictor(data, '_checkpoint.records')
            if route_records:
                for record in route_records:
                    self._results.checkpoint.records.append(to_route_record(record))

    def clear_records(self):
        """Cleanes up the file"""
        if not self._endpoint.startswith(('http:', 'https:', 'ftp:')):
            with open(self._endpoint, 'w') as fd:
                fd.truncate(0)

    def sort_records(self):
        """Sorts the route records according to their route id (This being i.e RouteScenario0_rep0)"""
        self._results.checkpoint.records.sort(key=lambda x: (
            int(x.route_id.split('_')[1]),
            int(x.route_id.split('_rep')[-1])
        ))

        for i, record in enumerate(self._results.checkpoint.records):
            record.index = i

    def write_live_results(self, index, ego_data: Dict[str, list], CBVs_data: Dict[str, list]):
        """Writes live results"""
        route_record = self._results.checkpoint.records[index]

        route_record.meta['cbv_total_game_time'] += round(CBVs_data.pop('total_game_time'), ROUND_DIGITS)
        route_record.meta['cbv_off_road_game_time'] += round(CBVs_data.pop('off_road_game_time'), ROUND_DIGITS)
        route_record.meta['cbv_uncomfortable_game_time'] += round(CBVs_data.pop('uncomfortable_game_time'), ROUND_DIGITS)
        route_record.meta['cbv_progress'] += round(CBVs_data.pop('total_progress'), ROUND_DIGITS)
        route_record.meta['cbv_collision_count'] += CBVs_data.pop('collision_count')
        route_record.meta['cbv_count'] += CBVs_data.pop('new_cbv_count')
        route_record.meta['cbv_reach_goal_count'] = max(CBVs_data.pop('reach_goal_count', 0), route_record.meta['cbv_reach_goal_count'])

        ego_speed = float(ego_data.pop('speed', 0.0))
        ego_control = ego_data.pop('control')
        ego_location = ego_data.pop('location')

        # process the cbv data
        for key, CBV_data in CBVs_data.items():
            for data in CBV_data:
                BINs = CBV_DATA_BINS[key]
                if data < BINs[0] or data >= BINs[-1]:
                    continue
                # use binary search to find the corresponding bin
                bin_index = bisect_left(BINs, data) - 1
                # make sure the bin index is within the valid range
                bin_index = max(0, min(bin_index, len(BINs) - 2))
                # update the data distribution
                data_bin = f"{BINs[bin_index]}~{BINs[bin_index + 1]}"
                route_record.meta[f'cbv_{key}_distribution'][data_bin] += 1
        
        # process the ego data
        for key, data in ego_data.items():
            if np.isnan(data):
                continue
            BINs = EGO_DATA_BINS[key]
            if data < BINs[0] or data >= BINs[-1]:
                continue
            # use binary search to find the corresponding bin
            bin_index = bisect_left(BINs, data) - 1
            # make sure the bin index is within the valid range
            bin_index = max(0, min(bin_index, len(BINs) - 2))
            # update the data distribution
            data_bin = f"{BINs[bin_index]}~{BINs[bin_index + 1]}"

            if np.isnan(ego_speed):
                continue
            if ego_speed < EGO_SPEED_BINS[0] or ego_speed >= EGO_SPEED_BINS[-1]:
                continue
            speed_bin_index = bisect_left(EGO_SPEED_BINS, ego_speed) - 1
            speed_bin_index = max(0, min(speed_bin_index, len(EGO_SPEED_BINS) - 2))
            speed_bin = f"{EGO_SPEED_BINS[speed_bin_index]}~{EGO_SPEED_BINS[speed_bin_index + 1]}"

            combo_key = f"speed{speed_bin}_{key}{data_bin}"
            route_record.meta[f'ego_{key}_distribution'][combo_key] += 1
        

        all_events = []
        if self._scenario:
            for node in self._scenario.get_criteria():
                all_events.extend(node.events)

        all_events.sort(key=lambda e: e.get_frame(), reverse=True)

        with open(self._debug_endpoint, 'w') as f:
            f.write("Route id: {}\n\n"
                    "Scenario: {}\n\n"
                    "Town name: {}\n\n"
                    "Weather id: {}\n\n"
                    "Save name: {}\n\n"
                    "Scores:\n"
                    "    Driving score:      {:.3f}\n"
                    "    Route completion:   {:.3f}\n"
                    "    Infraction penalty: {:.3f}\n\n"
                    "    Route length:    {:.3f}\n"
                    "    Game duration:   {:.3f}\n"
                    "    System duration: {:.3f}\n\n"
                    "Ego:\n"
                    "    Throttle:           {:.3f}\n"
                    "    Brake:              {:.3f}\n"
                    "    Steer:              {:.3f}\n\n"
                    "    Speed:           {:.3f} m/s\n\n"
                    "    Location:           ({:.3f} {:.3f} {:.3f})\n\n"
                    "Total infractions: {}\n"
                    "Last 5 infractions:\n".format(
                        route_record.route_id,
                        route_record.scenario_name,
                        route_record.town_name,
                        route_record.weather_id,
                        route_record.save_name,
                        route_record.scores["score_composed"],
                        route_record.scores["score_route"],
                        route_record.scores["score_penalty"],
                        route_record.meta["route_length"],
                        route_record.meta["duration_game"],
                        route_record.meta["duration_system"],
                        ego_control.throttle,
                        ego_control.brake,
                        ego_control.steer,
                        ego_speed,
                        ego_location.x,
                        ego_location.y,
                        ego_location.z,
                        route_record.num_infractions
                    )
                )
            for e in all_events[:5]:
                # Prevent showing the ROUTE_COMPLETION event.
                event_type = e.get_type()
                if event_type == TrafficEventType.ROUTE_COMPLETION:
                    continue
                string = "    " + str(e.get_type()).replace("TrafficEventType.", "")
                if event_type in PENALTY_VALUE_DICT:
                    string += " (penalty: " + str(PENALTY_VALUE_DICT[event_type]) + ")\n"
                elif event_type in PENALTY_PERC_DICT:
                    string += " (value: " + str(round(e.get_dict()['percentage'], 3)) + "%)\n"

                f.write(string)

    def save_sensors(self, sensors):
        self._results.sensors = sensors

    def save_entry_status(self, entry_status):
        if entry_status not in ENTRY_STATUS_VALUES:
            raise ValueError("Found an invalid value for 'entry_status'")
        self._results.entry_status = entry_status
        self._results.eligible = ELIGIBLE_VALUES[entry_status]

    def save_progress(self, route_index, total_routes):
        self._results.checkpoint.progress = [route_index, total_routes]
        self._total_routes = total_routes

    def create_route_data(self, route_id, scenario_name, weather_id, save_name, town_name, index):
        """
        Creates the basic route data.
        This is done at the beginning to ensure the data is saved, even if a crash occurs
        """
        route_record = RouteRecord()
        route_record.route_id = route_id
        route_record.scenario_name = scenario_name
        route_record.weather_id = weather_id
        route_record.save_name = save_name
        route_record.town_name = town_name

        # Check if we have to overwrite an element (when resuming), or create a new one
        route_records = self._results.checkpoint.records
        if index < len(route_records):
            self._results.checkpoint.records[index] = route_record
        else:
            self._results.checkpoint.records.append(route_record)

    def set_scenario(self, scenario):
        """Sets the scenario from which the statistics will be taken"""
        self._scenario = scenario
        self._route_length = round(compute_route_length(scenario.route), ROUND_DIGITS)

    def remove_scenario(self):
        """Removes the scenario"""
        self._scenario = None
        self._route_length = 0

    def compute_route_statistics(self, route_index, duration_time_system=-1, duration_time_game=-1, failure_message=""):
        """
        Compute the current statistics by evaluating all relevant scenario criteria.
        Failure message will not be empty if an external source has stopped the simulations (i.e simulation crash).
        For the rest of the cases, it will be filled by this function depending on the criteria.
        """
        def set_infraction_message():
            infraction_name = PENALTY_NAME_DICT[event.get_type()]
            route_record.infractions[infraction_name].append(event.get_message())

        def set_score_penalty(score_penalty):
            event_value = event.get_dict()['percentage']
            penalty_value, penalty_type = PENALTY_PERC_DICT[event.get_type()]
            if penalty_type == "decreases":
                score_penalty *= (1 - (1 - penalty_value) * (1 - event_value / 100))
            elif penalty_type == "increases":
                score_penalty *= (1 - (1 - penalty_value) * event_value / 100)
            elif penalty_type == "unused":
                pass
            else:
                raise ValueError("Found a criteria with an unknown penalty type")
            return score_penalty

        route_record = self._results.checkpoint.records[route_index]
        route_record.index = route_index

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0
        for event_name in PENALTY_NAME_DICT.values():
            route_record.infractions[event_name] = []

        # Update the route meta
        route_record.meta['route_length'] = self._route_length
        route_record.meta['duration_game'] = round(duration_time_game, ROUND_DIGITS)
        route_record.meta['duration_system'] = round(duration_time_system, ROUND_DIGITS)

        # Update the route infractions
        if self._scenario:
            if self._scenario.timeout_node.timeout:
                route_record.infractions['route_timeout'].append('Route timeout.')
                failure_message = "Agent timed out"

            for node in self._scenario.get_criteria():
                for event in node.events:
                    # Traffic events that substract a set amount of points
                    if event.get_type() in PENALTY_VALUE_DICT:
                        score_penalty *= PENALTY_VALUE_DICT[event.get_type()]
                        set_infraction_message()

                    # Traffic events that substract a varying amount of points
                    elif event.get_type() in PENALTY_PERC_DICT:
                        score_penalty = set_score_penalty(score_penalty)
                        set_infraction_message()

                    # Traffic events that stop the simulation
                    elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                        failure_message = "Agent deviated from the route"
                        set_infraction_message()

                    elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                        failure_message = "Agent got blocked"
                        set_infraction_message()

                    elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                        score_route = event.get_dict()['route_completed']
                        target_reached = score_route >= 100

        # Update route scores
        route_record.scores['score_route'] = round(score_route, ROUND_DIGITS_SCORE)
        route_record.scores['score_penalty'] = round(score_penalty, ROUND_DIGITS_SCORE)
        route_record.scores['score_composed'] = round(max(score_route * score_penalty, 0.0), ROUND_DIGITS_SCORE)

        # Update result
        route_record.num_infractions = sum([len(route_record.infractions[key]) for key in route_record.infractions])

        if target_reached:
            route_record.status = 'Completed' if route_record.num_infractions > 0 else 'Perfect'
        else:
            route_record.status = 'Failed'
            if failure_message:
                route_record.status += ' - ' + failure_message

        # Add the new data, or overwrite a previous result (happens when resuming the simulation)
        record_len = len(self._results.checkpoint.records)
        if route_index == record_len:
            self._results.checkpoint.records.append(route_record)
        elif route_index < record_len:
            self._results.checkpoint.records[route_index] = route_record
        else:
            raise ValueError("Not enough entries in the route record")

    def compute_global_statistics(self):
        """Computes and saves the global statistics of the routes"""
        def get_infractions_value(route_record, key):
            # Special case for the % based criteria. Extract the meters from the message. Very ugly, but it works
            if key == PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]:
                if not route_record.infractions[key]:
                    return 0.0
                return float(route_record.infractions[key][0].split(" ")[8])/1000

            return len(route_record.infractions[key])

        global_record = GlobalRecord()
        global_result = global_record.status

        route_records = self._results.checkpoint.records

        # init global cbv data
        total_cbv_data = {
            k: v
            for key, BINs in CBV_DATA_BINS.items()
            for k, v in {
                f"{key}_point": 0,
                f"{key}_distribution": {f"{BINs[i]}~{BINs[i+1]}": 0 for i in range(len(BINs)-1)}
            }.items()
        }

        # init global ego data (flattened key version)
        total_ego_data = {}

        for key, BINs in EGO_DATA_BINS.items():
            total_ego_data[f"{key}_distribution"] = {
                f"speed{EGO_SPEED_BINS[j]}~{EGO_SPEED_BINS[j+1]}_{key}{BINs[i]}~{BINs[i+1]}": 0
                for j in range(len(EGO_SPEED_BINS) - 1)
                for i in range(len(BINs) - 1)
            }
            total_ego_data[f"{key}_point"] = 0

        # Calculate the score's means and result
        for route_record in route_records:

            global_record.scores_mean['score_route'] += route_record.scores['score_route'] / self._total_routes
            global_record.scores_mean['score_penalty'] += route_record.scores['score_penalty'] / self._total_routes
            global_record.scores_mean['score_composed'] += route_record.scores['score_composed'] / self._total_routes

            global_record.meta['total_length'] += route_record.meta['route_length']
            global_record.meta['duration_game'] += route_record.meta['duration_game']
            global_record.meta['duration_system'] += route_record.meta['duration_system']
            global_record.meta['cbv_total_game_time'] += route_record.meta['cbv_total_game_time']
            global_record.meta['cbv_off_road_game_time'] += route_record.meta['cbv_off_road_game_time']
            global_record.meta['cbv_uncomfortable_game_time'] += route_record.meta['cbv_uncomfortable_game_time']
            global_record.meta['cbv_progress'] += route_record.meta['cbv_progress']
            global_record.meta['cbv_collision_count'] += route_record.meta['cbv_collision_count']
            global_record.meta['cbv_count'] += route_record.meta['cbv_count']
            global_record.meta['cbv_reach_goal_count'] += route_record.meta['cbv_reach_goal_count']

            # Update the global cbv data
            for key in CBV_DATA_BINS.keys():
                for data_bin, count in route_record.meta[f'cbv_{key}_distribution'].items():
                    total_cbv_data[f"{key}_distribution"][data_bin] += count
                    total_cbv_data[f"{key}_point"] += count

            # update global ego data
            for key in EGO_DATA_BINS.keys():
                for combo_key, count in route_record.meta[f'ego_{key}_distribution'].items():
                    total_ego_data[f"{key}_distribution"][combo_key] += count
                    total_ego_data[f"{key}_point"] += count

            # Downgrade the global result if need be ('Perfect' -> 'Completed' -> 'Failed'), and record the failed routes
            route_result = 'Failed' if 'Failed' in route_record.status else route_record.status
            if route_result == 'Failed':
                global_record.meta['exceptions'].append((route_record.route_id,
                                                         route_record.index,
                                                         route_record.status))
                global_result = route_result
            elif global_result == 'Perfect' and route_result != 'Perfect':
                global_result = route_result

        for item in global_record.scores_mean:
            global_record.scores_mean[item] = round(global_record.scores_mean[item], ROUND_DIGITS_SCORE)
        global_record.status = global_result

        # Calculate the score's standard deviation
        if self._total_routes == 1:
            for key in global_record.scores_std_dev:
                global_record.scores_std_dev[key] = 0
        else:
            for route_record in route_records:
                for key in global_record.scores_std_dev:
                    diff = route_record.scores[key] - global_record.scores_mean[key]
                    global_record.scores_std_dev[key] += math.pow(diff, 2)

            for key in global_record.scores_std_dev:
                value = round(math.sqrt(global_record.scores_std_dev[key] / float(self._total_routes - 1)), ROUND_DIGITS)
                global_record.scores_std_dev[key] = value

        # Calculate the number of infractions per km
        km_driven = 0
        for route_record in route_records:
            km_driven += route_record.meta['route_length'] / 1000 * route_record.scores['score_route'] / 100
            for key in global_record.infractions:
                global_record.infractions[key] += get_infractions_value(route_record, key)
        km_driven = max(km_driven, 0.001)

        for key in global_record.infractions:
            # Special case for the % based criteria.
            if key != PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]:
                global_record.infractions[key] /= km_driven
            global_record.infractions[key] = round(global_record.infractions[key], ROUND_DIGITS)

        # Update the global CBV data
        for key in CBV_DATA_BINS.keys():
            total_data_distribution = total_cbv_data[f"{key}_distribution"]
            total_data_points = total_cbv_data[f"{key}_point"]

            total_bin_data = {
                data_bin: ((bin_min + bin_max) / 2, ((bin_min + bin_max) / 2) ** 2)
                for data_bin in total_data_distribution.keys()
                for bin_min, bin_max in [map(float, data_bin.split('~'))]
            }

            if total_data_points > 0:
                mean = sum(bin_mid * count for data_bin, count in total_data_distribution.items()
                                for bin_mid, _ in [total_bin_data[data_bin]]) / total_data_points
                variance = sum(bin_mid_squared * count for data_bin, count in total_data_distribution.items()
                                    for _, bin_mid_squared in [total_bin_data[data_bin]]) / total_data_points

                variance_term = variance - (mean ** 2)
                if variance_term < 0:
                    variance_term = 0

                global_record.meta[f'cbv_{key}_mean'] = round(mean, ROUND_DIGITS)
                global_record.meta[f'cbv_{key}_std'] = round(math.sqrt(variance_term), ROUND_DIGITS)
                global_record.meta[f'cbv_{key}_distribution'] = total_data_distribution

        # Update the global ego data
        for key in EGO_DATA_BINS.keys():
            total_data_distribution = total_ego_data[f"{key}_distribution"]
            total_data_points = total_ego_data[f"{key}_point"]

            total_bin_data = {}

            for combo_key in total_data_distribution.keys():
                try:
                    _, data_part = combo_key.split('_', 1)
                    data_bin = data_part[len(key):]  # remove the key prefix
                    bin_min, bin_max = map(float, data_bin.split('~'))
                except Exception as e:
                    print(f"[Warning] Failed to parse bin from {combo_key}: {e}")
                    continue

                bin_mid = (bin_min + bin_max) / 2
                bin_mid_squared = bin_mid ** 2
                total_bin_data[combo_key] = (bin_mid, bin_mid_squared)

            if total_data_points > 0:
                sum_mean = 0.0
                sum_variance = 0.0

                for combo_key, count in total_data_distribution.items():
                    if combo_key not in total_bin_data:
                        continue
                    bin_mid, bin_mid_squared = total_bin_data[combo_key]
                    sum_mean += bin_mid * count
                    sum_variance += bin_mid_squared * count

                mean = sum_mean / total_data_points
                variance = sum_variance / total_data_points
                variance_term = variance - (mean ** 2)
                variance_term = max(0, variance_term)

                global_record.meta[f'ego_{key}_mean'] = round(mean, ROUND_DIGITS)
                global_record.meta[f'ego_{key}_std'] = round(math.sqrt(variance_term), ROUND_DIGITS)
                global_record.meta[f'ego_{key}_distribution'] = total_data_distribution
        

        # Save the global records
        self._results.checkpoint.global_record = global_record

        # Change the values and labels. These MUST HAVE A MATCHING ORDER
        self._results.values = [
            str(global_record.scores_mean['score_composed']),
            str(global_record.scores_mean['score_route']),
            str(global_record.scores_mean['score_penalty']),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_PEDESTRIAN]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_VEHICLE]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_STATIC]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.TRAFFIC_LIGHT_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.STOP_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.ROUTE_DEVIATION]]),
            str(global_record.infractions['route_timeout']),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.VEHICLE_BLOCKED]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.SCENARIO_TIMEOUT]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.MIN_SPEED_INFRACTION]]),
        ]
        self._results.values += [
            str(value) for value in global_record.meta.values()
        ]

        self._results.labels = [
            "Avg. driving score",
            "Avg. route completion",
            "Avg. infraction penalty",
            "Collisions with pedestrians",
            "Collisions with vehicles",
            "Collisions with layout",
            "Red lights infractions",
            "Stop sign infractions",
            "Off-road infractions",
            "Route deviations",
            "Route timeouts",
            "Agent blocked",
            "Yield emergency vehicles infractions",
            "Scenario timeouts",
            "Min speed infractions",
        ]
        self._results.labels += [
            key for key in global_record.meta
        ]

        # Change the entry status and eligible
        entry_status = 'Finished'
        for route_record in route_records:
            route_status = route_record.status
            if 'Simulation crashed' in route_status:
                entry_status = 'Crashed'
            elif "Agent's sensors were invalid" in route_status:
                entry_status = 'Rejected'

        self.save_entry_status(entry_status)

    def validate_and_write_statistics(self, sensors_initialized=False, crashed=False):
        """
        Makes sure that all the relevant data is there.
        Changes the 'entry status' to 'Invalid' if this isn't the case
        """
        error_message = ""
        if sensors_initialized and not self._results.sensors:
            error_message = "Missing 'sensors' data"

        elif not self._results.values:
            error_message = "Missing 'values' data"

        elif self._results.entry_status == 'Started':
            error_message = "'entry_status' has the 'Started' value"

        else:
            global_records = self._results.checkpoint.global_record
            progress = self._results.checkpoint.progress
            route_records = self._results.checkpoint.records

            if not global_records:
                error_message = "Missing 'global_records' data"

            elif not progress:
                error_message = "Missing 'progress' data"

            elif not crashed and (progress[0] != progress[1] or progress[0] != len(route_records)):
                error_message = "'progress' data doesn't match its expected value"

            else:
                for record in route_records:
                    if record.status == 'Started':
                        error_message = "Found a route record with missing data"
                        break

        if error_message:
            print("\n\033[91mThe statistics are badly formed. Setting their status to 'Invalid':")
            print("> {}\033[0m\n".format(error_message))

            self.save_entry_status('Invalid')

        self.write_statistics()

    def write_statistics(self):
        """
        Writes the results into the endpoint. Meant to be used only for partial evaluations,
        use 'validate_and_write_statistics' for the final one as it only validates the data.
        """
        save_dict(self._endpoint, self._results.to_json())
