#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : route_planner.py
@Date    : 2024/10/12
'''
import math
import random
from typing import Deque, Dict, List, Tuple
import carla


from agents.navigation.local_planner import RoadOption
from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
from rift.cbv.planning.route_planner.base_planner import CBVBasePlanner
from rift.ego.route_planner.route_planner import retrieve_options
from rift.scenario.tools.global_route_planner import GlobalRoutePlanner
from rift.gym_carla.visualization.visualize import draw_waypoints
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class CBVRoutePlanner(CBVBasePlanner):
    def __init__(self, env_params, hop_resolution=1.0, max_len=150, min_distance=3.0):
        super(CBVRoutePlanner, self).__init__(env_params)
        self._world = CarlaDataProvider.get_world()
        self._map = self._world.get_map()
        self.map_api: CarlaMap = CarlaDataProvider.get_map_api()
        self.hop_resolution = hop_resolution
        self.max_len = max_len
        self.min_distance = min_distance
        self.frame_rate = env_params['frame_rate']
        # init the global route planner
        self.global_route_planner: GlobalRoutePlanner = CarlaDataProvider.get_global_route_planner()

        self.CBVs_route_elements = None
        self.CBVs_route_ids = None
        self.CBVs_interaction_waypoints = None
        self.CBVs_reference_lines = None

    def reset(self, ego_vehicle, ego_global_route_waypoints: List[carla.Waypoint]):
        """
        Reset the CBV global routes
        """
        self.ego_vehicle = ego_vehicle
        self.ego_global_route_waypoints = ego_global_route_waypoints
        self.CBVs_route_elements = {}
        self.CBVs_route_ids = {}
        self.CBVs_interaction_waypoints = {}
        self.CBVs_reference_lines = {}

    def run_step(self, ego_local_route_waypoints: List[carla.Waypoint], ego_rest_route_waypoints: List[carla.Waypoint]):
        """
        Update the CBV global routes
        """

        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).items():
            if CBV_id not in self.CBVs_route_elements:
                new_route_elements, new_route_ids, interaction_wp = self.init_CBV_route(CBV)
                self.CBVs_route_ids[CBV_id] = new_route_ids  # only reset the route road ids in the initialization 
            else:
                new_route_elements, interaction_wp = self.update_CBV_route(CBV)

            self._check_CBVs_reach_goal(CBV, new_route_elements)
            
            self.CBVs_route_elements[CBV_id] = new_route_elements
            self.CBVs_interaction_waypoints[CBV_id] = interaction_wp
    
    def _check_CBVs_reach_goal(self, CBV, route_elements):
        """
        Check whether the CBV reach the goal
        """
        if route_elements is None or len(route_elements) == 0:
            return
        
        if 2 <= len(route_elements) < 10:
            CBV_trans = CarlaDataProvider.get_transform(CBV)
            CBV_forward_vector = CBV_trans.rotation.get_forward_vector()

            nearest_lane_wp = self._map.get_waypoint(CBV_trans.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if nearest_lane_wp is None:
                return
            
            nearest_lane_wp_forward_vector = nearest_lane_wp.transform.rotation.get_forward_vector()
            delta_angle = math.degrees(CBV_forward_vector.get_vector_angle(nearest_lane_wp_forward_vector))
            if delta_angle < 30:
                # for smoother freeing the CBV
                CarlaDataProvider.CBV_reach_goal(self.ego_vehicle, CBV)
        elif len(route_elements) < 2:
            CarlaDataProvider.CBV_reach_goal(self.ego_vehicle, CBV)

    def init_CBV_route(self, CBV) -> Deque[carla.Waypoint]:
        """
        Init the global route for new CBV
        """
        # find the interaction waypoint and the corresponding route waypoints
        inter_waypoint, inter_route_elements, inter_route_ids, inter_CBV_dis = self.find_interaction_waypoint(CBV)
        # expand the route elements
        CBV_route_elements, route_road_ids = self._add_extra_waypoints(inter_route_elements, inter_route_ids, inter_CBV_dis)

        return CBV_route_elements, route_road_ids, inter_waypoint

    def update_CBV_route(self, CBV) -> Tuple[Deque[Tuple[carla.Waypoint, RoadOption]], carla.Waypoint]:
        """
        Update the CBV route for exist CBV
        """
        # check whether the current CBVs_route_elements got too closed waypoints, if so, remove the too closed waypoints
        CBVs_route_elements = self.CBVs_route_elements[CBV.id]
        CBV_intercation_waypoint = self.CBVs_interaction_waypoints[CBV.id]

        CBV_trans = CarlaDataProvider.get_transform(CBV)
        max_index = -1
        for idx, (wp, road_option) in enumerate(CBVs_route_elements):
            if self.is_over_distance_ahead(wp.transform, CBV_trans, self.min_distance):
                max_index = idx
                break
        if max_index >= 0:
            for i in range(max_index - 1):
                if len(CBVs_route_elements) >= 2:
                    CBVs_route_elements.popleft()  # remove the already passed waypoints

        return CBVs_route_elements, CBV_intercation_waypoint
    
    def update_CBV(self, ego_rest_route_waypoints: List[carla.Waypoint]):
        """
        After changing the CBV, update the CBV global routes
        """
        # Gather current CBV ids from CarlaDataProvider
        current_CBVs = CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id)
        current_CBV_ids = set(current_CBVs.keys())

        # Initialize route for new CBVs
        for CBV_id, CBV in current_CBVs.items():
            if CBV_id not in self.CBVs_route_elements:
                self.CBVs_route_elements[CBV_id], self.CBVs_route_ids[CBV_id], self.CBVs_interaction_waypoints[CBV_id] = self.init_CBV_route(CBV)
        
        # Remove old CBVs which have been destroyed or moved back to BV
        keys_to_remove = self.CBVs_route_elements.keys() - current_CBV_ids
        for CBV_id in keys_to_remove:
            del self.CBVs_route_elements[CBV_id]
            del self.CBVs_route_ids[CBV_id]
            del self.CBVs_interaction_waypoints[CBV_id]

    def get_rest_route(self, CBV) -> Tuple[List[Tuple[carla.Waypoint, RoadOption]], Dict[str, List[int]], carla.Waypoint]:
        # get specific CBV's route elements and route road ids set
        return self.CBVs_route_elements[CBV.id], self.CBVs_route_ids[CBV.id], self.CBVs_interaction_waypoints[CBV.id]
    
    def build_reference_line(self, CBV, current_state, max_length):
        route_elements, route_ids, interaction_wp = self.get_rest_route(CBV)
        current_waypoints = [wp for wp, _ in route_elements]
        reference_lines = self.map_api.query_reference_lines(
            current_waypoints=current_waypoints,
            current_state=current_state,
            route_ids=route_ids,
            max_length=max_length,
        )
        self.CBVs_reference_lines[CBV.id] = reference_lines
        return reference_lines, route_elements, route_ids, interaction_wp

    def get_reference_line(self, CBV):
        return self.CBVs_reference_lines[CBV.id]

    def _add_extra_waypoints(self, route_elements: List[Tuple[carla.Waypoint, RoadOption]], route_ids: Dict[str, List[int]], CBV_dis=20):
        """
        Add new waypoints to the waypoint trajectory queue, create global route longer than 110 meters
        Parameters:
        - CBV_dis: the current CBV route distance
        """
        k = int((self.max_len + 10 - CBV_dis) // self.hop_resolution)
        next_waypoint = None
        # add k more waypoints
        for _ in range(k):
            last_waypoint = route_elements[-1][0]
            next_waypoints = list(last_waypoint.next(self.hop_resolution))
            # handle the case that the next_waypoint is None
            if not next_waypoints:
                break
            
            if len(next_waypoints) == 1:
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            route_elements.append((next_waypoint, road_option))
            route_ids['road_ids'].append(next_waypoint.road_id)
            route_ids['lane_ids'].append(next_waypoint.lane_id)
        
        # add more waypoints incase the last waypoint is in the junction
        while next_waypoint and next_waypoint.is_junction:
            temp_waypoints = list(next_waypoint.next(self.hop_resolution))
            # handle the case that the next_waypoint is None
            if not temp_waypoints:
                break

            if len(temp_waypoints) == 1:
                temp_waypoint = temp_waypoints[0]
                temp_road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                temp_road_options_list = retrieve_options(
                    temp_waypoints, next_waypoint)
                temp_road_option = random.choice(temp_road_options_list)
                temp_waypoint = temp_waypoints[temp_road_options_list.index(
                    temp_road_option)]
            route_elements.append((temp_waypoint, temp_road_option))
            route_ids['road_ids'].append(temp_waypoint.road_id)
            route_ids['lane_ids'].append(temp_waypoint.lane_id)
            next_waypoint = temp_waypoint
        
        return route_elements, route_ids

    def find_interaction_waypoint(self, CBV) -> Tuple[carla.Waypoint, Deque[Tuple[carla.Waypoint, RoadOption]]]:
        inter_info = self.inter_info.get(CBV.id, None)
        if inter_info is not None:
            return inter_info['inter_waypoint'], inter_info['inter_CBV_route_elements'], inter_info['inter_CBV_route_ids'], inter_info['inter_CBV_dis']
        else:
            raise ValueError('The interaction information is not found in the inter_info dict')

    def compute_route_distance(self, route_waypoints, down_sample_step=1):
        """
        Calculate the total distance of a route, considering every 'down_sample_step' waypoints.

        Parameters:
        - route_waypoints: List of waypoints forming the route.
        - down_sample_step: Downsampling factor; distance is computed between waypoints every 'down_sample_step' apart.
        """
        distance = 0.0
        for i in range(0, len(route_waypoints) - down_sample_step, down_sample_step):
            loc1 = route_waypoints[i].transform.location
            loc2 = route_waypoints[i + down_sample_step].transform.location
            segment_distance = loc1.distance(loc2)
            distance += segment_distance
        return distance
    
    def is_over_distance_ahead(self, target_transform, current_transform, min_distance, angle=90):
        """
            Check if a target object is over a certain distance in front of a reference object.
        """
        target_loc = target_transform.location
        current_loc = current_transform.location

        relative_direction = (target_loc - current_loc)
        current_forward_vector = current_transform.rotation.get_forward_vector()
        # Compute the forward distance by projecting the relative direction onto the forward vector
        forward_distance = (relative_direction.x * current_forward_vector.x +
                        relative_direction.y * current_forward_vector.y +
                        relative_direction.z * current_forward_vector.z)
        delta_angle = math.degrees(current_forward_vector.get_vector_angle(relative_direction))

        return (delta_angle < angle) and (forward_distance > min_distance)

    def vis_route(self):
        for route_element in self.CBVs_route_elements.values():
            # draw all the CBV's waypoints route
            draw_waypoints(self._world, route_element[0], max_length=40, frame_rate=self.frame_rate)
