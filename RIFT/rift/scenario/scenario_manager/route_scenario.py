#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : route_scenario.py
@Date    : 2023/10/4
"""
from collections import defaultdict
from typing import Dict
import py_trees
import carla

from rift.cbv.recognition.base_cbv import BaseCBVRecog
from rift.gym_carla.utils.common import filter_spawn_points
from rift.scenario.tools.atomic_criteria import ActorBlockedTest, MinimumSpeedRouteTest, OutsideRouteLanesTest
from rift.scenario.tools.timer import TimeOut
from rift.scenario.tools.carla_data_provider import CarlaDataProvider

from rift.scenario.tools.route_manipulation import interpolate_trajectory
from rift.gym_carla.action.cbv_action import CBVAction
from rift.gym_carla.action.ego_action import EgoAction
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.scenario.tools.scenario_utils import (
    convert_transform_to_location
)
from rift.scenario.tools.atomic_criteria import (
    Criterion,
    CollisionTest,
    InRouteTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
)


class RouteScenario():
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(
        self,
        world,
        config: RouteScenarioConfiguration,
        env_id: int,
        env_params: Dict,
        cbv_recog_policy: BaseCBVRecog,
        EgoAction: EgoAction,
        CBVAction: CBVAction,
        logger
    ):
        self.world = world
        self.logger = logger
        self.config = config
        self.env_id = env_id
        self.mode = env_params['mode']
        self.timeout = 40
        self.spawn_radius = cbv_recog_policy.spawn_radius
        self.activate_radius = cbv_recog_policy.activate_radius
        self.coord_interval = cbv_recog_policy.coord_interval
        self.traffic_intensity = cbv_recog_policy.traffic_intensity

        # Action
        self.EgoAction = EgoAction
        self.CBVAction = CBVAction

        # create the route and ego's position (the start point of the route)
        self.route, self.ego_vehicle, self.gps_route = self._update_route_and_ego(timeout=self.timeout)
        self.global_route_waypoints, self.global_route_info = self._global_route_to_waypoints()
        self.unactivated_actors = []
        
        # create the scenario tree
        self.scenario_tree = py_trees.composites.Parallel(config.name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # create the criteria tree
        self.criteria_tree = self._create_test_criteria()

        self.scenario_tree.add_child(self.criteria_tree)

        # Create the timeout behavior
        self.timeout_node = self._create_timeout_behavior()
        if self.timeout_node:
            self.scenario_tree.add_child(self.timeout_node)
        self.scenario_tree.setup(timeout=1)

    def _global_route_to_waypoints(self):
        waypoints_list = []
        waypoints_info = []
        carla_map = CarlaDataProvider.get_map()
        for node in self.route:
            loc = node[0].location
            waypoint = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            waypoints_list.append(waypoint)
            waypoints_info.append((waypoint.is_junction, waypoint.junction_id, waypoint.transform.location))
        return waypoints_list, waypoints_info

    def _update_route_and_ego(self, timeout=None):
        ego_vehicle = None
        route = None
        gps_route = None

        gps_route, route = interpolate_trajectory(self.config.keypoints)
        ego_vehicle = self._spawn_ego_vehicle(route[0][0])

        CarlaDataProvider.set_ego_vehicle_route(ego_vehicle, convert_transform_to_location(route), self.env_id)

        # Timeout of a scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle, gps_route

    def _estimate_route_timeout(self, route):
        route_length = 0.0  # in meters
        min_length = 100.0
        SECONDS_GIVEN_PER_METERS = 1

        if len(route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _spawn_ego_vehicle(self, elevate_transform, autopilot=False):
        elevate_transform.location.z += 0.2

        ego_vehicle = CarlaDataProvider.request_new_actor(
            'vehicle.lincoln.mkz_2017',
            elevate_transform,
            rolename='hero', 
            autopilot=autopilot
        )
        self.world.tick()

        return ego_vehicle

    def generate_spawn_points(self):
        start_loc = self.route[0][0].location
        mid_loc = self.route[len(self.route)//2][0].location
        
        groups = defaultdict(list)
        for wp_info in self.global_route_info:
            if wp_info[0]:  # is_junction
                groups[wp_info[1]].append(wp_info[2])  # append the location of the junction
        junction_loc_list = [wps[len(wps)//2] for wps in groups.values()]

        locations_list = [start_loc, mid_loc] + junction_loc_list
        radius_list = [self.spawn_radius] * len(locations_list)

        spawn_points = filter_spawn_points(
            locations_list, radius_list, coord_interval=self.coord_interval, intensity=self.traffic_intensity
        )
        amount = len(spawn_points)
        return amount, spawn_points

    def initialize_actors(self):
        amount, spawn_points = self.generate_spawn_points()
        # don't activate all the actors when initialization
        new_actors = CarlaDataProvider.request_new_batch_actors(
            model='vehicle.*',
            amount=amount,
            spawn_points=spawn_points,
            autopilot=False,
            random_location=False,
            rolename='background'
        )
        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")
        self.logger.log(f'>> Successfully spawning {len(new_actors)} Autopilot vehicles', color='green')
        self.unactivated_actors.extend(new_actors)
        CarlaDataProvider.set_scenario_actors(self.ego_vehicle, new_actors)

    def activate_background_actors(self):
        all_egos = CarlaDataProvider.get_all_ego_vehicles()
        ego_locations = [CarlaDataProvider.get_location(ego) for ego in all_egos]

        actor_locations = {actor: CarlaDataProvider.get_location(actor) for actor in self.unactivated_actors}
        activated_actors = []
        for actor, actor_location in actor_locations.items():
            if any(actor_location.distance(ego_location) < self.activate_radius for ego_location in ego_locations):
                actor.set_autopilot(True, CarlaDataProvider.get_traffic_manager_port())
                activated_actors.append(actor)

        for actor in activated_actors:
            self.unactivated_actors.remove(actor)

    def update_actions(self, ego_action, cbv_actions):
        # update ego action
        reverse, throttle, steer, brake = self.EgoAction.convert_action(ego_action, allow_reverse=False)
        carla_ego_action = carla.VehicleControl(reverse=reverse, throttle=float(throttle), steer=float(steer), brake=float(brake))
        self.ego_vehicle.apply_control(carla_ego_action)  # apply action of the ego vehicle on the next tick

        # add the ego control and the cbv control
        py_trees.blackboard.Blackboard().set("Ego_action", carla_ego_action, overwrite=True)

        # update cbv action
        if not cbv_actions:
            return

        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).items():
            cbv_action = cbv_actions[CBV_id]
            # convert raw action (acc, steer) to (throttle, steer, brake)
            a = self.CBVAction.convert_action(cbv_action)
            carla_cbv_action = carla.VehicleControl(reverse=a[0], throttle=float(a[1]), steer=float(a[2]), brake=float(a[3]))
            CBV.apply_control(carla_cbv_action)  # apply the control of the CBV on the next tick

    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(name="Criteria",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # the criteria needed both in training and evaluating
        criteria.add_child(RouteCompletionTest(self.ego_vehicle, route=self.route))
        criteria.add_child(OutsideRouteLanesTest(self.ego_vehicle, route=self.route))
        criteria.add_child(CollisionTest(self.ego_vehicle, name="CollisionTest"))
        criteria.add_child(ActorBlockedTest(self.ego_vehicle, min_speed=0.1, max_time=3.0, terminate_on_failure=True, name="AgentBlockedTest"))

        if self.mode == 'eval':
            # extra criteria for evaluating
            criteria.add_child(RunningRedLightTest(self.ego_vehicle))
            criteria.add_child(RunningStopTest(self.ego_vehicle))
            criteria.add_child(MinimumSpeedRouteTest(self.ego_vehicle, self.route, checkpoints=20, name="MinSpeedTest"))
            criteria.add_child(InRouteTest(self.ego_vehicle, route=self.route, offroad_max=30, terminate_on_failure=True))
        return criteria
    
    def _create_timeout_behavior(self):
        """
        Default initialization of the timeout behavior.
        Override this method in child class to provide custom initialization.
        """
        return TimeOut(self.timeout, name="TimeOut")  # Timeout node
    
    def get_criteria(self):
        """
        Return the list of test criteria, including all the leaf nodes.
        Some criteria might have trigger conditions, which have to be filtered out.
        """
        criteria = []
        if not self.criteria_tree:
            return criteria

        criteria_nodes = self._extract_nodes_from_tree(self.criteria_tree)
        for criterion in criteria_nodes:
            if isinstance(criterion, Criterion):
                criteria.append(criterion)

        return criteria

    def _extract_nodes_from_tree(self, tree):  # pylint: disable=no-self-use
        """
        Returns the list of all nodes from the given tree
        """
        node_list = [tree]
        more_nodes_exist = True
        while more_nodes_exist:
            more_nodes_exist = False
            for node in node_list:
                if node.children:
                    node_list.remove(node)
                    more_nodes_exist = True
                    for child in node.children:
                        node_list.append(child)

        if len(node_list) == 1 and isinstance(node_list[0], py_trees.composites.Parallel):
            return []

        return node_list

    def clean_up(self):
        node_list = self._extract_nodes_from_tree(self.scenario_tree)

        # Set status to INVALID
        for node in node_list:
            node.terminate(py_trees.common.Status.INVALID)

        # clean background vehicle (the vehicle will be destroyed in CarlaDataProvider)
        self.unactivated_actors = []


