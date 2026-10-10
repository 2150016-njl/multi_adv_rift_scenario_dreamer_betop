#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : autopilot.py
@Date    : 2025/01/14
'''

"""
PDM-Lite Agent
Source: https://github.com/OpenDriveLab/DriveLM/blob/DriveLM-CARLA/pdm_lite/
"""

import os
import pathlib

import math
import numpy as np
import carla
from rift.scenario.tools.route_manipulation import downsample_route
from rift.ego.pdm_lite import transfuser_utils as t_u

from scipy.integrate import RK45
from collections import deque
from agents.navigation.local_planner import RoadOption

from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.ego.pdm_lite.nav_planner import RoutePlanner
from rift.ego.pdm_lite.lateral_controller import LateralPIDController
from rift.ego.pdm_lite.privileged_route_planner import PrivilegedRoutePlanner
from rift.ego.pdm_lite.config import GlobalConfig
from rift.ego.pdm_lite.longitudinal_controller import LongitudinalLinearRegressionController
from rift.ego.pdm_lite.kinematic_bicycle_model import KinematicBicycleModel


def calculate_velocity(velocity):
    """
    Method to calculate the velocity of a actor
    """
    velocity_squared = velocity.x**2
    velocity_squared += velocity.y**2
    return math.sqrt(velocity_squared)


class AutoPilot():
    """
      Privileged driving agent used for data collection.
      Drives by accessing the simulator directly.
      """

    def __init__(self, target_speed=7, frame_rate=10):
        """
        Set up the autonomous agent for the CARLA simulation.
        """
        self.step = -1

        self.config = GlobalConfig()
        self.config.target_speed_fast = target_speed
        self.config.target_speed_slow = target_speed - 3
        self.config.lateral_pid_kp = 2.0  # original lateral pid kp will cause oscillation in lane changing
        self.config.lateral_pid_window_size = 20
        self.config.carla_frame_rate = frame_rate
        self.config.carla_fps = frame_rate
        self.config.fps = frame_rate
        self.config.bicycle_frame_rate = frame_rate

        # Dynamics models
        self.ego_model = KinematicBicycleModel(self.config)
        self.vehicle_model = KinematicBicycleModel(self.config)

        # Configuration
        self.visualize = 0

        self.walker_close = False
        self.distance_to_walker = np.inf
        self.stop_sign_close = False

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.ego_blocked_for_ticks = 0

        # Controllers
        self._turn_controller = LateralPIDController(self.config)

        self.list_traffic_lights = []

        # Navigation command buffer, needed because the correct command comes from the last cleared waypoint
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.next_commands = deque(maxlen=2)
        self.next_commands.append(4)
        self.next_commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_speed = self.config.target_speed_fast

        self.augmentation_translation = 0
        self.augmentation_rotation = 0

        # Angle to the next waypoint, normalized in [-1, 1] corresponding to [-90, 90]
        self.angle = 0.0
        self.stop_sign_hazard = False
        self.traffic_light_hazard = False
        self.walker_hazard = False
        self.vehicle_hazard = False
        self.junction = False
        self.aim_wp = None  # Waypoint the expert is steering towards
        self.remaining_route = None  # Remaining route
        self.remaining_route_original = None  # Remaining original route
        self.close_traffic_lights = []
        self.close_stop_signs = []
        self.was_at_stop_sign = False
        self.cleared_stop_sign = False

        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

    def set_planner(self, ego_vehicle, global_plan_gps, global_plan_world_coord):
        # Get the world map and the ego vehicle
        self.world_map = CarlaDataProvider.get_map()
        # set global plan
        self.set_global_plan(global_plan_gps, global_plan_world_coord)

        # Get the hero vehicle and the CARLA world
        self._vehicle = ego_vehicle
        self._world = CarlaDataProvider.get_world()

        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(CarlaDataProvider.get_location(self._vehicle))
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2

        # Set up the route planner and extrapolation
        self._waypoint_planner = PrivilegedRoutePlanner(self.config)
        self._waypoint_planner.setup_route(self.org_dense_route_world_coord, self._world, self.world_map,
                                           starts_with_parking_exit, CarlaDataProvider.get_location(self._vehicle))
        self._waypoint_planner.save()

        # Set up the longitudinal controller and command planner
        self._longitudinal_controller = LongitudinalLinearRegressionController(self.config)
        self._command_planner = RoutePlanner(self.config.route_planner_min_distance,
                                             self.config.route_planner_max_distance)
        self._command_planner.set_route(self._global_plan_world_coord)

        # Preprocess traffic lights
        all_actors = self._world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = t_u.get_traffic_light_waypoints(actor, self.world_map)
                self.list_traffic_lights.append((actor, center, waypoints))

        # Remove bugged 2-wheelers
        # https://github.com/carla-simulator/carla/issues/3670
        for actor in all_actors:
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()


    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self.org_dense_route_gps = global_plan_gps
        self.org_dense_route_world_coord = global_plan_world_coord
        ds_ids = downsample_route(global_plan_world_coord, 200)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]


    def tick_autopilot(self, input_data=None):
        """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        Args:
            input_data (dict): Input data containing sensor information.

        Returns:
            dict: A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
        # Get the vehicle's speed from its velocity vector
        speed = CarlaDataProvider.get_velocity(self._vehicle).length()

        trans = CarlaDataProvider.get_transform(self._vehicle)
        loc = trans.location
        gps = np.array([loc.x, loc.y, loc.z])
        compass = np.deg2rad(trans.rotation.yaw)

        # Create a dictionary containing the vehicle's state
        vehicle_state = {
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        return vehicle_state

    def run_step(self, input_data=None, plant=False):
        """
        Run a single step of the agent's control loop.

        Args:
            input_data (dict): Input data for the current step.
            plant (bool, optional): Flag indicating whether to run the plant simulation or not. Default is False.

        Returns:
            returns the control commands (steer, throttle, brake).
        """
        self.step += 1

        # Get the control commands and driving data for the current step
        control = self._get_control(input_data, plant)

        return control

    def _get_control(self, input_data=None, plant=False):
        """
        Compute the control commands and save the driving data for the current frame.

        Args:
            input_data (dict): Input data for the current frame.
            plant (object): The plant object representing the vehicle dynamics.

        Returns:
            tuple: A tuple containing the control commands (steer, throttle, brake) and the driving data.
        """
        tick_data = self.tick_autopilot(input_data)
        ego_position = tick_data["gps"]

        # Waypoint planning and route generation
        route_np, route_wp, _, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign,\
                                        next_stop_sign, speed_limit = self._waypoint_planner.run_step(ego_position)

        # Extract relevant route information
        self.remaining_route = route_np[self.config.tf_first_checkpoint_distance:][::self.config.points_per_meter]
        self.remaining_route_original = self._waypoint_planner.original_route_points[
            self._waypoint_planner.route_index:][self.config.tf_first_checkpoint_distance:][::self.config.
                                                                                            points_per_meter]

        # Get the current speed and target speed
        ego_speed = tick_data["speed"]
        target_speed = speed_limit * self.config.ratio_target_speed_limit

        # Reduce target speed if there is a junction ahead
        for i in range(min(self.config.max_lookahead_to_check_for_junction, len(route_wp))):
            if route_wp[i].is_junction:
                target_speed = min(target_speed, self.config.max_speed_in_junction)
                break

        # Get the list of vehicles in the scene
        actors = self._world.get_actors()
        vehicles = list(actors.filter("*vehicle*"))

        # # Manage route obstacle scenarios and adjust target speed
        # target_speed_route_obstacle, keep_driving, speed_reduced_by_obj = self._manage_route_obstacle_scenarios(
        #     target_speed, ego_speed, route_wp, vehicles, route_np)

        # # In case the agent overtakes an obstacle, keep driving in case the opposite lane is free instead of using idm
        # # and the kinematic bicycle model forecasts
        # if keep_driving:
        #     brake, target_speed = False, target_speed_route_obstacle
        # else:
        #     brake, target_speed, speed_reduced_by_obj = self.get_brake_and_target_speed(
        #         plant, route_np, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign,
        #         next_stop_sign, vehicles, actors, target_speed, speed_reduced_by_obj)

        # target_speed = min(target_speed, target_speed_route_obstacle)

        speed_reduced_by_obj = None

        brake, target_speed, speed_reduced_by_obj = self.get_brake_and_target_speed(
            plant, route_np, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign,
            next_stop_sign, vehicles, actors, target_speed, speed_reduced_by_obj)

        # Determine if the ego vehicle is at a junction
        ego_vehicle_waypoint = self.world_map.get_waypoint(CarlaDataProvider.get_location(self._vehicle))
        self.junction = ego_vehicle_waypoint.is_junction

        # Compute throttle and brake control
        throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(brake, target_speed, ego_speed)

        # Compute steering control
        steer = self._get_steer(route_np, ego_position, tick_data["compass"], ego_speed)

        # Create the control command
        control = carla.VehicleControl()
        control.steer = steer + self.config.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake or control_brake)

        # Apply brake if the vehicle is stopped to prevent rolling back
        if control.throttle == 0 and ego_speed < self.config.minimum_speed_to_prevent_rolling_back:
            control.brake = 1

        # Apply throttle if the vehicle is blocked for too long
        ego_velocity = calculate_velocity(CarlaDataProvider.get_velocity(self._vehicle))
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0

        if self.ego_blocked_for_ticks >= self.config.max_blocked_ticks:
            control.throttle = 1
            control.brake = 0

        # Save control commands and target speed
        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self.target_speed = target_speed

        # Get the target and next target points from the command planner
        command_route = self._command_planner.run_step(ego_position)
        if len(command_route) > 2:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[2]
        elif len(command_route) > 1:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[1]
        else:
            target_point, far_command = command_route[0]
            next_target_point, next_far_command = command_route[0]

        # Update command history and save driving datas
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)
            self.next_commands.append(next_far_command.value)

        return control

    def _get_steer(self, route_points, current_position, current_heading, current_speed):
        """
        Calculate the steering angle based on the current position, heading, speed, and the route points.

        Args:
            route_points (numpy.ndarray): An array of (x, y) coordinates representing the route points.
            current_position (tuple): The current position (x, y) of the vehicle.
            current_heading (float): The current heading angle (in radians) of the vehicle.
            current_speed (float): The current speed of the vehicle (in m/s).

        Returns:
            float: The calculated steering angle.
        """
        speed_scale = self.config.lateral_pid_speed_scale
        speed_offset = self.config.lateral_pid_speed_offset

        # Calculate the lookahead index based on the current speed
        speed_in_kmph = current_speed * 3.6
        lookahead_distance = speed_scale * speed_in_kmph + speed_offset
        lookahead_distance = np.clip(lookahead_distance, self.config.lateral_pid_default_lookahead,
                                     self.config.lateral_pid_maximum_lookahead_distance)
        lookahead_index = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Get the target point from the route points
        target_point = route_points[lookahead_index]

        # Calculate the angle between the current heading and the target point
        angle_unnorm = self._get_angle_to(current_position, current_heading, target_point)
        normalized_angle = angle_unnorm / 90

        self.aim_wp = target_point
        self.angle = normalized_angle

        # Calculate the steering angle using the turn controller
        steering_angle = self._turn_controller.step(route_points, current_speed, current_position, current_heading)
        steering_angle = round(steering_angle, 3)

        return steering_angle

    def _compute_target_speed_idm(self,
                                  desired_speed,
                                  leading_actor_length,
                                  ego_speed,
                                  leading_actor_speed,
                                  distance_to_leading_actor,
                                  s0=4.,
                                  T=0.5):
        """
        Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

        Args:
            desired_speed (float): The desired speed of the ego vehicle.
            leading_actor_length (float): The length of the leading actor (vehicle or obstacle).
            ego_speed (float): The current speed of the ego vehicle.
            leading_actor_speed (float): The speed of the leading actor.
            distance_to_leading_actor (float): The distance to the leading actor.
            s0 (float, optional): The minimum desired net distance.
            T (float, optional): The desired time headway.

        Returns:
            float: The computed target speed for the ego vehicle.
        """
        a = self.config.idm_maximum_acceleration # Maximum acceleration [m/s²]
        b = self.config.idm_comfortable_braking_deceleration_high_speed if ego_speed > \
                        self.config.idm_comfortable_braking_deceleration_threshold else \
                        self.config.idm_comfortable_braking_deceleration_low_speed # Comfortable deceleration [m/s²]
        delta = self.config.idm_acceleration_exponent # Acceleration exponent
        
        t_bound = self.config.idm_t_bound

        def idm_equations(t, x):
            """
            Differential equations for the Intelligent Driver Model.

            Args:
                t (float): Time.
                x (list): State variables [position, speed].

            Returns:
                list: Derivatives of the state variables.
            """
            ego_position, ego_speed = x

            speed_diff = ego_speed - leading_actor_speed
            s_star = s0 + ego_speed * T + ego_speed * speed_diff / 2. / np.sqrt(a * b)
            # The maximum is needed to avoid numerical unstabilities
            s = max(0.1, distance_to_leading_actor + t * leading_actor_speed - ego_position - leading_actor_length)
            dvdt = a * (1. - (ego_speed / desired_speed)**delta - (s_star / s)**2)

            return [ego_speed, dvdt]

        # Set the initial conditions
        y0 = [0., ego_speed]

        # Integrate the differential equations using RK45
        rk45 = RK45(fun=idm_equations, t0=0., y0=y0, t_bound=t_bound)
        while rk45.status == "running":
            rk45.step()

        # The target speed is the final speed obtained from the integration
        target_speed = rk45.y[1]

        # Clip the target speed to non-negative values
        return np.clip(target_speed, 0, np.inf)

    def is_near_lane_change(self, ego_velocity, route_points):
        """
        Computes if the ego agent is/was close to a lane change maneuver.

        Args:
            ego_velocity (float): The current velocity of the ego agent in m/s.
            route_points (numpy.ndarray): An array of locations representing the planned route.

        Returns:
            bool: True if the ego agent is close to a lane change, False otherwise.
        """
        # Calculate the braking distance based on the ego velocity
        braking_distance = ((
            (ego_velocity * 3.6) / 10.0)**2 / 2.0) + self.config.braking_distance_calculation_safety_distance

        # Determine the number of waypoints to look ahead based on the braking distance
        look_ahead_points = max(self.config.minimum_lookahead_distance_to_compute_near_lane_change,
                                min(route_points.shape[0], self.config.points_per_meter * int(braking_distance)))
        current_route_index = self._waypoint_planner.route_index
        max_route_length = len(self._waypoint_planner.commands)

        from_index = max(0, current_route_index - self.config.check_previous_distance_for_lane_change)
        to_index = min(max_route_length - 1, current_route_index + look_ahead_points)
        # Iterate over the points around the current position, checking for lane change commands
        for i in range(from_index, to_index, 1):
            if self._waypoint_planner.commands[i] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                return True

        return False

    def predict_other_actors_bounding_boxes(self, plant, actor_list, ego_vehicle_location, num_future_frames,
                                            near_lane_change):
        """
        Predict the future bounding boxes of actors for a given number of frames.

        Args:
            plant (bool): Whether to use PlanT.
            actor_list (list): A list of actors (e.g., vehicles) in the simulation.
            ego_vehicle_location (carla.Location): The current location of the ego vehicle.
            num_future_frames (int): The number of future frames to predict.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.

        Returns:
            dict: A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
        """
        predicted_bounding_boxes = {}

        if not plant:
            # Filter out nearby actors within the detection radius, excluding the ego vehicle
            nearby_actors = [
                actor for actor in actor_list if actor.id != self._vehicle.id and
                actor.get_location().distance(ego_vehicle_location) < self.config.detection_radius
            ]

            # If there are nearby actors, calculate their future bounding boxes
            if nearby_actors:
                # Get the previous control inputs (steering, throttle, brake) for the nearby actors
                previous_controls = [actor.get_control() for actor in nearby_actors]
                previous_actions = np.array(
                    [[control.steer, control.throttle, control.brake] for control in previous_controls])

                # Get the current velocities, locations, and headings of the nearby actors
                velocities = np.array([actor.get_velocity().length() for actor in nearby_actors])
                locations = np.array([[actor.get_location().x,
                                       actor.get_location().y,
                                       actor.get_location().z] for actor in nearby_actors])
                headings = np.deg2rad(np.array([actor.get_transform().rotation.yaw for actor in nearby_actors]))

                # Initialize arrays to store future locations, headings, and velocities
                future_locations = np.empty((num_future_frames, len(nearby_actors), 3), dtype="float")
                future_headings = np.empty((num_future_frames, len(nearby_actors)), dtype="float")
                future_velocities = np.empty((num_future_frames, len(nearby_actors)), dtype="float")

                # Forecast the future locations, headings, and velocities for the nearby actors
                for i in range(num_future_frames):
                    locations, headings, velocities = self.vehicle_model.forecast_other_vehicles(
                        locations, headings, velocities, previous_actions)
                    future_locations[i] = locations.copy()
                    future_velocities[i] = velocities.copy()
                    future_headings[i] = headings.copy()

                # Convert future headings to degrees
                future_headings = np.rad2deg(future_headings)

                # Calculate the predicted bounding boxes for each nearby actor and future frame
                for actor_idx, actor in enumerate(nearby_actors):
                    predicted_actor_boxes = []

                    for i in range(num_future_frames):
                        # Calculate the future location of the actor
                        location = carla.Location(x=future_locations[i, actor_idx, 0].item(),
                                                  y=future_locations[i, actor_idx, 1].item(),
                                                  z=future_locations[i, actor_idx, 2].item())

                        # Calculate the future rotation of the actor
                        rotation = carla.Rotation(pitch=0, yaw=future_headings[i, actor_idx], roll=0)

                        # Get the extent (dimensions) of the actor's bounding box
                        extent = actor.bounding_box.extent
                        # Otherwise we would increase the extent of the bounding box of the vehicle
                        extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                        # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                        # uncertainty during forecasting
                        s = self.config.high_speed_min_extent_x_other_vehicle_lane_change if near_lane_change \
                            else self.config.high_speed_min_extent_x_other_vehicle
                        extent.x *= self.config.slow_speed_extent_factor_ego if future_velocities[
                            i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                                s,
                                self.config.high_speed_min_extent_x_other_vehicle * float(i) / float(num_future_frames))
                        extent.y *= self.config.slow_speed_extent_factor_ego if future_velocities[
                            i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                                self.config.high_speed_min_extent_y_other_vehicle,
                                self.config.high_speed_extent_y_factor_other_vehicle * float(i) /
                                float(num_future_frames))

                        # Create the bounding box for the future frame
                        bounding_box = carla.BoundingBox(location, extent)
                        bounding_box.rotation = rotation

                        # Append the bounding box to the list of predicted bounding boxes for this actor
                        predicted_actor_boxes.append(bounding_box)

                    # Store the predicted bounding boxes for this actor in the dictionary
                    predicted_bounding_boxes[actor.id] = predicted_actor_boxes

                if self.visualize == 1:
                    for actor_idx, actors_forecasted_bounding_boxes in predicted_bounding_boxes.items():
                        for bb in actors_forecasted_bounding_boxes:
                            self._world.debug.draw_box(box=bb,
                                                       rotation=bb.rotation,
                                                       thickness=0.1,
                                                       color=self.config.other_vehicles_forecasted_bbs_color,
                                                       life_time=self.config.draw_life_time)

        return predicted_bounding_boxes

    def compute_target_speed_wrt_leading_vehicle(self, initial_target_speed, predicted_bounding_boxes, near_lane_change,
                                                 ego_location, rear_vehicle_ids, leading_vehicle_ids,
                                                 speed_reduced_by_obj, plant):
        """
        Compute the target speed for the ego vehicle considering the leading vehicle.

        Args:
            initial_target_speed (float): The initial target speed for the ego vehicle.
            predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            ego_location (carla.Location): The current location of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] 
                for the object that caused the most speed reduction, or None if no speed reduction.
            plant (bool): Whether to use plant.

        Returns:
            float: The target speed considering the leading vehicle.
        """
        target_speed_wrt_leading_vehicle = initial_target_speed

        if not plant:
            for vehicle_id, _ in predicted_bounding_boxes.items():
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    # Vehicle is in front of the ego vehicle
                    ego_speed = CarlaDataProvider.get_velocity(self._vehicle).length()
                    vehicle = self._world.get_actor(vehicle_id)
                    other_speed = vehicle.get_velocity().length()
                    distance_to_vehicle = ego_location.distance(vehicle.get_location())

                    # Compute the target speed using the IDM
                    target_speed_wrt_leading_vehicle = min(
                        target_speed_wrt_leading_vehicle,
                        self._compute_target_speed_idm(
                            desired_speed=initial_target_speed,
                            leading_actor_length=vehicle.bounding_box.extent.x * 2,
                            ego_speed=ego_speed,
                            leading_actor_speed=other_speed,
                            distance_to_leading_actor=distance_to_vehicle,
                            s0=self.config.idm_leading_vehicle_minimum_distance,
                            T=self.config.idm_leading_vehicle_time_headway
                        ))

                    # Update the object causing the most speed reduction
                    if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_wrt_leading_vehicle:
                        speed_reduced_by_obj = [
                            target_speed_wrt_leading_vehicle, vehicle.type_id, vehicle.id, distance_to_vehicle
                        ]

            if self.visualize == 1:
                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if vehicle_id in leading_vehicle_ids and not near_lane_change:
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0)
                        self._world.debug.draw_box(box=bb,
                                                   rotation=bb.rotation,
                                                   thickness=0.5,
                                                   color=self.config.leading_vehicle_color,
                                                   life_time=self.config.draw_life_time)
                    elif vehicle_id in rear_vehicle_ids:
                        vehicle = self._world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0)
                        self._world.debug.draw_box(box=bb,
                                                   rotation=bb.rotation,
                                                   thickness=0.5,
                                                   color=self.config.trailing_vehicle_color,
                                                   life_time=self.config.draw_life_time)

        return target_speed_wrt_leading_vehicle, speed_reduced_by_obj

    def compute_target_speeds_wrt_all_actors(self, initial_target_speed, ego_bounding_boxes, predicted_bounding_boxes,
                                             near_lane_change, leading_vehicle_ids, rear_vehicle_ids,
                                             speed_reduced_by_obj, nearby_walkers, nearby_walkers_ids):
        """
        Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles, 
        and pedestrians) by checking for intersecting bounding boxes.

        Args:
            initial_target_speed (float): The initial target speed for the ego vehicle.
            ego_bounding_boxes (list): A list of bounding boxes for the ego vehicle at different future frames.
            predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, 
                object ID, distance] for the object that caused the most speed reduction, or None if 
                no speed reduction.
            nearby_walkers (dict): A list of predicted bounding boxes of nearby pedestrians.
            nearby_walkers_ids (list): A list of IDs for nearby pedestrians.

        Returns:
            tuple: A tuple containing the target speeds for bicycles, pedestrians, vehicles, and the updated 
                speed_reduced_by_obj list.
        """
        target_speed_bicycle = initial_target_speed
        target_speed_pedestrian = initial_target_speed
        target_speed_vehicle = initial_target_speed
        ego_vehicle_location = CarlaDataProvider.get_location(self._vehicle)
        hazard_color = self.config.ego_vehicle_forecasted_bbs_hazard_color
        normal_color = self.config.ego_vehicle_forecasted_bbs_normal_color
        color = normal_color

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i, ego_bounding_box in enumerate(ego_bounding_boxes):
            for vehicle_id, bounding_boxes in predicted_bounding_boxes.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    continue
                elif vehicle_id in rear_vehicle_ids and not near_lane_change:
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = self.check_obb_intersection(ego_bounding_box, bounding_boxes[i])
                    ego_speed = CarlaDataProvider.get_velocity(self._vehicle).length()

                    if intersects_with_ego:
                        blocking_actor = self._world.get_actor(vehicle_id)

                        # Handle the case when the blocking actor is a bicycle
                        if "base_type" in blocking_actor.attributes and blocking_actor.attributes[
                                "base_type"] == "bicycle":
                            other_speed = blocking_actor.get_velocity().length()
                            distance_to_actor = ego_vehicle_location.distance(blocking_actor.get_location())

                            # Compute the target speed for bicycles using the IDM
                            target_speed_bicycle = min(
                                target_speed_bicycle,
                                self._compute_target_speed_idm(
                                    desired_speed=initial_target_speed,
                                    leading_actor_length=blocking_actor.bounding_box.extent.x * 2,
                                    ego_speed=ego_speed,
                                    leading_actor_speed=other_speed,
                                    distance_to_leading_actor=distance_to_actor,
                                    s0=self.config.idm_bicycle_minimum_distance,
                                    T=self.config.idm_bicycle_desired_time_headway
                                ))

                            # Update the object causing the most speed reduction
                            if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_bicycle:
                                speed_reduced_by_obj = [
                                    target_speed_bicycle, blocking_actor.type_id, blocking_actor.id, distance_to_actor
                                ]

                        # Handle the case when the blocking actor is not a bicycle
                        else:
                            self.vehicle_hazard = True  # Set the vehicle hazard flag
                            self.vehicle_affecting_id = vehicle_id  # Store the ID of the vehicle causing the hazard
                            color = hazard_color  # Change the following colors from green to red (no hazard to hazard)
                            target_speed_vehicle = 0  # Set the target speed for vehicles to zero
                            distance_to_actor = blocking_actor.get_location().distance(ego_vehicle_location)

                            # Update the object causing the most speed reduction
                            if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_vehicle:
                                speed_reduced_by_obj = [
                                    target_speed_vehicle, blocking_actor.type_id, blocking_actor.id, distance_to_actor
                                ]

            # Iterate over nearby pedestrians and check for intersections with the ego bounding box
            for pedestrian_bb, pedestrian_id in zip(nearby_walkers, nearby_walkers_ids):
                if self.check_obb_intersection(ego_bounding_box, pedestrian_bb[i]):
                    color = hazard_color
                    ego_speed = CarlaDataProvider.get_velocity(self._vehicle).length()
                    blocking_actor = self._world.get_actor(pedestrian_id)
                    distance_to_actor = ego_vehicle_location.distance(blocking_actor.get_location())

                    # Compute the target speed for pedestrians using the IDM
                    target_speed_pedestrian = min(
                        target_speed_pedestrian,
                        self._compute_target_speed_idm(
                            desired_speed=initial_target_speed,
                            leading_actor_length=0.5 + self._vehicle.bounding_box.extent.x,
                            ego_speed=ego_speed,
                            leading_actor_speed=0.,
                            distance_to_leading_actor=distance_to_actor,
                            s0=self.config.idm_pedestrian_minimum_distance,
                            T=self.config.idm_pedestrian_desired_time_headway
                        ))

                    # Update the object causing the most speed reduction
                    if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_pedestrian:
                        speed_reduced_by_obj = [
                            target_speed_pedestrian, blocking_actor.type_id, blocking_actor.id, distance_to_actor
                        ]

            if self.visualize == 1:
                self._world.debug.draw_box(box=ego_bounding_box,
                                           rotation=ego_bounding_box.rotation,
                                           thickness=0.1,
                                           color=color,
                                           life_time=self.config.draw_life_time)

        return target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, speed_reduced_by_obj

    def get_brake_and_target_speed(self, plant, route_points, distance_to_next_traffic_light, next_traffic_light,
                                   distance_to_next_stop_sign, next_stop_sign, vehicle_list, actor_list,
                                   initial_target_speed, speed_reduced_by_obj):
        """
        Compute the brake command and target speed for the ego vehicle based on various factors.

        Args:
            plant (bool): Whether to use PlanT.
            route_points (numpy.ndarray): An array of waypoints representing the planned route.
            distance_to_next_traffic_light (float): The distance to the next traffic light.
            next_traffic_light (carla.TrafficLight): The next traffic light actor.
            distance_to_next_stop_sign (float): The distance to the next stop sign.
            next_stop_sign (carla.StopSign): The next stop sign actor.
            vehicle_list (list): A list of vehicle actors in the simulation.
            actor_list (list): A list of all actors (vehicles, pedestrians, etc.) in the simulation.
            initial_target_speed (float): The initial target speed for the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] 
                    for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            tuple: A tuple containing the brake command (bool), target speed (float), and the updated 
                    speed_reduced_by_obj list.
        """
        ego_speed = CarlaDataProvider.get_velocity(self._vehicle).length()
        target_speed = initial_target_speed

        ego_vehicle_location = CarlaDataProvider.get_location(self._vehicle)
        ego_vehicle_transform = CarlaDataProvider.get_transform(self._vehicle)

        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(self._vehicle.bounding_box.location)
        ego_bb_global = carla.BoundingBox(center_ego_bb_global, self._vehicle.bounding_box.extent)
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        if self.visualize == 1:
            self._world.debug.draw_box(box=ego_bb_global,
                                       rotation=ego_bb_global.rotation,
                                       thickness=0.1,
                                       color=self.config.ego_vehicle_bb_color,
                                       life_time=self.config.draw_life_time)

        # Reset hazard flags
        self.stop_sign_close = False
        self.walker_close = False
        self.walker_close_id = None
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False
        self.walker_affecting_id = None
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False
        self.walker_hazard = False
        self.stop_sign_close = False

        # Compute if there will be a lane change close
        near_lane_change = self.is_near_lane_change(ego_speed, route_points)

        # Compute the number of future frames to consider for collision detection
        num_future_frames = int(
            self.config.bicycle_frame_rate *
            (self.config.forecast_length_lane_change if near_lane_change else self.config.default_forecast_length))

        # Get future bounding boxes of pedestrians
        if not plant:
            nearby_pedestrians, nearby_pedestrian_ids = self.forecast_walkers(actor_list, ego_vehicle_location,
                                                                              num_future_frames)

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_bounding_boxes = self.forecast_ego_agent(ego_vehicle_transform, ego_speed, num_future_frames,
                                                     initial_target_speed, route_points)

        # Predict bounding boxes of other actors (vehicles, bicycles, etc.)
        predicted_bounding_boxes = self.predict_other_actors_bounding_boxes(plant, vehicle_list, ego_vehicle_location,
                                                                            num_future_frames, near_lane_change)

        # Compute the leading and trailing vehicle IDs
        leading_vehicle_ids = self._waypoint_planner.compute_leading_vehicles(vehicle_list, self._vehicle.id)
        trailing_vehicle_ids = self._waypoint_planner.compute_trailing_vehicles(vehicle_list, self._vehicle.id)

        # Compute the target speed with respect to the leading vehicle
        target_speed_leading, speed_reduced_by_obj = self.compute_target_speed_wrt_leading_vehicle(
            initial_target_speed, predicted_bounding_boxes, near_lane_change, ego_vehicle_location,
            trailing_vehicle_ids, leading_vehicle_ids, speed_reduced_by_obj, plant)

        # Compute the target speeds with respect to all actors (vehicles, bicycles, pedestrians)
        target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, speed_reduced_by_obj = \
            self.compute_target_speeds_wrt_all_actors(initial_target_speed, ego_bounding_boxes,
            predicted_bounding_boxes, near_lane_change, leading_vehicle_ids, trailing_vehicle_ids, speed_reduced_by_obj,
            nearby_pedestrians, nearby_pedestrian_ids)

        # Compute the target speed with respect to the red light
        target_speed_red_light = self.ego_agent_affected_by_red_light(ego_vehicle_location, ego_speed, 
                                        distance_to_next_traffic_light, next_traffic_light, route_points, 
                                        initial_target_speed)

        # Update the object causing the most speed reduction
        if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_red_light:
            speed_reduced_by_obj = [
                target_speed_red_light, None if next_traffic_light is None else next_traffic_light.type_id,
                None if next_traffic_light is None else next_traffic_light.id, distance_to_next_traffic_light
            ]

        # Compute the target speed with respect to the stop sign
        target_speed_stop_sign = self.ego_agent_affected_by_stop_sign(ego_vehicle_location, ego_speed, next_stop_sign,
                                                                      initial_target_speed, actor_list)
        # Update the object causing the most speed reduction
        if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_stop_sign:
            speed_reduced_by_obj = [
                target_speed_stop_sign, None if next_stop_sign is None else next_stop_sign.type_id,
                None if next_stop_sign is None else next_stop_sign.id, distance_to_next_stop_sign
            ]

        # Compute the minimum target speed considering all factors
        target_speed = min(target_speed_leading, target_speed_bicycle, target_speed_vehicle, target_speed_pedestrian,
                           target_speed_red_light, target_speed_stop_sign)

        # Set the hazard flags based on the target speed and its cause
        if target_speed == target_speed_pedestrian and target_speed_pedestrian != initial_target_speed:
            self.walker_hazard = True
            self.walker_close = True
        elif target_speed == target_speed_red_light and target_speed_red_light != initial_target_speed:
            self.traffic_light_hazard = True
        elif target_speed == target_speed_stop_sign and target_speed_stop_sign != initial_target_speed:
            self.stop_sign_hazard = True
            self.stop_sign_close = True

        # Determine if the ego vehicle needs to brake based on the target speed
        brake = target_speed == 0
        return brake, target_speed, speed_reduced_by_obj

    def forecast_ego_agent(self, current_ego_transform, current_ego_speed, num_future_frames, initial_target_speed,
                           route_points):
        """
        Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to
        check subsequently whether the ego vehicle would collide.

        Args:
            current_ego_transform (carla.Transform): The current transform of the ego vehicle.
            current_ego_speed (float): The current speed of the ego vehicle in m/s.
            num_future_frames (int): The number of future frames to forecast.
            initial_target_speed (float): The initial target speed for the ego vehicle.
            route_points (numpy.ndarray): An array of waypoints representing the planned route.

        Returns:
            list: A list of bounding boxes representing the future states of the ego vehicle.
        """
        self._turn_controller.save_state()
        self._waypoint_planner.save()

        # Initialize the initial state without braking
        location = np.array(
            [current_ego_transform.location.x, current_ego_transform.location.y, current_ego_transform.location.z])
        heading_angle = np.array([np.deg2rad(current_ego_transform.rotation.yaw)])
        # speed = np.array([current_ego_speed])
        speed = current_ego_speed

        # Calculate the throttle command based on the target speed and current speed
        throttle = self._longitudinal_controller.get_throttle_extrapolation(initial_target_speed, current_ego_speed)
        steering = self._turn_controller.step(route_points, speed, location, heading_angle.item())
        action = np.array([steering, throttle, 0.0]).flatten()

        future_bounding_boxes = []
        # Iterate over the future frames and forecast the ego agent's state
        for _ in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            location, heading_angle, speed = self.ego_model.forecast_ego_vehicle(location, heading_angle, speed, action)

            # Update the route and extrapolate steering and throttle commands
            extrapolated_route, _, _, _, _, _, _, _ = self._waypoint_planner.run_step(location)
            steering = self._turn_controller.step(extrapolated_route, speed, location, heading_angle.item())
            throttle = self._longitudinal_controller.get_throttle_extrapolation(initial_target_speed, speed)
            action = np.array([steering, throttle, 0.0]).flatten()

            heading_angle_degrees = np.rad2deg(heading_angle).item()

            # Decrease the ego vehicles bounding box if it is slow and resolve permanent bounding box
            # intersectinos at collisions.
            # In case of driving increase them for safety.
            extent = self._vehicle.bounding_box.extent
            # Otherwise we would increase the extent of the bounding box of the vehicle
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
            extent.x *= self.config.slow_speed_extent_factor_ego if current_ego_speed < \
                            self.config.extent_ego_bbs_speed_threshold else self.config.high_speed_extent_factor_ego_x
            extent.y *= self.config.slow_speed_extent_factor_ego if current_ego_speed < \
                            self.config.extent_ego_bbs_speed_threshold else self.config.high_speed_extent_factor_ego_y

            transform = carla.Transform(carla.Location(x=location[0].item(), y=location[1].item(),
                                                       z=location[2].item()))

            ego_bounding_box = carla.BoundingBox(transform.location, extent)
            ego_bounding_box.rotation = carla.Rotation(pitch=0, yaw=heading_angle_degrees, roll=0)

            future_bounding_boxes.append(ego_bounding_box)

        self._turn_controller.load_state()
        self._waypoint_planner.load()

        return future_bounding_boxes

    def forecast_walkers(self, actors, ego_vehicle_location, number_of_future_frames):
        """
        Forecast the future locations of pedestrians in the vicinity of the ego vehicle assuming they 
        keep their velocity and direction

        Args:
            actors (carla.ActorList): A list of actors in the simulation.
            ego_vehicle_location (carla.Location): The current location of the ego vehicle.
            number_of_future_frames (int): The number of future frames to forecast.

        Returns:
            tuple: A tuple containing two lists:
                - list: A list of lists, where each inner list contains the future bounding boxes for a pedestrian.
                - list: A list of IDs for the pedestrians whose locations were forecasted.
        """
        nearby_pedestrians_bbs, nearby_pedestrian_ids = [], []

        # Filter pedestrians within the detection radius
        pedestrians = [
            ped for ped in actors.filter("*walker*")
            if ped.get_location().distance(ego_vehicle_location) < self.config.detection_radius
        ]

        # If no pedestrians are found, return empty lists
        if not pedestrians:
            return nearby_pedestrians_bbs, nearby_pedestrian_ids

        # Extract pedestrian locations, speeds, and directions
        pedestrian_locations = np.array([[ped.get_location().x,
                                          ped.get_location().y,
                                          ped.get_location().z] for ped in pedestrians])
        pedestrian_speeds = np.array([ped.get_velocity().length() for ped in pedestrians])
        pedestrian_speeds = np.maximum(pedestrian_speeds, self.config.min_walker_speed)
        pedestrian_directions = np.array(
            [[ped.get_control().direction.x,
              ped.get_control().direction.y,
              ped.get_control().direction.z] for ped in pedestrians])

        # Calculate future pedestrian locations based on their current locations, speeds, and directions
        future_pedestrian_locations = pedestrian_locations[:, None, :] + np.arange(1, number_of_future_frames + 1)[
            None, :, None] * pedestrian_directions[:,
                                                   None, :] * pedestrian_speeds[:, None,
                                                                                None] / self.config.bicycle_frame_rate

        # Iterate over pedestrians and calculate their future bounding boxes
        for i, ped in enumerate(pedestrians):
            bb, transform = ped.bounding_box, ped.get_transform()
            rotation = carla.Rotation(pitch=bb.rotation.pitch + transform.rotation.pitch,
                                      yaw=bb.rotation.yaw + transform.rotation.yaw,
                                      roll=bb.rotation.roll + transform.rotation.roll)
            extent = bb.extent
            extent.x = max(self.config.pedestrian_minimum_extent, extent.x)  # Ensure a minimum width
            extent.y = max(self.config.pedestrian_minimum_extent, extent.y)  # Ensure a minimum length

            pedestrian_future_bboxes = []
            for j in range(number_of_future_frames):
                location = carla.Location(future_pedestrian_locations[i, j, 0], future_pedestrian_locations[i, j, 1],
                                          future_pedestrian_locations[i, j, 2])

                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation
                pedestrian_future_bboxes.append(bounding_box)

            nearby_pedestrian_ids.append(ped.id)
            nearby_pedestrians_bbs.append(pedestrian_future_bboxes)

        # Visualize the future bounding boxes of pedestrians (if enabled)
        if self.visualize == 1:
            for bbs in nearby_pedestrians_bbs:
                for bbox in bbs:
                    self._world.debug.draw_box(box=bbox,
                                               rotation=bbox.rotation,
                                               thickness=0.1,
                                               color=self.config.pedestrian_forecasted_bbs_color,
                                               life_time=self.config.draw_life_time)

        return nearby_pedestrians_bbs, nearby_pedestrian_ids

    def ego_agent_affected_by_red_light(self, ego_vehicle_location, ego_vehicle_speed, distance_to_traffic_light, 
                                        next_traffic_light, route_points, target_speed):
        """
        Handles the behavior of the ego vehicle when approaching a traffic light.

        Args:
            ego_vehicle_location (carla.Location): The ego vehicle location.
            ego_vehicle_speed (float): The current speed of the ego vehicle in m/s.
            distance_to_traffic_light (float): The distance from the ego vehicle to the next traffic light.
            next_traffic_light (carla.TrafficLight or None): The next traffic light in the route.
            route_points (numpy.ndarray): An array of (x, y, z) coordinates representing the route.
            target_speed (float): The target speed for the ego vehicle.

        Returns:
            float: The adjusted target speed for the ego vehicle.
        """

        self.close_traffic_lights.clear()

        for light, center, waypoints in self.list_traffic_lights:

            center_loc = carla.Location(center)
            if center_loc.distance(ego_vehicle_location) > self.config.light_radius:
                continue

            for wp in waypoints:
                # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
                length_bounding_box = carla.Vector3D((wp.lane_width / 2.0) * 0.9, light.trigger_volume.extent.y,
                                                                                         light.trigger_volume.extent.z)
                length_bounding_box = carla.Vector3D(1.5, 1.5, 0.5)

                bounding_box = carla.BoundingBox(wp.transform.location, length_bounding_box)

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch=gloabl_rot.pitch,
                                                       yaw=gloabl_rot.yaw,
                                                       roll=gloabl_rot.roll)

                affects_ego = next_traffic_light is not None and light.id==next_traffic_light.id

                self.close_traffic_lights.append([bounding_box, light.state, light.id, affects_ego])

                if self.visualize == 1:
                    if light.state == carla.libcarla.TrafficLightState.Red:
                        color = carla.Color(255, 0, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Yellow:
                        color = carla.Color(255, 255, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Green:
                        color = carla.Color(0, 255, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Off:
                        color = carla.Color(0, 0, 0, 255)
                    else:  # unknown
                        color = carla.Color(0, 0, 255, 255)

                    self._world.debug.draw_box(box=bounding_box,
                                                    rotation=bounding_box.rotation,
                                                    thickness=0.1,
                                                    color=color,
                                                    life_time=0.051)

                    self._world.debug.draw_point(wp.transform.location + carla.Location(z=light.trigger_volume.location.z),
                                                                             size=0.1,
                                                                             color=color,
                                                                             life_time=(1.0 / self.config.carla_fps)+1e-6)

        if next_traffic_light is None or next_traffic_light.state == carla.TrafficLightState.Green:
            # No traffic light or green light, continue with the current target speed
            return target_speed

        # Compute the target speed using the IDM
        target_speed = self._compute_target_speed_idm(
            desired_speed=target_speed,
            leading_actor_length=0.,
            ego_speed=ego_vehicle_speed,
            leading_actor_speed=0.,
            distance_to_leading_actor=distance_to_traffic_light,
            s0=self.config.idm_red_light_minimum_distance,
            T=self.config.idm_red_light_desired_time_headway
        )

        return target_speed

    def ego_agent_affected_by_stop_sign(self, ego_vehicle_location, ego_vehicle_speed, next_stop_sign, target_speed, 
                                        actor_list):
        """
        Handles the behavior of the ego vehicle when approaching a stop sign.

        Args:
            ego_vehicle_location (carla.Location): The location of the ego vehicle.
            ego_vehicle_speed (float): The current speed of the ego vehicle in m/s.
            next_stop_sign (carla.TrafficSign or None): The next stop sign in the route.
            target_speed (float): The target speed for the ego vehicle.
            actor_list (list): A list of all actors (vehicles, pedestrians, etc.) in the simulation.

        Returns:
            float: The adjusted target speed for the ego vehicle.
        """
        self.close_stop_signs.clear()
        stop_signs = self.get_nearby_object(ego_vehicle_location, actor_list.filter('*traffic.stop*'), self.config.light_radius)
        
        for stop_sign in stop_signs:
            center_bb_stop_sign = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
            wp = self.world_map.get_waypoint(center_bb_stop_sign)
            stop_sign_extent = carla.Vector3D(1.5, 1.5, 0.5)
            bounding_box_stop_sign = carla.BoundingBox(center_bb_stop_sign, stop_sign_extent)
            rotation_stop_sign = stop_sign.get_transform().rotation
            bounding_box_stop_sign.rotation = carla.Rotation(pitch=rotation_stop_sign.pitch,
                                                             yaw=rotation_stop_sign.yaw,
                                                             roll=rotation_stop_sign.roll)

            affects_ego = (next_stop_sign is not None and next_stop_sign.id==stop_sign.id and not self.cleared_stop_sign)
            self.close_stop_signs.append([bounding_box_stop_sign, stop_sign.id, affects_ego])

            if self.visualize:
                color = carla.Color(0, 1, 0) if affects_ego else carla.Color(1, 0, 0)
                self._world.debug.draw_box(box=bounding_box_stop_sign,
                                                                     rotation=bounding_box_stop_sign.rotation,
                                                                     thickness=0.1,
                                                                     color=color,
                                                                     life_time=(1.0 / self.config.carla_fps)+1e-6)

        if next_stop_sign is None:
            # No stop sign, continue with the current target speed
            return target_speed

        # Calculate the accurate distance to the stop sign
        distance_to_stop_sign = next_stop_sign.get_transform().transform(next_stop_sign.trigger_volume.location) \
            .distance(ego_vehicle_location)

        # Reset the stop sign flag if we are farther than 10m away
        if distance_to_stop_sign > self.config.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
        else:
            # Set the stop sign flag if we are closer than 3m and speed is low enough
            if ego_vehicle_speed < 0.1 and distance_to_stop_sign < self.config.clearing_distance_to_stop_sign:
                self.cleared_stop_sign = True

        # Set the distance to stop sign as infinity if the stop sign has been cleared
        distance_to_stop_sign = np.inf if self.cleared_stop_sign else distance_to_stop_sign

        # Compute the target speed using the IDM
        target_speed = self._compute_target_speed_idm(
            desired_speed=target_speed,
            leading_actor_length=0,  #self._vehicle.bounding_box.extent.x,
            ego_speed=ego_vehicle_speed,
            leading_actor_speed=0.,
            distance_to_leading_actor=distance_to_stop_sign,
            s0=self.config.idm_stop_sign_minimum_distance,
            T=self.config.idm_stop_sign_desired_time_headway)

        # Return whether the ego vehicle is affected by the stop sign and the adjusted target speed
        return target_speed

    def _dot_product(self, vector1, vector2):
        """
        Calculate the dot product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            float: The dot product of the two vectors.
        """
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

    def cross_product(self, vector1, vector2):
        """
        Calculate the cross product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            carla.Vector3D: The cross product of the two vectors.
        """
        x = vector1.y * vector2.z - vector1.z * vector2.y
        y = vector1.z * vector2.x - vector1.x * vector2.z
        z = vector1.x * vector2.y - vector1.y * vector2.x

        return carla.Vector3D(x=x, y=y, z=z)

    def get_separating_plane(self, relative_position, plane_normal, obb1, obb2):
        """
        Check if there is a separating plane between two oriented bounding boxes (OBBs).

        Args:
            relative_position (carla.Vector3D): The relative position between the two OBBs.
            plane_normal (carla.Vector3D): The normal vector of the plane.
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if there is a separating plane, False otherwise.
        """
        # Calculate the projection of the relative position onto the plane normal
        projection_distance = abs(self._dot_product(relative_position, plane_normal))

        # Calculate the sum of the projections of the OBB extents onto the plane normal
        obb1_projection = (abs(self._dot_product(obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal)) +
                           abs(self._dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal)) +
                           abs(self._dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal)))

        obb2_projection = (abs(self._dot_product(obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal)) +
                           abs(self._dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal)) +
                           abs(self._dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal)))

        # Check if the projection distance is greater than the sum of the OBB projections
        return projection_distance > obb1_projection + obb2_projection

    def check_obb_intersection(self, obb1, obb2):
        """
        Check if two 3D oriented bounding boxes (OBBs) intersect.

        Args:
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if the two OBBs intersect, False otherwise.
        """
        relative_position = obb2.location - obb1.location

        # Check for separating planes along the axes of both OBBs
        if (self.get_separating_plane(relative_position, obb1.rotation.get_forward_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb1.rotation.get_right_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb1.rotation.get_up_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb2.rotation.get_forward_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb2.rotation.get_right_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb2.rotation.get_up_vector(), obb1, obb2)):

            return False

        # Check for separating planes along the cross products of the axes of both OBBs
        if (self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1, obb2)):

            return False

        # If no separating plane is found, the OBBs intersect
        return True

    def _get_angle_to(self, current_position, current_heading, target_position):
        """
        Calculate the angle (in degrees) from the current position and heading to a target position.

        Args:
            current_position (list): A list of (x, y) coordinates representing the current position.
            current_heading (float): The current heading angle in radians.
            target_position (tuple or list): A tuple or list of (x, y) coordinates representing the target position.

        Returns:
            float: The angle (in degrees) from the current position and heading to the target position.
        """
        cos_heading = math.cos(current_heading)
        sin_heading = math.sin(current_heading)

        # Calculate the vector from the current position to the target position
        position_delta = target_position - current_position

        # Calculate the dot product of the position delta vector and the current heading vector
        aim_x = cos_heading * position_delta[0] + sin_heading * position_delta[1]
        aim_y = -sin_heading * position_delta[0] + cos_heading * position_delta[1]

        # Calculate the angle (in radians) from the current heading to the target position
        angle_radians = -math.atan2(-aim_y, aim_x)

        # Convert the angle from radians to degrees
        angle_degrees = np.float_(math.degrees(angle_radians))

        return angle_degrees

    def get_nearby_object(self, ego_vehicle_position, all_actors, search_radius):
        """
        Find actors, who's trigger boxes are within a specified radius around the ego vehicle.

        Args:
            ego_vehicle_position (carla.Location): The position of the ego vehicle.
            all_actors (list): A list of all actors.
            search_radius (float): The radius (in meters) around the ego vehicle to search for nearby actors.

        Returns:
            list: A list of actors within the specified search radius.
        """
        nearby_objects = []
        for actor in all_actors:
            try:
                trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            except:
                print("Warning! Error caught in get_nearby_objects. (probably AttributeError: actor.trigger_volume)")
                print("Skipping this object.")
                continue

            # Convert the vector to a carla.Location for distance calculation
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x,
                                                    y=trigger_box_global_pos.y,
                                                    z=trigger_box_global_pos.z)

            # Check if the actor's trigger volume is within the search radius
            if trigger_box_global_pos.distance(ego_vehicle_position) < search_radius:
                nearby_objects.append(actor)

        return nearby_objects
