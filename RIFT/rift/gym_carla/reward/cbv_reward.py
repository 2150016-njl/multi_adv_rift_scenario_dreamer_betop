#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : cbv_reward.py
@Date    : 2024/10/13
'''
import numpy as np
import carla
from rift.cbv.planning.route_planner.goal_planner import CBVGoalPlanner
from rift.cbv.planning.route_planner.route_planner import CBVRoutePlanner
from rift.gym_carla.reward.reward_model import DenseRewardModel, SparseRewardModel
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class CBVBaseReward():
    type = 'base'
    def __init__(self, CBV_route_planner, desired_speed, out_lane_thres):
        self.ego_vehicle = None
        self.CBV_route_planner = CBV_route_planner
        self.desired_speed = desired_speed
        self.out_lane_thres = out_lane_thres

    def set_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def get_reward(self, CBVs_collision):
        return {}


class CBVFullTrainReward():
    type = 'FullTrain'
    """
    Reward model for RL model full training.
    """
    def __init__(self, CBV_route_planner, desired_speed, out_lane_thres):
        self.ego_vehicle = None
        self.desired_speed = desired_speed
        self.out_lane_thres = out_lane_thres
        self.CBV_route_planner: CBVGoalPlanner = CBV_route_planner

    def set_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def get_reward(self, CBVs_collision):
        """
        Get the CBV step reward
        """
        CBVs_reward = {}
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).items():

            # CBV collision punish collide with another vehicle
            # r_collision = -1 if CBVs_collision[CBV_id] is not None else 0
            r_collision = -1 if CBVs_collision[CBV_id] is not None and CBVs_collision[CBV_id]['other_actor_id'] != self.ego_vehicle.id else 0

            # encourage CBV to get closer to the goal point
            delta_dis = self.get_CBV_ego_reward(CBV)  # [-1, 1]

            # terminal reward (reach the goal)
            if CBV_id in CarlaDataProvider.get_CBVs_reach_goal_by_ego(self.ego_vehicle.id).keys():
                terminal_reward = 1
            else:
                terminal_reward = 0

            CBVs_reward[CBV_id] = delta_dis + 15 * r_collision + 15 * terminal_reward
            
        return CBVs_reward
    

    def get_CBV_ego_reward(self, CBV):
        """
            distance ratio and delta distance calculation
        """
        # get the dis between the goal and the CBV
        CBV_loc = CarlaDataProvider.get_location(CBV)
        goal_waypoint = self.CBV_route_planner.get_goal_waypoint()
        dis = CBV_loc.distance(goal_waypoint.transform.location)

        # delta_dis > 0 means ego and CBV are getting closer, otherwise punish CBV drive away from ego
        delta_dis = np.clip(self.CBV_route_planner.get_previous_goal_CBV_dis(CBV.id) - dis, a_min=-1., a_max=1.)
        return delta_dis


    # def get_CBV_bv_reward(self, CBV, search_radius, CBV_nearby_agents, tou=1):
    #     min_dis = search_radius  # the searching radius of the nearby_agent
    #     if CBV and CBV_nearby_agents:
    #         for i, vehicle in enumerate(CBV_nearby_agents):
    #             if vehicle.attributes.get('role_name') == 'background' and i < 3:  # except the ego vehicle and calculate only the closest two vehicles
    #                 # the min distance between bounding boxes of two vehicles
    #                 min_dis = get_min_distance_across_bboxes(CBV, vehicle)
    #         min_dis_reward = min(min_dis, tou) - tou  # the controlled bv shouldn't be too close to the other bvs
    #     else:
    #         min_dis_reward = 0
    #     return min_dis, min_dis_reward


class CBVFineTuneReward():
    type = 'FineTune'
    """
    Reward model for pluto fine-tuning.
    """
    def __init__(self, CBV_route_planner, config):
        self.ego_vehicle = None
        self.config = config

        # set the route planner
        self.CBV_route_planner: CBVRoutePlanner = CBV_route_planner
        # get the reward model from the config file
        self.reward_model = REWARD_MODELS[self.config['reward_model']](self.CBV_route_planner)

    def set_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def get_reward(self, CBVs_collision):
        CBVs_reward = {}
        for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).items():

            # get the cooresponding reward for the CBV
            CBVs_reward[CBV_id] = self.reward_model.get_reward(CBV, CBVs_collision[CBV_id])

        return CBVs_reward


class CBVDenseReward:
    def __init__(self, route_planner):
        """
        Dense reward model considering driving naturalism.
        """
        self.route_planner: CBVRoutePlanner = route_planner
        self.map = CarlaDataProvider.get_map()
        self.reward_model = DenseRewardModel()

    def get_reward(self, CBV, collision):
        """
            Compute the Dense Reward.
        """        
        loc = CarlaDataProvider.get_location(CBV)
        delta_dis, delta_angle = self.get_delta_info(CBV)

        boundary_wp = self.map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
        offroad = 1 - int(bool(boundary_wp))  # 0 if not violate, 1 if violate
        collision = int(bool(collision))  # 0 if not collide, 1 if collide
        
        # processed data
        speed = CarlaDataProvider.get_velocity(CBV).length()
        acc = CarlaDataProvider.get_acceleration(CBV).length()
        angular_speed = 0.0 
        angular_acc = 0.0  # no API to get the angular acceleration, assumed to be 0

        total_reward = self.reward_model.get_reward(delta_dis, delta_angle, speed, acc, angular_speed, angular_acc, collision, offroad)
        
        return total_reward
    
    def get_delta_info(self, CBV):
        n_points = 120
        reference_lines = self.route_planner.get_reference_line(CBV)
        ref_pos = np.zeros((len(reference_lines), n_points, 2), dtype=np.float64)  # [R, 120, 2]
        ref_angle = np.zeros((len(reference_lines), n_points), dtype=np.float64)  # [R, 120,]
        ref_valid_mask = np.zeros((len(reference_lines), n_points), dtype=np.bool_)  # [R, 120,]
        
        CBV_trans = CarlaDataProvider.get_transform(CBV)
        CBV_angle = -np.deg2rad(CBV_trans.rotation.yaw)
        CBV_pos = np.array([CBV_trans.location.x, -CBV_trans.location.y])

        for i, line in enumerate(reference_lines):
            subsample = line[::4]
            if len(subsample) < 2:
                subsample = line[:2]
            subsample = subsample[:n_points + 1]
            n_valid = len(subsample)
            ref_pos[i, : n_valid - 1] = subsample[:-1, :2]
            ref_angle[i, : n_valid - 1] = subsample[:-1, 2]
            ref_valid_mask[i, : n_valid - 1] = True
        
        valid_points = ref_pos[ref_valid_mask]  # [N_valid, 2]
        valid_angles = ref_angle[ref_valid_mask]  # [N_valid,]

        if len(valid_points) == 0:
            # no valid points
            delta_dis = 0.0
            delta_angle = 0.0
        else:
            distances = np.linalg.norm(valid_points - CBV_pos, axis=1)  # [N_valid,]
            # find the closest index
            min_dist_idx = np.argmin(distances)
            closest_pos = valid_points[min_dist_idx]  # [2,]
            closest_angle = valid_angles[min_dist_idx]  # [1,]

            delta_angle = CBV_angle - closest_angle
            delta_angle = np.arctan2(np.sin(delta_angle), np.cos(delta_angle))

            rel_pos = CBV_pos - closest_pos
            tangent_dir = np.array([np.cos(closest_angle), np.sin(closest_angle)])
            delta_dis = -np.cross(rel_pos, tangent_dir)

        return abs(delta_dis), abs(delta_angle)
    

class CBVSparseReward:
    def __init__(self, route_planner):
        """
        Sparse infraction-based reward model.
        """
        self.route_planner: CBVRoutePlanner = route_planner
        self.map = CarlaDataProvider.get_map()
        self.reward_model = SparseRewardModel()
    
    def get_reward(self, CBV, collision):
        """
            Compute the Sparse Reward.
        """
        loc = CarlaDataProvider.get_location(CBV)
        boundary_wp = self.map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
        offroad = 1 - int(bool(boundary_wp))
        collision = int(bool(collision))

        total_reward = self.reward_model.get_reward(collision, offroad)

        return total_reward


class CBVNoneReward:
    def __init__(self, route_planner):
        """
        None reward model.
        """
        self.route_planner: CBVRoutePlanner = route_planner
        self.params = self._sample_params()
    
    def _sample_params(self):
        """
        Sample hyperparameters.
        """
        return None
    
    def get_reward(self, CBV, collision):
        return 0
    
    def get_params(self):
        """
        Return the current hyperparameters of the reward model.
        """
        return self.params


REWARD_MODELS = {
    'dense': CBVDenseReward,
    'sparse': CBVSparseReward,
    'none': CBVNoneReward
}