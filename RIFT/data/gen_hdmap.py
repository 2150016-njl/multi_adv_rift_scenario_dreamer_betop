#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : gen_hdmap.py
@Date    : 2024/9/11
"""
import json
import carla
import numpy as np
import argparse

from tqdm import tqdm
from shapely import LineString, Point
from collections import defaultdict
from shapely.geometry import Polygon, box

LANE_CONNECT_DIS = {
    'Town01': 2.0,
    'Town02': 2.0,
    'Town03': 2.0,
    'Town04': 2.0,
    'Town05': 2.0,
    'Town06': 2.0,
    'Town07': 2.0,
    'Town10HD': 2.0,
    'Town11': 2.0,
    'Town12': 2.0,
    'Town13': 4.0,
    'Town15': 3.0
}

LANE_MIN_DIS = {
    'Town01': 100.0,
    'Town02': 100.0,
    'Town03': 100.0,
    'Town04': 100.0,
    'Town05': 100.0,
    'Town06': 100.0,
    'Town07': 100.0,
    'Town10HD': 100.0,
    'Town11': 100.0,
    'Town12': 100.0,  # have highway extry and exit point problem
    'Town13': 100.0,  # have highway extry and exit point problem
    'Town15': 30.0  # have highway extry and exit point problem
}


def check_waypoints_status(waypoints_list):
    first_wp = waypoints_list[0]
    init_status = first_wp.is_junction
    current_status = first_wp.is_junction
    change_status_time = 0
    for wp in waypoints_list[1:]:
        if wp.is_junction != current_status:
            current_status = wp.is_junction
            change_status_time += 1
        pass
    if change_status_time == 0:
        return 'Junction' if init_status else 'Normal'
    elif change_status_time == 1:
        return 'EnterNormal' if init_status else 'EnterJunction'
    elif change_status_time == 2:
        return 'PassNormal' if init_status else 'PassJunction'
    else:
        return 'StartJunctionMultiChange' if init_status else 'StartNormalMultiChange'


class TriggerVolumeGettor(object):

    @staticmethod
    def get_global_bbx(actor, bbx):
        if actor.is_alive:
            bbx.location = actor.get_transform().transform(bbx.location)
            bbx.rotation = actor.get_transform().rotation
            bbx.rotation.pitch = bbx.rotation.pitch
            bbx.rotation.yaw = -bbx.rotation.yaw  # inverse the yaw angle

            return bbx
        return None

    @staticmethod
    def get_corners_from_actor_list(actor_list):
        for actor_transform, bb_loc, bb_ext in actor_list:
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners = [[corner.x, -corner.y, corner.z] for corner in corners]  # from left-hand system to the right-hand system
        return corners

    @staticmethod
    def insert_point_into_dict(lane_marking_dict, corners, road_id, parent_actor_location, Volume_Type=None):
        if road_id not in lane_marking_dict.keys():
            print("Cannot find road:", road_id)
            raise
        if Volume_Type is None:
            print("Missing 'Volume Type' ")
            raise

        polygon = Polygon(corners)  # using the Polygon instead of points

        if 'Trigger_Volumes' not in lane_marking_dict[road_id]:
            lane_marking_dict[road_id]['Trigger_Volumes'] = [{'Polygon': polygon, 'Type': Volume_Type, 'ParentActor_Location': parent_actor_location[:]}]
        else:
            lane_marking_dict[road_id]['Trigger_Volumes'].append({'Polygon': polygon, 'Type': Volume_Type, 'ParentActor_Location': parent_actor_location[:]})

    @staticmethod
    def get_stop_sign_trigger_volume(all_stop_sign_actors, lane_marking_dict, carla_map):
        for actor in all_stop_sign_actors:
            bb_loc = carla.Location(actor.trigger_volume.location)
            bb_ext = carla.Vector3D(actor.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            base_transform = actor.get_transform()
            stop_info_list = [(carla.Transform(base_transform.location, base_transform.rotation), bb_loc, bb_ext)]
            corners = TriggerVolumeGettor.get_corners_from_actor_list(stop_info_list)

            trigger_volume_wp = carla_map.get_waypoint(base_transform.transform(bb_loc))
            actor_loc = actor.get_location()
            actor_loc_points = [actor_loc.x, -actor_loc.y, actor_loc.z]  # from left-hand system to the right-hand system
            TriggerVolumeGettor.insert_point_into_dict(lane_marking_dict, corners, trigger_volume_wp.road_id, actor_loc_points, Volume_Type='StopSign')

        pass

    @staticmethod
    def get_traffic_light_trigger_volume(all_trafficlight_actors, lane_marking_dict, carla_map):
        for actor in all_trafficlight_actors:
            base_transform = actor.get_transform()
            tv_loc = actor.trigger_volume.location
            tv_ext = actor.trigger_volume.extent
            x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)
            area = []
            for x in x_values:
                point_location = base_transform.transform(tv_loc + carla.Location(x=x))
                area.append(point_location)
            ini_wps = []
            for pt in area:
                wpx = carla_map.get_waypoint(pt)
                # As x_values are arranged in order, only the last one has to be checked
                if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                    ini_wps.append(wpx)

            close2junction_points = []
            littlefar2junction_points = []
            for wpx in ini_wps:
                while not wpx.is_intersection:
                    next_wp = wpx.next(0.5)
                    if not next_wp:
                        break
                    next_wp = next_wp[0]
                    if next_wp and not next_wp.is_intersection:
                        wpx = next_wp
                    else:
                        break
                vec_forward = wpx.transform.get_forward_vector()
                vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)  # 2D

                loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
                loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
                close2junction_points.append([loc_left.x, loc_left.y, loc_left.z])
                close2junction_points.append([loc_right.x, loc_right.y, loc_right.z])

                try:
                    loc_far_left = wpx.previous(0.5)[0].transform.location - 0.4 * wpx.lane_width * vec_right
                    loc_far_right = wpx.previous(0.5)[0].transform.location + 0.4 * wpx.lane_width * vec_right
                except Exception:
                    continue

                littlefar2junction_points.append([loc_far_left.x, loc_far_left.y, loc_far_left.z])
                littlefar2junction_points.append([loc_far_right.x, loc_far_right.y, loc_far_right.z])

            traffic_light_points = close2junction_points + littlefar2junction_points[::-1]
            traffic_light_points = [[point[0], -point[1], point[2]] for point in traffic_light_points]  # from left-hand system to the right-hand system

            trigger_volume_wp = carla_map.get_waypoint(base_transform.transform(tv_loc))
            actor_loc = actor.get_location()
            actor_loc_points = [actor_loc.x, -actor_loc.y, actor_loc.z]  # from left-hand system to the right-hand system
            TriggerVolumeGettor.insert_point_into_dict(lane_marking_dict, traffic_light_points, trigger_volume_wp.road_id, actor_loc_points, Volume_Type='TrafficLight')
        pass

    pass


t = 0


class LankMarkingGettor(object):
    '''
        structure of lane_marking_dict:
        {
            road_id_0: {
                lane_id_0: 
                {'Center': ['Points': [((location.x,y,z) array, (rotation.roll, pitch, yaw)), is_junction], 'Type': 'lane_marking_type', 'Color':'color', 'Topology':[neighbor array]],
                 'Left':   ['Points': [((location.x,y,z) array, (rotation.roll, pitch, yaw)), is_junction], 'Type': 'lane_marking_type', 'Color':'color', 'Topology':[neighbor array]],
                 'Right':  ['Points': [((location.x,y,z) array, (rotation.roll, pitch, yaw)), is_junction], 'Type': 'lane_marking_type', 'Color':'color', 'Topology':[neighbor array]]},
                ... ...
                'Trigger_Volumes': [{'Points': [(location.x,y,z) array], 'Type': 'trigger volume type', 'ParentActor_Location': (location.x,y,z)}]
            }
            ... ...
        }
        "location array" is an array formed as (location_x, location_y, location_z) ...
        'lane_marking_type' is string of landmarking type, can be 'Broken', 'Solid', 'SolidSolid', 'Other', 'NONE', etc.
        'color' is string of landmarking color, can be 'Blue', 'White', 'Yellow',  etc.
         neighbor array contains the ('road_id', 'lane_id') of the current landmarking adjacent to, it is directional.
         and if current 'Type' == 'Center', there will exist a 'TopologyType' key which record the current lane's topology status.
         if there exist a trigger volume in current road, key 'Trigger_Volumes' will be added into dict
         where 'Points' refer to the vertexs location array, 'Type' can be 'StopSign' or 'TrafficLight'
         'ParentActor_Location' is the location of parent actor relevant to this trigger volume.
    '''

    @staticmethod
    def get_lane_markings(carla_map, lane_marking_dict={}, pixels_per_meter=2, precision=0.5):
        # topology contains a list of tuples, tuple contain pairs of waypoints located either at the point a road begins or ends (driving road)
        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.road_id)

        all_x = []
        all_y = []

        for waypoint in tqdm(topology, desc="Processing waypoints", unit="waypoint"):
            waypoints = [waypoint]
            # Generate waypoints of a road id. Stop when road id differs
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            # print("current road id: ", waypoint.road_id)
            # print("lane id:", waypoint.lane_id)
            LankMarkingGettor.get_lane_markings_two_side(waypoints, lane_marking_dict, precision)

            # Collect x and y coordinates from waypoints
            x_coords = [wp.transform.location.x for wp in waypoints]
            y_coords = [wp.transform.location.y for wp in waypoints]
            all_x.extend(x_coords)
            all_y.extend(y_coords)

        # post-processing
        LankMarkingGettor.postprocess_lane_marking(lane_marking_dict)

        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)

        # Return the bounding box polygon of the world
        return box(min_x, min_y, max_x, max_y)

    @staticmethod
    def get_lane_markings_two_side(waypoints, lane_marking_dict, precision):
        left_lane_marking_list = []
        right_lane_marking_list = []
        center_lane_list = []
        center_lane_wps = []

        left_previous_lane_marking_type = 1
        left_previous_lane_marking_color = 1
        right_previous_lane_marking_type = 1
        right_previous_lane_marking_color = 1

        center_previous_lane_id = waypoints[0].lane_id
        lane_type = str(waypoints[0].lane_type)  # get initial lane type

        for waypoint in waypoints:
            flag = False
            if waypoint.lane_id != center_previous_lane_id:
                # Handle lane ID change
                flag = True
                # Process accumulated center lane markings
                if len(center_lane_list) > 1:
                    LankMarkingGettor.process_center_lane_marking(
                        waypoint, center_previous_lane_id, lane_type, center_lane_list, center_lane_wps, lane_marking_dict, precision
                    )
                center_lane_list = []
                center_lane_wps = []

            left_lane_marking = waypoint.left_lane_marking
            right_lane_marking = waypoint.right_lane_marking

            # Check for changes in left or right lane markings
            left_change = left_lane_marking.type != left_previous_lane_marking_type or \
                left_lane_marking.color != left_previous_lane_marking_color or flag
            right_change = right_lane_marking.type != right_previous_lane_marking_type or \
                right_lane_marking.color != right_previous_lane_marking_color or flag

            if left_change:
                if len(left_lane_marking_list) > 1:
                    # Process accumulated left lane markings
                    LankMarkingGettor.process_side_lane_marking(
                        waypoint, center_previous_lane_id, lane_type, left_lane_marking_list,
                        left_previous_lane_marking_type, left_previous_lane_marking_color,
                        lane_marking_dict, precision, side='Left'
                    )
                    left_lane_marking_list = []

            if right_change:
                if len(right_lane_marking_list) > 1:
                    # Process accumulated right lane markings
                    LankMarkingGettor.process_side_lane_marking(
                        waypoint, center_previous_lane_id, lane_type, right_lane_marking_list,
                        right_previous_lane_marking_type, right_previous_lane_marking_color,
                        lane_marking_dict, precision, side='Right'
                    )
                    right_lane_marking_list = []

            # Append current points to the lists
            center_lane_list.append(
                (
                    *LankMarkingGettor.get_lateral_shifted_transform(waypoint.transform, 0),
                    waypoint.is_junction
                )
            )
            center_lane_wps.append(waypoint)

            left_lane_marking_list.append(
                (
                    *LankMarkingGettor.get_lateral_shifted_transform(waypoint.transform, -0.5 * waypoint.lane_width),
                    waypoint.is_junction
                )
            )

            right_lane_marking_list.append(
                (
                    *LankMarkingGettor.get_lateral_shifted_transform(waypoint.transform, 0.5 * waypoint.lane_width),
                    waypoint.is_junction
                )
            )

            # Update previous lane marking types and colors
            left_previous_lane_marking_type = left_lane_marking.type
            left_previous_lane_marking_color = left_lane_marking.color
            right_previous_lane_marking_type = right_lane_marking.type
            right_previous_lane_marking_color = right_lane_marking.color
            center_previous_lane_id = waypoint.lane_id  # update lane id
            lane_type = str(waypoint.lane_type)  # update lane type

        # Process any remaining lane markings after the loop
        if len(center_lane_list) > 1:
            LankMarkingGettor.process_center_lane_marking(
                waypoint, center_previous_lane_id, lane_type, center_lane_list, center_lane_wps, lane_marking_dict, precision
            )

        if len(left_lane_marking_list) > 1:
            LankMarkingGettor.process_side_lane_marking(
                waypoint, center_previous_lane_id, lane_type, left_lane_marking_list,
                left_previous_lane_marking_type, left_previous_lane_marking_color,
                lane_marking_dict, precision, side='Left'
            )

        if len(right_lane_marking_list) > 1:
            LankMarkingGettor.process_side_lane_marking(
                waypoint, center_previous_lane_id, lane_type, right_lane_marking_list,
                right_previous_lane_marking_type, right_previous_lane_marking_color,
                lane_marking_dict, precision, side='Right'
            )

    @staticmethod
    def postprocess_lane_marking(lane_marking_dict):
        for road_id, road_data in lane_marking_dict.items():
            for lane_id, lane_data in road_data.items():
                lane_mark_dict = lane_data.get('LaneMark', {})
                for key, lane_mark_list in lane_mark_dict.items():
                    lane_mark_dict[key] = LankMarkingGettor.remove_duplicate_dicts(key, lane_mark_list, road_id, lane_id)
                    
    @staticmethod
    def remove_duplicate_dicts(key, dict_list, road_id, lane_id):
        seen = set()
        unique_dicts = []

        # remove the exact same dict
        for d in dict_list:
            dict_str = json.dumps(d, sort_keys=True)
            if dict_str not in seen:
                seen.add(dict_str)
                unique_dicts.append(d)

        # reorder the dict list and remove the duplicate Points
        line_string_dict = {}
        for i, d in enumerate(unique_dicts):
            raw_points = d['Points']
            if len(raw_points) < 2:
                print(f"Invalid {key} lane marking dict with less than 2 points")
                continue
            points = [tuple(p[0][:2]) for p in raw_points]
            line_string_dict[i] = LineString(points)

        # create a graph to find the longest path
        adj = defaultdict(list)
        start_points = {}
        end_points = {}
        for i, line in line_string_dict.items():
            start_points[i] = line.coords[0]
            end_points[i] = line.coords[-1]

        for i in line_string_dict:
            for j in line_string_dict:
                if i < j: 
                    i2j_len = LineString([end_points[i], start_points[j]]).length
                    j2i_len = LineString([end_points[j], start_points[i]]).length
                    if i2j_len < LANE_CONNECT_DIS[args.carla_town] and j2i_len > LANE_CONNECT_DIS[args.carla_town]:
                        adj[i].append(j)  # direction i -> j
                    elif j2i_len < LANE_CONNECT_DIS[args.carla_town] and i2j_len > LANE_CONNECT_DIS[args.carla_town]:
                        adj[j].append(i)  # direction j -> i
                    elif i2j_len < LANE_CONNECT_DIS[args.carla_town] and j2i_len < LANE_CONNECT_DIS[args.carla_town]:
                        if i2j_len < j2i_len:
                            adj[i].append(j)  # direction i -> j
                        else:
                            adj[j].append(i)  # direction j -> i

        # dfs to find the longest path
        memo = {}
        def dfs(node):
            if node in memo:
                return memo[node]
            max_len = line_string_dict[node].length
            best_path = [node]
            for neighbor in adj[node]:
                length, path = dfs(neighbor)
                if line_string_dict[node].length + length > max_len:
                    max_len = line_string_dict[node].length + length
                    best_path = [node] + path
            memo[node] = (max_len, best_path)
            return (max_len, best_path)

        # loop through all nodes to find the longest path
        max_length, best_path = 0, []
        for node in line_string_dict:
            length, path = dfs(node)
            if length > max_length:
                max_length, best_path = length, path


        longest_dicts = [unique_dicts[i] for i in best_path]

        longest_string_dict = [line_string_dict[i] for i in best_path]
        longest_string = LineString([point for line in longest_string_dict for point in line.coords])
        all_indices = set(range(len(unique_dicts)))
        remaining_indices = all_indices - set(best_path)
        possible_second_index = []
        for index in remaining_indices:
            remain_string = line_string_dict[index]
            min_dis = longest_string.distance(remain_string)
            if min_dis > LANE_MIN_DIS[args.carla_town]:
                possible_second_index.append(index)

        if possible_second_index:
            print(f'Road {road_id} Lane {lane_id} has {len(possible_second_index)} possible remain paths in {key}, choosing the second longest path')
            second_max_length, second_best_path = 0, []
            for node in possible_second_index:
                length, path = dfs(node)
                if length > second_max_length:
                    second_max_length, second_best_path = length, path

            second_longest_dict = [unique_dicts[i] for i in second_best_path]
            second_best_string_dict = [line_string_dict[i] for i in second_best_path]
            second_best_string = LineString([point for line in second_best_string_dict for point in line.coords])
            end2start = LineString([longest_string.coords[-1], second_best_string.coords[0]]).length
            start2end = LineString([longest_string.coords[0], second_best_string.coords[-1]]).length
            if end2start < start2end:
                result_dicts = longest_dicts + second_longest_dict
            else:
                result_dicts = second_longest_dict + longest_dicts
        else:
            result_dicts = longest_dicts
        
        return result_dicts


    @staticmethod
    def process_center_lane_marking(waypoint, lane_id, lane_type, center_lane_list, center_lane_wps, lane_marking_dict, precision):
        if waypoint.road_id not in lane_marking_dict:
            lane_marking_dict[waypoint.road_id] = {}
        if lane_id not in lane_marking_dict[waypoint.road_id]:
            status = check_waypoints_status(center_lane_wps)
            lane_marking_dict[waypoint.road_id][lane_id] = {
                'LaneType': lane_type,
                'LaneWidth': waypoint.lane_width,
                'LaneMark': {'Center': [], 'Left': [], 'Right': []}
            }
        else:
            status = check_waypoints_status(center_lane_wps)
        lane_marking_dict[waypoint.road_id][lane_id]['LaneMark']['Center'].append(
            {
                'Points': center_lane_list[:],
                'Type': 'Center',
                'Color': 'White',
                'Topology': LankMarkingGettor.get_connected_road_id(waypoint, precision)[:],
                'TopologyType': status,
                'Left': (
                    center_lane_wps[-1].get_left_lane().road_id if center_lane_wps[-1].get_left_lane() else None,
                    center_lane_wps[-1].get_left_lane().lane_id if center_lane_wps[-1].get_left_lane() else None
                ),
                'Right': (
                    center_lane_wps[-1].get_right_lane().road_id if center_lane_wps[-1].get_right_lane() else None,
                    center_lane_wps[-1].get_right_lane().lane_id if center_lane_wps[-1].get_right_lane() else None
                )
            }
        )

    @staticmethod
    def process_side_lane_marking(waypoint, lane_id, lane_type, lane_marking_list, previous_type, previous_color, lane_marking_dict, precision, side='Left'):
        connect_to = LankMarkingGettor.get_connected_road_id(waypoint, precision)
        candidate_dict = {
            'Points': lane_marking_list[:],
            'Type': str(previous_type),
            'Color': str(previous_color),
            'Topology': connect_to[:]
        }
        if waypoint.road_id not in lane_marking_dict:
            lane_marking_dict[waypoint.road_id] = {}
        if lane_id not in lane_marking_dict[waypoint.road_id]:
            lane_marking_dict[waypoint.road_id][lane_id] = {
                'LaneType': lane_type,
                'LaneWidth': waypoint.lane_width,
                'LaneMark': {'Center': [], 'Left': [], 'Right': []}
            }
        lane_marking_dict[waypoint.road_id][lane_id]['LaneMark'][side].append(candidate_dict)

    @staticmethod
    def get_connected_road_id(waypoint, precision=0.5):
        next_waypoint = waypoint.next(precision)
        if next_waypoint is None:
            return []
        else:
            return [(w.road_id, w.lane_id) for w in next_waypoint if w.lane_type == carla.LaneType.Driving]

    @staticmethod
    def insert_element_into_dict(id, element, lane_marking_dict):
        if id not in lane_marking_dict:
            lane_marking_dict[id] = []
            lane_marking_dict[id].append(element)
        else:
            lane_marking_dict[id].append(element)

    @staticmethod
    def get_lateral_shifted_transform(transform, shift):
        right_vector = transform.get_right_vector()
        x_offset = right_vector.x * shift
        y_offset = right_vector.y * shift
        z_offset = right_vector.z * shift
        x = transform.location.x + x_offset
        y = - (transform.location.y + y_offset)  # inverse the y-axis
        z = transform.location.z + z_offset
        roll = np.deg2rad(transform.rotation.roll)  # rad
        pitch = np.deg2rad(transform.rotation.pitch)  # rad
        yaw = -np.deg2rad(transform.rotation.yaw)  # rad
        return ((x, y, z), (roll, pitch, yaw))

    @staticmethod
    def parse_crosswalks(crosswalk_locations):
        crosswalks = []
        current_polygon = []
        first_point = None

        for location in crosswalk_locations:
            point = (location.x, -location.y)  # from left-hand system to the right-hand system
            if not current_polygon:
                # start new polygon
                first_point = point
                current_polygon.append(point)
            else:
                current_polygon.append(point)
                if point == first_point:
                    # end current polygon
                    crosswalks.append(current_polygon.copy())
                    current_polygon = []
                    first_point = None

        return crosswalks

    @staticmethod
    def get_crosswalk(carla_map, lane_marking_dict, bounds_polygon):
        crosswalk_locations = carla_map.get_crosswalks()
        parsed_crosswalks = LankMarkingGettor.parse_crosswalks(crosswalk_locations)

        crosswalk_list = []
        for polygon_points in parsed_crosswalks:
            polygon = Polygon(polygon_points)
            if polygon.intersects(bounds_polygon):
                # only keep the polygon in the bounds_polygon of the world
                centroid = polygon.centroid
                crosswalk_list.append({
                    'Polygon': polygon,
                    'Location': (centroid.x, centroid.y)
                })
            else:
                # Polygon is completely outside the bounds; skip it
                continue

        lane_marking_dict['Crosswalks'] = crosswalk_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='data/map_data')
    parser.add_argument('--carla_town', '-town', default='Town05')
    parser.add_argument('--waypoint_precision', '-precision', default=0.5, help='default waypoint precision between point in one polyline 0.5m')

    args = parser.parse_args()
    carla_town = args.carla_town
    precision = args.waypoint_precision

    client = carla.Client('localhost', 2000)
    client.set_timeout(600)
    world = client.load_world(carla_town)
    print("******** successfully load the town:", carla_town, " ********")
    carla_map = world.get_map()

    lane_marking_dict = {}
    bounds_polygon = LankMarkingGettor.get_lane_markings(world.get_map(), lane_marking_dict, precision=precision)
    print("****** get all lane markings ******")

    all_actors = world.get_actors()
    all_stop_sign_actors = []
    all_traffic_light_actors = []
    for actor in all_actors:
        if 'traffic.stop' in actor.type_id:
            all_stop_sign_actors.append(actor)
        if 'traffic_light' in actor.type_id:
            all_traffic_light_actors.append(actor)

    print("Getting all trigger volumes ...")
    TriggerVolumeGettor.get_stop_sign_trigger_volume(all_stop_sign_actors, lane_marking_dict, carla_map)
    TriggerVolumeGettor.get_traffic_light_trigger_volume(all_traffic_light_actors, lane_marking_dict, carla_map)
    print("******* Have get all trigger volumes ! *********")

    print("Getting all Crosswalk ...")
    LankMarkingGettor.get_crosswalk(world.get_map(), lane_marking_dict, bounds_polygon)
    print("******* Have get all Crosswalk ! *********")

    arr = np.array(list(lane_marking_dict.items()), dtype=object)
    np.savez_compressed(args.save_dir + "/" + args.carla_town + "_HD_map.npz", arr=arr)