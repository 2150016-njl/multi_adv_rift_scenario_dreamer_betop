#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : test_global_route_planner.py
@Date    : 2024/10/17
'''
import carla
import random
import time
import json
import numpy as np
from rift.scenario.tools.global_route_planner import GlobalRoutePlanner


def parse_route_info(route_infos, start_wp, end_wp, global_route_planner: GlobalRoutePlanner, type: str):
    route, route_ids, route_dis = global_route_planner.trace_route(start_wp.transform.location, end_wp.transform.location)
    route_waypoints = [route_element[0] for route_element in route]
    route_state = [[wp.transform.location.x, -wp.transform.location.y, -np.deg2rad(wp.transform.rotation.yaw)] for wp in route_waypoints]
    road_id_count = {}

    for wp in route_waypoints:
        if wp.road_id in road_id_count:
            road_id_count[wp.road_id] += 1
        else:
            road_id_count[wp.road_id] = 1

    # route_id_list = []
    # # print("road id count", road_id_count)

    # for road_id, count in road_id_count.items():
    #     if count >= 5:
    #         route_id_list.append(road_id)
    
    route_infos[type] = {
        'state': route_state,
        'road_ids': route_ids['road_ids'],
        'lane_ids': route_ids['lane_ids'],
    }
    print(f"{type} route ids", route_ids)
    return route_infos


def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        map = world.get_map()
        global_route_planner = GlobalRoutePlanner(map, 1.0)
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.find('vehicle.tesla.model3')
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        start_wp = map.get_waypoint(carla.Location(x=15.0, y=0.0, z=2.0), project_to_road=True)
        
        end_wp_1 = map.get_waypoint(carla.Location(x=-52, y=50.0, z=2.0), project_to_road=True)
        end_wp_2 = map.get_waypoint(carla.Location(x=-80, y=0, z=2.0), project_to_road=True)
        end_wp_3 = map.get_waypoint(carla.Location(x=-40, y=-50.0, z=2.0), project_to_road=True)
        
        route_infos = {'start_point_xy': [15.0, 0.0], 'start_point_road_id': start_wp.road_id, 'start_point_lane_id': start_wp.lane_id}
        route_infos = parse_route_info(route_infos, start_wp, end_wp_1, global_route_planner, type='left')

        route_infos = parse_route_info(route_infos, start_wp, end_wp_2, global_route_planner, type='straight')

        route_infos = parse_route_info(route_infos, start_wp, end_wp_3, global_route_planner, type='right')

        file_path = 'tools/test/test_global_route_planner/example_routes.json'

        with open(file_path, 'w') as json_file:
            json.dump(route_infos, json_file, indent=4)

        print("successfully save the example routes")

        for _ in range(250):
            world.tick()
            

    finally:
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

