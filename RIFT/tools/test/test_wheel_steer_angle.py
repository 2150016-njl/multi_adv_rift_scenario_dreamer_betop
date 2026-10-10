#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : test_wheel_steer_angle.py
@Date    : 2024/09/28
"""
import carla
import random
import time
import numpy as np


def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        bp1 = blueprint_library.find('vehicle.tesla.model3')
        bp2 = blueprint_library.find('vehicle.diamondback.century')
        if bp1.has_attribute('color'):
            color1 = random.choice(bp1.get_attribute('color').recommended_values)
            bp1.set_attribute('color', color1)
        if bp2.has_attribute('color'):
            color2 = random.choice(bp2.get_attribute('color').recommended_values)
            bp2.set_attribute('color', color2)
        transform1 = random.choice(world.get_map().get_spawn_points())
        transform2 = random.choice(world.get_map().get_spawn_points())
        bp1.set_attribute('role_name', 'hero')
        bp2.set_attribute('role_name', 'background')
        ego = world.spawn_actor(bp1, transform1)
        bv = world.spawn_actor(bp2, transform2)
        actor_list.append(ego)
        actor_list.append(bv)
        print('created ego %s' % ego.type_id)
        print('ego extent:', ego.bounding_box.extent)
        print('created bv %s' % bv.type_id)
        print('bv extent:', bv.bounding_box.extent)
        # Let's put the vehicle to drive around.
        ego.set_autopilot(True)
        bv.set_autopilot(True)

        spectator = world.get_spectator()
        

        for _ in range(25000):
            world.tick()
            transform = ego.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(x=-3, z=50), carla.Rotation(yaw=transform.rotation.yaw, pitch=-80.0)
            )) 
            # get the physics angle in degrees of the vehicle's wheel
            front_wheel_angle = ego.get_wheel_steer_angle(carla.VehicleWheelLocation.Front_Wheel)
            print("front wheel angle: %f" % front_wheel_angle)
            front_left_wheel_angle = ego.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
            print("front left wheel angle: %f" % front_left_wheel_angle)
            front_right_wheel_angle = ego.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
            print("front right wheel angle: %f" % front_right_wheel_angle)
            back_wheel_angle = ego.get_wheel_steer_angle(carla.VehicleWheelLocation.Back_Wheel)
            print("back wheel angle: %f" % back_wheel_angle)
            print("--------------------------------")

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

