#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : test_HDmap
@Date    : 2024/9/14
"""
import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def load_hd_map(filename):
    data = np.load(filename, allow_pickle=True)
    arr = data['arr']
    lane_marking_dict = dict(arr)
    return lane_marking_dict


def display_lane_data(lane_marking_dict, carla_town):
    fig, ax = plt.subplots(figsize=(12, 8))
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    # Initialize variables to collect all x and y coordinates
    all_x = []
    all_y = []

    # Prepare to plot lane element arrows
    x_coords_not_junction = []
    y_coords_not_junction = []
    u_not_junction = []
    v_not_junction = []

    x_coords_junction = []
    y_coords_junction = []
    u_junction = []
    v_junction = []

    for road_id, data_dict in lane_marking_dict.items():
        if road_id == 'Crosswalks':
            # Handle and plot crosswalks
            crosswalks = data_dict
            for crosswalk_data in crosswalks:
                polygon = crosswalk_data['Polygon']
                x, y = polygon.exterior.xy
                # Add legend only once
                if 'Crosswalk' not in ax.get_legend_handles_labels()[1]:
                    ax.fill(x, y, alpha=0.8, fc='orange', ec='none', label='Crosswalk')
                else:
                    ax.fill(x, y, alpha=0.8, fc='orange', ec='none')
                # Collect coordinates to set axis limits
                all_x.extend(x)
                all_y.extend(y)
            continue  # Continue to the next road_id
        else:
            # Handle and plot trigger volumes
            trigger_volumes = data_dict.get('Trigger_Volumes', [])
            for tv in trigger_volumes:
                polygon = tv['Polygon']
                tv_type = tv['Type']
                # Assign colors based on the trigger volume type
                if tv_type == 'StopSign':
                    color = 'blue'
                elif tv_type == 'TrafficLight':
                    color = 'red'
                else:
                    color = 'magenta'  # Default color

                x, y = polygon.exterior.xy
                # Plot and fill the polygon
                if tv_type not in ax.get_legend_handles_labels()[1]:
                    ax.fill(x, y, alpha=0.8, fc=color, ec='none', label=tv_type)
                else:
                    ax.fill(x, y, alpha=0.8, fc=color, ec='none')
                # Collect coordinates to set axis limits
                all_x.extend(x)
                all_y.extend(y)

            # Handle and plot lane elements
            for lane_id, lane_element_dict in data_dict.items():
                if lane_id == 'Trigger_Volumes':
                    continue  # Trigger volumes have been processed
                lane_mark_dict = lane_element_dict.get('LaneMark', {})

                for key, lane_mark_list in lane_mark_dict.items():
                    for lane_mark in lane_mark_list:
                        points = lane_mark['Points']

                        # Sample points, e.g., take one point every 2 points
                        sample_rate = 1
                        sampled_points = points[::sample_rate]

                        for point in sampled_points:
                            # Get current position and orientation
                            location = point[0]  # (x, y, z)
                            rotation = point[1]  # (roll, pitch, yaw)
                            is_junction = point[2] if len(point) > 2 else False

                            x = location[0]
                            y = location[1]
                            yaw_rad = rotation[2]  # Yaw angle in radians

                            # Arrow direction components
                            dx = 0.5 * np.cos(yaw_rad)  # Arrow length is 0.5 meters
                            dy = 0.5 * np.sin(yaw_rad)

                            if is_junction:
                                x_coords_junction.append(x)
                                y_coords_junction.append(y)
                                u_junction.append(dx)
                                v_junction.append(dy)
                            else:
                                x_coords_not_junction.append(x)
                                y_coords_not_junction.append(y)
                                u_not_junction.append(dx)
                                v_not_junction.append(dy)
                            # Collect coordinates to set axis limits
                            all_x.append(x)
                            all_y.append(y)

    # ------------------------
    # Plot arrows using quiver
    # ------------------------
    if x_coords_not_junction:
        ax.quiver(x_coords_not_junction, y_coords_not_junction, u_not_junction, v_not_junction,
                  color='gray', scale_units='xy', angles='xy', scale=1, width=0.002, label='Not in Junction lane point')
    if x_coords_junction:
        ax.quiver(x_coords_junction, y_coords_junction, u_junction, v_junction,
                  color='green', scale_units='xy', angles='xy', scale=1, width=0.002, label='In Junction lane point')

    # ------------------------
    # Add the legend
    # ------------------------
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # ------------------------
    # Set axis limits
    # ------------------------
    if all_x and all_y:
        ax.set_xlim(min(all_x) - 20, max(all_x) + 20)
        ax.set_ylim(min(all_y) - 20, max(all_y) + 20)

    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    output_directory = 'data/map_data/anno'
    output_filename = f'{carla_town} HD Map visualization.png'
    output_path = os.path.join(output_directory, output_filename)

    # Ensure the directory exists
    Path(output_directory).mkdir(exist_ok=True, parents=True)

    plt.savefig(output_path, dpi=500)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', default='data/map_data')
    parser.add_argument('--carla_town', '-town', default='Town05')

    args = parser.parse_args()
    carla_town = args.carla_town
    filename = os.path.join(args.load_dir, carla_town + '_HD_map.npz')
    lane_marking_dict = load_hd_map(filename)
    display_lane_data(lane_marking_dict, carla_town)

