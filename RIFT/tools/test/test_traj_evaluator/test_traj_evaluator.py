#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : test_traj_evaluator.py
@Date    : 2025/02/23
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import random
import matplotlib as mpl
import matplotlib.patches as mpatches
import pandas as pd
import carla
from rift.cbv.planning.fine_tuner.rlft.traj_eval.traj_evaluator import TrajEvaluator

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'lines.linewidth': 2,
    'axes.labelsize': 22,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})
COLORS = {
    'dark_blue': '#2878B5',
    'light_blue': '#9AC9DB',
    'dark_red': '#C82423',
    'light_red': '#F8AC8C',
    'sky_blue': '#82B0D2',
    'sky_red': '#FA7F6F',
}


def plot_rollout_vertices(rollout_vertices_list):
    '''
    Plot vertices for each rollout on the same plot, with different colors for each rollout.
    Time progresses with increasing transparency (alpha).
    
    Args:
    rollout_vertices_list (list): A list of rollout_vertices where each element is an array with shape [N, Ts, 4, 2]
    '''

    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define a list of colors (ensure there are enough colors for your rollouts)
    colors = plt.cm.jet(np.linspace(0, 1, len(rollout_vertices_list)))
    
    # Loop over each rollout_vertices in the list
    for i, (rollout_vertices, color) in enumerate(zip(rollout_vertices_list, colors)):
        
        # Initialize the x and y limits for this subplot
        x_min, x_max, y_min, y_max = float('inf'), -float('inf'), float('inf'), -float('inf')
        
        # Loop over each vertex group in the rollout (each represents vertices for a time step)
        for vertices in rollout_vertices:
            # Calculate alpha transparency based on time step (time progresses -> alpha increases)
            length = len(vertices)
            # Loop over each set of 4 vertices (forming a rectangle)
            for t, vertex in enumerate(vertices):
                # The four vertices are [FL, RL, RR, FR], with each being (x, y)
                alpha = 1 - (t / length)   # 0 -> fully opaque, 1 -> fully transparent
                FL, RL, RR, FR = vertex

                # Update the x and y bounds
                x_min = min(x_min, FL[0], RL[0], RR[0], FR[0])
                x_max = max(x_max, FL[0], RL[0], RR[0], FR[0])
                y_min = min(y_min, FL[1], RL[1], RR[1], FR[1])
                y_max = max(y_max, FL[1], RL[1], RR[1], FR[1])

                # Create a Rectangle patch (polygon) with a different color for each rollout
                rect = patches.Polygon(
                    [FL, RL, RR, FR],  # Use the 4 corners to create the polygon
                    closed=True,
                    linewidth=2,
                    edgecolor=color,  # Color border based on the rollout
                    facecolor=color,  # Color fill based on the rollout
                    alpha=alpha  # transparency based on time step
                )
                # Add the rectangle (polygon) to the plot
                ax.add_patch(rect)
        
    # Set plot labels and title for the combined plot
    ax.set_title('Rollouts on Same Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set plot limits with some padding around the data range
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    
    # Set equal scaling for both axes
    ax.set_aspect('equal', adjustable='box')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig('tools/test/test_traj_evaluator/combined_rollout_plot.png')

    plt.close()


def visualize_ref_line_validation(trajectories, ref_line_pos, ref_line_angle, 
                                 delta_dis, delta_angle, 
                                 r_idx=0, m_idx=0, ts_step=1):

    R, M, Ts, _ = trajectories.shape
    delta_dis = delta_dis.reshape(R, M, Ts)
    delta_angle = delta_angle.reshape(R, M, Ts)

    ref_pos = ref_line_pos[r_idx]  # [120, 2]
    ref_angle = ref_line_angle[r_idx]

    ref_orient = np.stack([np.cos(ref_angle), 
                          np.sin(ref_angle)], axis=-1)  # [120, 2]
    traj = trajectories[r_idx, m_idx]  # [Ts, 4]
    traj_pos = traj[:, :2]  # [Ts, 2]
    traj_orient = traj[:, 2:4]  # [Ts, 2]
    
    closest_idx = np.argmin(np.linalg.norm(
        traj_pos[:, np.newaxis] - ref_pos, axis=-1), axis=-1)  # [Ts]
    closest_pos = ref_pos[closest_idx]  # [Ts, 2]
    closest_orient = ref_orient[closest_idx]  # [Ts, 2]

    plt.figure(figsize=(18, 6))
    
    ax1 = plt.subplot(1, 3, 1)
    
    ax1.scatter(ref_pos[:10, 0], ref_pos[:10, 1], c='gray', s=20, alpha=0.5, label='Reference Points')
    ax1.plot(ref_pos[:10, 0], ref_pos[:10, 1], '--', lw=1, c='gray', alpha=0.5, label='Reference Line')
    
    colors = plt.cm.jet(np.linspace(0, 1, Ts))
    ax1.scatter(traj_pos[:, 0], traj_pos[:, 1], c=colors, s=30, label='Trajectory')
    
    for t in range(0, Ts, ts_step):
        ax1.plot([traj_pos[t, 0], closest_pos[t, 0]],
                 [traj_pos[t, 1], closest_pos[t, 1]], 
                 c='orange', alpha=0.5, lw=0.8)
        
    ax1.quiver(ref_pos[:10:, 0], ref_pos[:10:, 1],
              ref_orient[:10:, 0], ref_orient[:10:, 1],
              color='gray', scale=15, width=0.003, label='Ref Orientation')
    
    ax1.quiver(traj_pos[::ts_step, 0], traj_pos[::ts_step, 1],
              traj_orient[::ts_step, 0], traj_orient[::ts_step, 1],
              color=colors[::ts_step], scale=15, width=0.004, label='Traj Orientation')
    
    ax1.set_title(f'Spatial Relationship\n(R={r_idx}, M={m_idx})')
    ax1.axis('equal')
    ax1.legend(loc='upper right')
    
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(delta_dis[r_idx, m_idx], 'b-o', lw=2, markersize=4, label='Computed Distance')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Distance to Reference Line')
    ax2.grid(True)
    
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(np.degrees(delta_angle[r_idx, m_idx]), 'r-o', lw=2, markersize=4, label='Computed Angle')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Angle Difference (deg)')
    ax3.set_title('Orientation Difference')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('tools/test/test_traj_evaluator/ref_info.png')
    plt.close()

def test_other_vehicle_rollout():
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

        traj_evaluator = TrajEvaluator()

        for _ in range(20):
            world.tick()
        
        vertices_list = []

        for _ in range(10):
            world.tick()
            # test nearby agent rollout
            vertices_list.append(traj_evaluator.get_other_vehicle_rollout(nearby_actors=[ego],num_future_frames=80))  # [N, Ts, ...]

        print('finished forward simulation')
        plot_rollout_vertices(vertices_list)
        print('finished plotting')            

    finally:
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

def test_saved_data():
    with open('tools/test/test_traj_evaluator/traj_eval_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    raw_trajectories = loaded_data['raw_trajectories']
    ref_line_pos = loaded_data['ref_line_pos']
    ref_line_angle = loaded_data['ref_line_angle']
    delta_dis = loaded_data['delta_dis']
    delta_angle = loaded_data['delta_angle']
    rollout_center = loaded_data['rollout_center']
    rollout_angle = loaded_data['rollout_angle']
    rollout_speed = loaded_data['rollout_speed']
    rollout_acc = loaded_data['rollout_acc']
    rollout_vertices = loaded_data['rollout_vertices']
    nearby_agents_rollout_vertices = loaded_data['nearby_agents_rollout_vertices']
    collision_matrix = loaded_data['collision_matrix']
    off_road_matrix = loaded_data['off_road_matrix']
    rollout_return = loaded_data['rollout_return']

    # ref line success
    visualize_ref_line_validation(raw_trajectories, ref_line_pos, ref_line_angle, delta_dis, delta_angle, r_idx=0, m_idx=9, ts_step=1)
    
    # rollout success
    plot_rollout_vertices([rollout_vertices, nearby_agents_rollout_vertices])

def test_rollout_control():
    # Read the dataset from a CSV file
    df = pd.read_csv('tools/test/test_traj_evaluator/rollout_data.csv')

    # Define a function to calculate the average of the first 'window_size' elements
    def calculate_average(arr, window_size=10):
        if len(arr) >= window_size:
            return np.mean(arr[0:window_size])  # Compute the average of the first 'window_size' elements
        else:
            return np.mean(arr)  # If there are fewer elements than 'window_size', compute the average of all elements

    # Convert the string representation of lists in 'rollout_speed' and 'rollout_acc' columns to actual lists,
    # then calculate the average of the first 10 elements for each row
    df['rollout_speed_avg'] = df['rollout_speed'].apply(eval).apply(calculate_average)
    df['rollout_acc_avg'] = df['rollout_acc'].apply(eval).apply(calculate_average)

    # Shift the 'speed' and 'acc' columns by 1 to create 'speed_shifted' and 'acc_shifted' for proper alignment
    df['speed_shifted'] = df['speed'].shift(-1)  # Shift speed by 1 to align with the shifted time axis
    df['acc_shifted'] = df['acc'].shift(-1)  # Shift acceleration by 1 to align with the shifted time axis

    # Set up the plot size
    plt.figure(figsize=(12, 10))  # Increase the figure size for better readability

    # Plot the rollout speed and speed_shifted in the first subplot (top part of the figure)
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, choose the first subplot
    plt.plot(df.index, df['rollout_speed_avg'], label='Traj. Reference Speed', color=COLORS['dark_blue'], linestyle='--')
    plt.plot(df.index, df['speed_shifted'], label='Actural Speed', color=COLORS['light_blue'], linestyle='-')
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Speed $(\mathit{m/s})$')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines for better readability

    # Plot the throttle and acceleration (shifted) in the second subplot (bottom part of the figure)
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, choose the second subplot
    ax1 = plt.gca()  # Get the current axes for the first plot

    # Plot acceleration (shifted) on the left y-axis
    line1, = ax1.plot(df.index, df['acc_shifted'], label='Acceleration', color=COLORS['light_red'], linestyle='-')
    ax1.set_xlabel('Time $(ms)$')
    ax1.set_ylabel('Acceleration $(\mathit{m/s^2})$', labelpad=2)
    # ax1.tick_params(axis='y', labelcolor=COLORS['light_red'])  # Set the y-axis label color for acceleration
    ax1.grid(True, linestyle='--', alpha=0.7)  # Add grid lines for better readability

    # Create a second y-axis for plotting throttle on the right
    ax2 = ax1.twinx()  # Create a twin y-axis for throttle
    line2, = ax2.plot(df.index, df['throttle'], label='Throttle', color=COLORS['dark_red'], linestyle='-')
    ax2.set_ylabel('Throttle')
    # ax2.tick_params(axis='y', labelcolor=COLORS['dark_red'])  # Set the y-axis label color for throttle

    # Add vertical lines and fill for throttle = 0 intervals
    zero_throttle_times = df.index[df['throttle'] <= 0.01].tolist()  # Find all time points where Throttle is 0
    
    # Find the intervals where Throttle is zero
    zero_intervals = []
    start = zero_throttle_times[0] if zero_throttle_times else None  # Start of the first interval
    for i in range(1, len(zero_throttle_times)):
        if zero_throttle_times[i] != zero_throttle_times[i-1] + 1:  # Interval ends when the times are not consecutive
            end = zero_throttle_times[i-1]
            if start is not None and end is not None:  # Check if both start and end are valid
                zero_intervals.append((start, end))
            start = zero_throttle_times[i]
    if start is not None and zero_throttle_times:
        zero_intervals.append((start, zero_throttle_times[-1]))  # Add the last interval if it's valid

    # Get the y-limits from both axes to cover the full range
    y1_min, y1_max = ax1.get_ylim()  # Left axis limits (throttle)
    y2_min, y2_max = ax2.get_ylim()  # Right axis limits (acceleration)
    
    # Calculate the overall y-range covering both axes
    y_min = min(y1_min, y2_min)
    y_max = max(y1_max, y2_max)

    # Combine the legends of both lines
    lines = [line1, line2]  # Combine the line handles for the legend
    labels = [line1.get_label(), line2.get_label()]  # Get the labels for the legend

    brake_time_patch = mpatches.Patch(color='gray', alpha=0.3, label='Braking Interval')

    # Add the Patch to the legend
    lines.append(brake_time_patch)
    labels.append('Braking Interval')

    # Draw vertical lines and shaded areas for each interval where Throttle is 0
    for start, end in zero_intervals:
        if start is not None and end is not None:  # Ensure that start and end are valid numbers
            # Draw dashed lines at the start and end of the interval
            ax1.axvline(x=start, color='gray', linestyle=':', linewidth=1.5)
            ax1.axvline(x=end, color='gray', linestyle=':', linewidth=1.5)
            
            # Fill the area between the lines with a transparent gray color
            ax1.fill_between(df.index, y_min, y_max, where=(df.index >= start) & (df.index <= end), 
                             color='gray', alpha=0.3, label='Braking Interval' if start == zero_intervals[0][0] else "")
    ax2.legend(lines, labels, loc='center', bbox_to_anchor=(0.4, 0.3))  # Show the combined legend on the second axis
    
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, choose the second subplot
    ax3 = plt.gca()  # Get the current axes for the first plot
    first_acc = df['acc'].iloc[12]
    first_rollout_acc = df['rollout_acc'].apply(eval)[12]
    ax3.plot(first_rollout_acc, label='Traj. Acceleration over Virtural Time', color=COLORS['sky_blue'], linestyle='-')
    ax3.axhline(y=first_acc, color=COLORS['sky_red'], label=f'Actural Acceleration', linestyle='--', linewidth=2)
    ax3.set_xlabel('Virtual Time in Trajectory $(ms)$')
    ax3.set_ylabel('Acceleration $(\mathit{m/s^2})$', labelpad=2)
    ax3.grid(True, linestyle='--', alpha=0.7)  # Add grid lines for better readability
    ax3.legend(loc='center', bbox_to_anchor=(0.7, 0.4))

    # Adjust layout to avoid overlap between subplots
    plt.tight_layout()

    # Save the figure to a PNG file
    plt.savefig('tools/test/test_traj_evaluator/acc_rollout_plot.png', dpi=600)

    # Close the plot to release resources
    plt.close()

def main():
    # 1. Test other vehicle rollout
    # test_other_vehicle_rollout()  # success

    # 2. Test the collision matrix and off road result
    # test_saved_data()

    # 3. Test the rollout control
    test_rollout_control()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('done.')


