#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : plot_reward.py
@Date    : 2025/02/26
'''
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
from palettable.colorbrewer.qualitative import Dark2_8 


def plot_training_curves(base_dir: Path, smooth_factor: float = 0.9):
    """Plot training curves from route_info.txt files in subdirectories of base_dir"""
    
    # Set Times New Roman font with fallback
    try:
        # Try to use system's Times New Roman
        font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        # Fallback to DejaVu Sans if not found
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("Times New Roman not found, using DejaVu Sans as fallback")

    plt.figure(figsize=(8, 6))
    colors = Dark2_8.mpl_colors
    
    # Flag to track if any valid route_info.txt files are found
    found_valid_file = False
    
    # Iterate through all entries in base directory
    for idx, folder_path in enumerate(base_dir.iterdir()):
        if not folder_path.is_dir():
            continue
        
        # Parse folder name components
        folder_name = folder_path.name
        parts = folder_name.split('-')
        if len(parts) != 4:
            continue
        
        ego_name, cbv_name, _, seed = parts
        
        # Check for route_info.txt existence
        data_file = folder_path / 'route_info.txt'
        if not data_file.exists():
            continue
        
        # Update validation flag
        found_valid_file = True
        
        # Process data
        episodes = []
        raw_rewards = []
        with data_file.open('r') as f:
            for line in f:
                entries = line.strip().split(', ')
                episode = int(entries[0].split(': ')[1])
                reward = float(entries[4].split(': ')[1])
                
                if reward != 0.0:
                    episodes.append(episode)
                    raw_rewards.append(reward)
        
        # Calculate smoothed rewards using exponential moving average
        smoothed = []
        last = raw_rewards[0] if raw_rewards else 0
        for reward in raw_rewards:
            smoothed_val = last * smooth_factor + (1 - smooth_factor) * reward
            smoothed.append(smoothed_val)
            last = smoothed_val
        
        # Create label with configuration
        label = f"Ego: {ego_name} CBV: {cbv_name}"
        color = colors[idx % len(colors)]
        
        # Plot both raw and smoothed lines
        plt.plot(episodes, raw_rewards, 
                 alpha=0.2,
                 color=color,
                 linewidth=1.5)
        
        plt.plot(episodes, smoothed,
                 alpha=0.8,
                 color=color,
                 linewidth=2.5,
                 label=f'{label}')

    # Early exit if no valid files found
    if not found_valid_file:
        print(f">> No valid route_info.txt files found in {base_dir}")
        return

    # Configure plot settings
    plt.title('Training Episode Rewards', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend with improved positioning
    legend = plt.legend(
        loc='best', 
        title='Configuration',
        title_fontsize=10,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor='gray'
    )
    legend.get_frame().set_linewidth(0.5)
    
    # Create output directory if needed
    output_dir = Path('assets') / base_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Add space for legend
    plt.savefig(output_dir / 'episode_reward.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Successfully save episode reward plot of {base_dir.name}')

def main():

    base_dirs = [Path(args.base_dir) / folder for folder in args.process_folder]
    for base_dir in base_dirs:
        plot_training_curves(base_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='log')
    parser.add_argument('--process_folder', '-f', type=str, default=['train_ego', 'train_cbv', 'eval'])
    args = parser.parse_args()

    main()
