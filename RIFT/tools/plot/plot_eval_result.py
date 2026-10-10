#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : plot_eval_result.py
@Date    : 2025/03/16
'''
import ast
import argparse
import csv
import statistics
import json
import math
import numpy as np

from pathlib import Path
from scipy import stats
from tabulate import tabulate
from collections import defaultdict


def parse_value(v):
    """
    Parse a value from string to appropriate data type (int, float, or original string).
    """
    if not isinstance(v, str):
        return v
    try:
        return ast.literal_eval(v)
    except (SyntaxError, ValueError):
        pass
    try:
        if '.' in v:
            return float(v)
        else:
            return int(v)
    except ValueError:
        pass
    return v

def load_labels_values_as_dict(json_path: str):
    """
    Load JSON file and parse labels and values into a dictionary.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    labels = data.get("labels", [])
    values = data.get("values", [])
    results_dict = {}
    for label, value in zip(labels, values):
        results_dict[label] = parse_value(value)
    return results_dict, data


def parse_metric(metric_str):
    """
    Parse a metric string formatted as "mean ± std" into a tuple of floats (mean, std).
    """
    parts = metric_str.split("±")
    mean_val = float(parts[0].strip())
    std_val = float(parts[1].strip())
    return mean_val, std_val

def process_multifile_metrics(json_paths: list):
    """
    Process multiple JSON files containing metrics from different seeds and aggregate the results.
    The final output is a unified dictionary where each metric is represented as "mean ± std".
    For a single seed, the std is set to 0.

    Parameters:
        json_paths (list): List of JSON file paths to process.
        experiment_count (int): Number of experiments (currently not used, but available for future customization).

    Returns:
        aggregated_metrics (dict): Aggregated metrics for all keys, formatted as "mean ± std".
    """
    # Dictionaries to store values across all seeds.
    # single_value_all will hold numerical values for each key.
    single_value_all = {}
    # mean_std_all will hold parsed means and stds for each key.
    mean_std_all = {}
    
    # Process each JSON file.
    for json_path in json_paths:
        single_value, mean_std_value = process_onefile_metrics(Path(json_path))
        
        # Aggregate single_value metrics.
        for key, value in single_value.items():
            single_value_all.setdefault(key, []).append(value)
            
        # Aggregate mean_std_value metrics by parsing the "mean ± std" string.
        for key, metric_str in mean_std_value.items():
            mean_val, std_val = parse_metric(metric_str)
            if key not in mean_std_all:
                mean_std_all[key] = {"means": [], "stds": []}
            mean_std_all[key]["means"].append(mean_val)
            mean_std_all[key]["stds"].append(std_val)
    
    # Aggregate the results into a unified dictionary.
    aggregated_metrics = {}
    
    # Process single_value metrics: calculate the average and sample standard deviation.
    for key, values in single_value_all.items():
        mean_val = statistics.mean(values)
        # If there's only one seed, set std to 0.
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        aggregated_metrics[key] = f"{mean_val:.2f} ± {std_val:.2f}"
    
    # Process mean_std_value metrics.
    for key, data in mean_std_all.items():
        means = data["means"]
        stds = data["stds"]
        aggregated_mean = statistics.mean(means)
        if len(means) > 1:
            # Combine individual variances and the variance of the means:
            # aggregated_variance = mean(individual variances) + variance(means)
            aggregated_variance = statistics.mean([s**2 for s in stds]) + statistics.variance(means)
        else:
            aggregated_variance = statistics.mean([s**2 for s in stds])
        aggregated_std = math.sqrt(aggregated_variance)
        aggregated_metrics[key] = f"{aggregated_mean:.2f} ± {aggregated_std:.2f}"
    
    return aggregated_metrics

def parse_hist_dict(hist_dict):
    """
    Parse histogram dictionary into a list of tuples (left, right, count).
    """
    parsed = []
    for bin_str, count in hist_dict.items():
        try:
            left_str, right_str = bin_str.split('~')
            left_val = float(left_str)
            right_val = float(right_str)
            parsed.append((left_val, right_val, count))
        except:
            pass
    parsed.sort(key=lambda x: x[0])  # Sort by the left boundary of bins
    return parsed

def sample_from_hist(parsed_bins, n_samples=3000, seed=0):
    """
    Generate random samples from histogram bins based on their counts.
    """
    np.random.seed(seed)

    total_count = sum(bin_item[2] for bin_item in parsed_bins)
    if total_count == 0:
        return np.zeros(n_samples)

    probs = [bin_item[2]/total_count for bin_item in parsed_bins]
    cum_probs = np.cumsum(probs)
    rand_vals = np.random.rand(n_samples)
    samples = np.zeros(n_samples)

    for i in range(n_samples):
        r = rand_vals[i]
        idx = np.searchsorted(cum_probs, r)
        left, right, _ = parsed_bins[idx]
        samples[i] = np.random.uniform(left, right)  # Sample uniformly within the bin
    return samples

def process_normal_simularity(data):
    """
    Compute similarity of the input data to a normal distribution using two widely-used tests:
    - Shapiro-Wilk W statistic (closer to 1 means more normal)
    - Anderson-Darling A² statistic (closer to 0 means more normal)

    Parameters
    ----------
    data : np.ndarray
        1D array of input data (e.g., acceleration values)

    Returns
    -------
    dict
        Dictionary with two keys:
            'shapiro_w' : float
                Shapiro-Wilk W statistic (ideal ~1.0)
            'anderson_a2' : float
                Anderson-Darling statistic A² (ideal ~0.0)
    """

    # Shapiro–Wilk test: good for small-medium sample sizes
    shapiro_w, _ = stats.shapiro(data)

    # Anderson–Darling test: sensitive to tail differences
    anderson_result = stats.anderson(data, dist='norm')
    anderson_a2 = anderson_result.statistic

    return shapiro_w, anderson_a2

def compute_metric_stats(data_dict, speed_threshold=3.0):
    """
    Compute the weighted mean and variance of the metric (e.g., RTTC, EI, ACT)
    for entries where the lower bound of speed > speed_threshold.

    Parameters:
        data_dict (dict): Dictionary with keys in the format
                          'speedX~Y_{metric}A~B' and integer values as counts.
        speed_threshold (float): Lower bound threshold for speed filtering.

    Returns:
        str: Formatted string in the form of "mean ± variance",
             or "No valid data" if nothing matches.
    """
    total_count = 0
    weighted_sum = 0.0
    weighted_sum_squared = 0.0

    for key, count in data_dict.items():
        if not key.startswith("speed"):
            continue  # Skip malformed keys

        try:
            # Split into speed and metric parts
            speed_part, metric_part = key.split('_')
            
            # Extract speed bounds
            speed_lower, speed_upper = map(float, speed_part.replace('speed', '').split('~'))
            
            # Extract metric name and bounds
            metric_name = ''.join([c for c in metric_part if not c.isdigit() and c not in '.~'])
            metric_range = metric_part.replace(metric_name, '')
            metric_lower, metric_upper = map(float, metric_range.split('~'))
            metric_mid = (metric_lower + metric_upper) / 2

            # Filter based on speed threshold
            if speed_lower >= speed_threshold:
                total_count += count
                weighted_sum += count * metric_mid
                weighted_sum_squared += count * (metric_mid ** 2)

        except Exception as e:
            print(f"Skipping invalid key '{key}': {e}")
            continue

    if total_count == 0:
        return "No valid data"

    # Calculate weighted mean and variance
    mean = weighted_sum / total_count
    variance = (weighted_sum_squared / total_count) - (mean ** 2)

    return f"{mean:.2f}±{variance:.2f}"

def process_onefile_metrics(json_path: Path):
    """Example processing function - customize this based on your needs"""
    # Sample processing: calculate derived metrics or filter data
    dict_data, data = load_labels_values_as_dict(json_path)
    progress = data["_checkpoint"]["progress"][-1]

    # acc and speed
    acc_bins = parse_hist_dict(dict_data["cbv_acc_distribution"])
    acc_samples = sample_from_hist(acc_bins)

    speed_bins = parse_hist_dict(dict_data["cbv_speed_distribution"])
    speed_samples = sample_from_hist(speed_bins)

    # simularity of the raw acc with the normal distribution
    shapiro_w_acc, anderson_a2_acc = process_normal_simularity(acc_samples)
    shapiro_w_speed, anderson_a2_speed = process_normal_simularity(speed_samples)

    # Ego got block
    exceptions = dict_data["exceptions"]
    ego_got_block_num = sum(1 for item in exceptions if item[2] == 'Failed - Agent got blocked')
    ego_got_block_ratio = round(ego_got_block_num / progress * 100, 2)

    # Off-road ratio
    cbv_off_road_time = dict_data["cbv_off_road_game_time"]
    totol_game_time = dict_data["cbv_total_game_time"]
    cbv_off_road_ratio = round(cbv_off_road_time / totol_game_time * 100, 2)

    # Total progress (m)
    cbv_progress = dict_data["cbv_progress"]
    
    # Collision ratio
    cbv_num = dict_data["cbv_count"]
    cbv_collision = dict_data["cbv_collision_count"]
    collision_per_km = round(cbv_collision / cbv_progress * 1000, 2)

    # Reach goal ratio
    cbv_reach_goal = dict_data["cbv_reach_goal_count"]
    reach_goal_per_cbv = round(cbv_reach_goal / cbv_num * 100, 2)

    # ACC, Speed, Jerk
    cbv_acc = f"{dict_data['cbv_acc_mean']}±{dict_data['cbv_acc_std']}"
    cbv_speed = f"{dict_data['cbv_speed_mean']}±{dict_data['cbv_speed_std']}"
    cbv_jerk = f"{dict_data['cbv_jerk_mean']}±{dict_data['cbv_jerk_std']}"
    
    # delta / Wasserstein distance with the target speed
    delta_speed = f"{dict_data['cbv_delta_speed_mean']}±{dict_data['cbv_delta_speed_std']}"
    speed_W_dis = math.sqrt((dict_data['cbv_speed_mean'] - dict_data['cbv_target_speed_mean'])**2 + (dict_data['cbv_speed_std'] - dict_data['cbv_target_speed_std'])**2)
    
    # Uncomfortable ratio
    cbv_uncomfortable_game_time = dict_data["cbv_uncomfortable_game_time"]
    cbv_uncomfortable_ratio = round(cbv_uncomfortable_game_time / totol_game_time * 100, 2)
    
    # Ego safety-critical metrics
    EI_dist = dict_data["ego_EI_distribution"]
    EI = compute_metric_stats(EI_dist)
    RTTC_dist = dict_data["ego_RTTC_distribution"]
    RTTC = compute_metric_stats(RTTC_dist)
    ACT_dist = dict_data["ego_ACT_distribution"]
    ACT = compute_metric_stats(ACT_dist)

    single_value = {
        "Driving Score (↑)": dict_data["Avg. driving score"],       # Drving Score of Ego
        "Route Completion (↑)": dict_data["Avg. route completion"],    # Route Completion of Ego
        "Infraction Penalty (↑)": dict_data["Avg. infraction penalty"],  # Infraction Penalty of Ego
        "Ego Blocked Ratio (↓)": ego_got_block_ratio,                   # Block ratio of Ego
        "ORR (↓)": cbv_off_road_ratio,                    # Off-road ratio of CBV
        "UC (%)": cbv_uncomfortable_ratio,               # Uncomfortable ratio of CBV
        "CPK (↓)": collision_per_km,                # Collision per km of CBV
        # "RG (%)": reach_goal_per_cbv,                    # Reach goal ratio per CBV
        "RP (↑)": cbv_progress,                         # Total progress of CBV
        "SW speed (↑)": shapiro_w_speed,             # Shapiro-Wilk W statistic of CBV speed
        # "AD speed (↓)": anderson_a2_speed,      # Anderson-Darling A² statistic of CBV speed
        "WD speed(↓)": speed_W_dis,                      # Wasserstein distance of the CBV speed with the target speed
        "SW acc (↑)": shapiro_w_acc,                 # Shapiro-Wilk W statistic of CBV acceleration
        # "AD acc (↓)": anderson_a2_acc,          # Anderson-Darling A² statistic of CBV acceleration
    }
    mean_std_value = {
        "RTTC (↑)": RTTC,                                # RTTC of Ego
        "ACT (↑)": ACT,                                  # ACT of Ego
        # "EI (↓)": EI,                                    # EI of Ego
        "Acc (m/s²)": cbv_acc,                           # Acceleration of CBV
        "Speed (m/s)": cbv_speed,                        # Speed of CBV
        "Jerk (m/s³)": cbv_jerk,                         # Jerk of CBV
        # "ΔSpeed (m/s²)(↓)": delta_speed,                 # Delta Speed of CBV
    }
    return single_value, mean_std_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='log/eval')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    results = defaultdict(dict)  # Nested dictionary: results[group][variant] = metrics
    group_files = defaultdict(list)

    # Scan all subdirectories under base_dir
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            json_path = subdir / "simulation_results.json"
            parts = subdir.name.split('-')
            group = parts[0]
            variant = parts[1]
            group_name = f"{group}-{variant}"
            group_files[group_name].append(json_path)

    # Process and store metrics into nested dictionary
    for group_name, json_paths in group_files.items():
        parts = group_name.split('-')
        group = parts[0]
        variant = parts[1]
        # Rename special case
        if variant == "rift_pluto":
            variant = "RIFT (ours)"
        try:
            metrics = process_multifile_metrics(json_paths)
            results[group][variant] = metrics
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error processing {json_paths}: {e}")
            continue

    if not results:
        print("No valid results found")
        exit()

    # Extract header fields from sample entry
    sample_group = next(iter(results.values()))
    sample_variant = next(iter(sample_group.values()))
    headers = ["Method"] + list(sample_variant.keys())

    # Custom desired order of variants for consistent display
    desired_order = [
        "pluto", "ppo", "frea", "fppo_rs", "sft_pluto",
        "rs_pluto", "rtr_pluto", "ppo_pluto", "reinforce_pluto", "grpo_pluto", "RIFT (ours)"
    ]

    # Print formatted tables grouped by first-level method (group)
    for group, variants in results.items():
        print(f"\n===== Results for {group} =====")
        table = []
        for variant in desired_order:
            if variant in variants:
                metrics = variants[variant]
                row = [variant] + [f"{v:.2f}" if isinstance(v, float) else str(v)
                                   for v in metrics.values()]
                table.append(row)

        print(tabulate(table,
                       headers=headers,
                       tablefmt="pretty",
                       colalign=("center",) * len(headers),
                       missingval="N/A"))

    # Save the results to a CSV file, grouped by method
    csv_path = "assets/results.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for group, variants in results.items():
            writer.writerow([f"Group: {group}"])
            writer.writerow(headers)
            for variant in desired_order:
                if variant in variants:
                    metrics = variants[variant]
                    row = [variant] + [f"{v:.2f}" if isinstance(v, float) else str(v)
                                       for v in metrics.values()]
                    writer.writerow(row)
            writer.writerow([])  # Empty line to separate groups