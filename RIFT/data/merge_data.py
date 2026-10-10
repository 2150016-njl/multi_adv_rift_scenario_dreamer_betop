#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : merge_data.py
@Date    : 2024/12/12
"""
import h5py
import os

import numpy as np


def merge_all_data(base_directory, output_file):
    all_data_file = os.path.join(base_directory, output_file)
    if not os.path.exists(all_data_file):
        print("\nMerging all town_data.hdf5 files into all_data.hdf5...")
        merged_data = {}
        total_length = 0
        attribute_values = {}  # Dictionary for consistent attribute handling

        for folder in os.listdir(base_directory):
            folder_path = os.path.join(base_directory, folder)
            town_data_path = os.path.join(folder_path, 'town_data.hdf5')

            if os.path.isfile(town_data_path):
                print(f"Processing town_data.hdf5 from folder: {folder}")
                with h5py.File(town_data_path, 'r') as file:
                    # Handle attributes
                    total_length += file.attrs.get('length', 0)
                    for attr_key, attr_value in file.attrs.items():
                        if attr_key == 'length':  # Skip length as it's being summed
                            continue
                        standardized_value = tuple(attr_value) if isinstance(attr_value, (np.ndarray, list, tuple)) else attr_value
                        if attr_key not in attribute_values:
                            attribute_values[attr_key] = standardized_value
                        elif attribute_values[attr_key] != standardized_value:
                            raise ValueError(f"Inconsistent attribute '{attr_key}' in {town_data_path}: "
                                            f"{standardized_value} != {attribute_values[attr_key]}")

                    # Merge datasets
                    for key in file.keys():
                        data = file[key][:]
                        if key in merged_data:
                            # Concatenate along the first axis
                            merged_data[key] = np.concatenate((merged_data[key], data), axis=0)
                        else:
                            # Initialize with the first dataset
                            merged_data[key] = data

        # Save the final merged file
        with h5py.File(all_data_file, 'w') as out_file:
            # Write attributes
            out_file.attrs['length'] = total_length
            print(f"  length: {total_length}")
            for attr_key, attr_value in attribute_values.items():
                out_file.attrs[attr_key] = attr_value
                print(f"  {attr_key}: {attr_value}")
            # Write merged datasets
            for key, data in merged_data.items():
                out_file.create_dataset(key, data=data, compression='gzip')
                print(f"  {key}: shape={data.shape}")

        print(f"\nFinal merged HDF5 file saved to {all_data_file}")
    else:
        print("\nall_data.h5df already exists.")


def merge_town_files(base_directory, output_file):
    """
    Merge multiple HDF5 files into a single file without creating groups.
    
    Args:
        base_directory (str): Directory containing HDF5 files to merge.
        output_file (str): Path to the output HDF5 file.
    """
    if not os.path.exists(output_file):
        merged_data = {}  # Dictionary to store merged data
        total_length = 0
        attribute_values = {}

        # Iterate over all HDF5 files in the base directory
        for file_name in os.listdir(base_directory):
            file_path = os.path.join(base_directory, file_name)

            if file_name.endswith('.hdf5'):
                print(f"Processing file: {file_name}")
                with h5py.File(file_path, 'r') as file:

                    # Check consistency of other attributes
                    total_length += file.attrs.get('length', 0)
                    for attr_key, attr_value in file.attrs.items():
                        if attr_key == 'length':  # Skip 'length', already processed
                            continue
                        standardized_value = tuple(attr_value) if isinstance(attr_value, (np.ndarray, list, tuple)) else attr_value
                        if attr_key not in attribute_values:
                            attribute_values[attr_key] = standardized_value
                        elif attribute_values[attr_key] != standardized_value:
                            raise ValueError(f"Inconsistent attribute '{attr_key}' in {file_name}: "
                                            f"{standardized_value} != {attribute_values[attr_key]}")

                    # Merge datasets
                    for key in file.keys():
                        data = file[key][:]
                        if key in merged_data:
                            # Concatenate along the first axis
                            merged_data[key] = np.concatenate((merged_data[key], data), axis=0)
                        else:
                            # Initialize with the first dataset
                            merged_data[key] = data

        # Save merged data to the output file
        with h5py.File(output_file, 'w') as out_file:
            # Write attributes
            out_file.attrs['length'] = total_length
            print(f"  length: {total_length}")
            for attr_key, attr_value in attribute_values.items():
                out_file.attrs[attr_key] = attr_value
                print(f"  {attr_key}: {attr_value}")
            
            # Write merged datasets
            for key, data in merged_data.items():
                out_file.create_dataset(key, data=data, compression='gzip')
                print(f"  {key}: shape={data.shape}")

        print(f"\nMerged HDF5 file saved to {output_file}")
    else:
        print("already exists merged data")


if __name__ == '__main__':
    base_directory = 'data/offline_dataset'

    # merge as group
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        
        if os.path.isdir(folder_path) and any(file.endswith('.hdf5') for file in os.listdir(folder_path)):
            print(f"Processing files in folder: {folder}")

            output_filepath = os.path.join(folder_path, 'town_data.hdf5')
            # step 1. merge all data in the folder
            merge_town_files(folder_path, output_filepath)
    
    # step 2. merge all data in the base directory
    merge_all_data(base_directory, 'all_data.hdf5')

