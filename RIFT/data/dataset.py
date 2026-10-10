#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : dataset.py
@Date    : 2024/7/14
"""
import cv2
import os.path as osp
import h5py
import torch

from torch.utils.data import Dataset, DataLoader
from rift.util.run_util import load_config


class LightningDataset(Dataset):
    def __init__(self, data_location, data_config):
        self.data_location = data_location
        # required data keys
        self.data_keys = data_config['data_keys']
        self.need_image = data_config['need_image']

        # read necessary meta info
        with h5py.File(self.data_location, 'r') as f:
            self.dataset_len = f.attrs['length']
            self.available_datasets = list(f.keys())
            print("keys", self.available_datasets)

        # Delay opening the file handle (done in the __getitem__ method)
        self.file_handle = None

    def _open_file_if_needed(self):
        if self.file_handle is None:
            self.file_handle = h5py.File(self.data_location, 'r')

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        self._open_file_if_needed()

        sample = {}
        for key in self.data_keys:
            prefix = key.split('_')[0]
            if prefix == 'collect':
                # fix length data
                dset = self.file_handle[key]
                data = dset[index]
                data = torch.tensor(data, dtype=torch.float32)
                sample[key] = data

            elif prefix == 'camera' and self.need_image:
                # camera image dataset
                dset = self.file_handle[key]
                image_path = dset[index].decode('utf-8')
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = torch.tensor(image.transpose(2,0,1), dtype=torch.float32)
                sample['image'] = image_tensor

        return sample

    def __del__(self):
        if self.file_handle is not None:
            self.file_handle.close()


if __name__ == '__main__':

    data_filename = 'all_data.hdf5'

    base_directory = 'data/offline_dataset'
    data_config = load_config('data/config/normal.yaml')

    data_path = osp.join(base_directory, data_filename)

    dataset = LightningDataset(data_path, data_config)
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"  {key}: list of length {len(value)}")
            else:
                print(f"  {key}: type={type(value)}")

        if idx >= 1:
            break
