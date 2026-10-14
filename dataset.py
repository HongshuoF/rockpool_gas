import random

import torch

from torch.utils.data import Dataset
import numpy as np
import config

cfg = config.Config()
dataset_root = cfg.dataset_root  # Dataset root directory.
def dat_read(fileName):
    with open(fileName, 'r') as file:
        data = file.readlines()
    return data


def data_normed(data, scale):
    if len(data) != len(scale):
        print("ERROR IN NORM!!")
        return data
    # norm = data
    norm = np.zeros(len(data))  # Do not alias data: modifying norm would also change data and self.all_data.
    for i in range(len(data)):
        norm[i] = (data[i] - scale[i][1]) / (scale[i][0] - scale[i][1])
    return norm

def data_split(length, train_propotion, val_propotion):
    indices = list(range(length))
    random.shuffle(indices)
    train_end = int(length * train_propotion)
    val_end = train_end + int(length * val_propotion)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    return train_indices, val_indices, test_indices

class MyDataset(Dataset):
    def __init__(self, dataset_path, label_map=None, label_list=None, sensor_num=8, sensor_start=1, samples_perdat=30):
        full_dataset_path = dataset_root + dataset_path
        # if file name does not end with .csv, raise an error
        if not full_dataset_path.endswith('.csv'):
            raise ValueError("daset_path must be a .csv file, please check the path: %s" % full_dataset_path)
        with open(full_dataset_path, 'r') as f:
            lines = f.readlines()
        # if lines is empty, raise an error
        if not lines:
            raise ValueError("file is empty, please check the path: %s" % full_dataset_path)
        self.length = len(lines)
        label_map = label_list if label_map is None else label_map

        features_list = []
        labels_list = []

        for line in lines:
            parts = line.rstrip('\n').split(',')

            # Look up the label using the original labellist[data[0]] convention.
            label_name = parts[0].strip()
            if label_name not in label_map:
                raise KeyError(
                    "未知标签 '%s'，当前 labellist 为: %s。"
                    "请确认 --dataset 与训练时一致，或在 dataset_generalV2.py 中切换 labellist。"
                    % (label_name, list(label_map.keys()))
                )
            label = label_map[label_name]
            labels_list.append(label)

            # Extract the required sensor readings from this row.
            row_feats = [float(x) for x in parts[sensor_start : sensor_start + sensor_num * samples_perdat]]  
            features_list.append(row_feats)

        # Convert to NumPy arrays; features have shape [N, sensorlen * samples_perdat].
        features = np.asarray(features_list, dtype=np.float32)
        labels = np.asarray(labels_list, dtype=np.int64)

        self.features = features      # Use this array from now on instead of lines / data / data_float.
        self.labels = labels

        # ------- Compute normalization parameters for each sensor channel. -------

        # Reshape to [N, T, C] = [samples, time steps, sensors].
        N, F = self.features.shape
        assert F == sensor_num * samples_perdat, "特征长度和 sensor_num*samples_perdat 不一致"

        feats_3d = self.features.reshape(N, samples_perdat, sensor_num)  # [N, T, C]

        # Compute each channel's max/min over all samples and time steps -> [C].
        max_vals = feats_3d.max(axis=(0, 1))   # Maximum for each sensor.
        min_vals = feats_3d.min(axis=(0, 1))   # Minimum for each sensor.

        # Store one [max, min] pair per channel.
        self.norm_para = np.stack([max_vals, min_vals], axis=1)   # [C, 2]

        # To normalize each flattened sample,
        # repeat the channel max/min values across time to obtain vectors of length F.
        max_repeat = np.tile(max_vals, samples_perdat)   # [F]
        min_repeat = np.tile(min_vals, samples_perdat)   # [F]

        # Normalize all data once in __init__ to avoid recomputing it in __getitem__.
        denom = (max_repeat - min_repeat)
        denom[denom == 0] = 1.0  # Avoid division by zero.
        self.features = (self.features - min_repeat) / denom

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features[idx]          # Normalized NumPy array [F].
        y = self.labels[idx]

        x = torch.from_numpy(x)         # Convert to a tensor for the model.
        return x, y

    def getlen(self):
        return self.length
    