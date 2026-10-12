import random

import torch

from torch.utils.data import Dataset
import numpy as np
import config

cfg = config.Config()
dataset_root = cfg.dataset_root  # 数据集根目录
def dat_read(fileName):
    with open(fileName, 'r') as file:
        data = file.readlines()
    return data


def data_normed(data, scale):
    if len(data) != len(scale):
        print("ERROR IN NORM!!")
        return data
    # norm = data
    norm = np.zeros(len(data))  # !!!!!!特别注意， 这里原来是norm = data 此时norm变会导致data里的数跟着变，直接传导到self.all_data也变，很麻烦
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

            # 标签：沿用你原来的写法 labellist[data[0]]
            label_name = parts[0].strip()
            if label_name not in label_map:
                raise KeyError(
                    "未知标签 '%s'，当前 labellist 为: %s。"
                    "请确认 --dataset 与训练时一致，或在 dataset_generalV2.py 中切换 labellist。"
                    % (label_name, list(label_map.keys()))
                )
            label = label_map[label_name]
            labels_list.append(label)

            # 取这一行中用得到的传感器数据
            row_feats = [float(x) for x in parts[sensor_start : sensor_start + sensor_num * samples_perdat]]  
            features_list.append(row_feats)

        # 转成 numpy 数组，形状 [N, sensorlen * samples_perdat]
        features = np.asarray(features_list, dtype=np.float32)
        labels = np.asarray(labels_list, dtype=np.int64)

        self.features = features      # 之后就只用这个，不再用原来的 lines / data / data_float
        self.labels = labels

        # ------- 计算归一化参数（按传感器通道） -------

        # 先 reshape 成 [N, T, C] = [样本数, 时间步, 传感器数]
        N, F = self.features.shape
        assert F == sensor_num * samples_perdat, "特征长度和 sensor_num*samples_perdat 不一致"

        feats_3d = self.features.reshape(N, samples_perdat, sensor_num)  # [N, T, C]

        # 在所有样本 + 所有时间步上，按通道求 max/min -> [C]
        max_vals = feats_3d.max(axis=(0, 1))   # 每个传感器的最大值
        min_vals = feats_3d.min(axis=(0, 1))   # 每个传感器的最小值

        # 如果你还想保留“每个通道一对 [max, min]”的结构，可以这样存
        self.norm_para = np.stack([max_vals, min_vals], axis=1)   # [C, 2]

        # 为了对 flatten 后的一整段数据做归一化，
        # 把每个通道的 max/min 在时间维上 repeat 一下，得到长度 F 的向量
        max_repeat = np.tile(max_vals, samples_perdat)   # [F]
        min_repeat = np.tile(min_vals, samples_perdat)   # [F]

        # 在 __init__ 里一次性把所有数据归一化，后面 __getitem__ 就不用再算了
        denom = (max_repeat - min_repeat)
        denom[denom == 0] = 1.0  # 避免除 0
        self.features = (self.features - min_repeat) / denom

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features[idx]          # 已经归一化好的 numpy 数组 [F]
        y = self.labels[idx]

        x = torch.from_numpy(x)         # 转成 tensor 给模型用
        return x, y

    def getlen(self):
        return self.length
    