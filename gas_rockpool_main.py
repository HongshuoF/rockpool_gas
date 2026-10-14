import torch
import torchvision

import numpy as np

import matplotlib.pyplot as plt

# - Import the SNN modules from rockpool
from rockpool.nn.modules import LIF
from tqdm import trange, tqdm
import dataset
from torch.utils.data import Subset
import config

cfg = config.Config()
samples_perdat = cfg.time_all  # Number of sampling points per sample.
category_num = len(cfg.labellist)  # Number of classes.


# - Proportion of the dataset to use for training and validation
train_proportion = 0.8
val_proportion = 0.1
test_proportion = 0.1
full_dataset = dataset.MyDataset(cfg.dataset_path, label_list=cfg.labellist, sensor_num=cfg.sensor_num, sensor_start=cfg.sensor_start, samples_perdat=samples_perdat)  # Convert to a Dataset with separate features and labels.

train_idx, val_idx, test_idx = dataset.data_split(len(full_dataset), train_proportion, val_proportion)
train_set = Subset(full_dataset, train_idx)
val_set = Subset(full_dataset, val_idx)
test_set = Subset(full_dataset, test_idx)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)


if __name__ == "__main__":
    from train import fit

    snn, history, test_metrics = fit(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        test_loader=test_loader,
    )

