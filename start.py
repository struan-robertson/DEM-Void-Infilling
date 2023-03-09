#!/usr/bin/env python3
import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from trainer import Trainer
from data.dataset_n import Dataset # TODO rename when network finished
#from utils.tools import get_config, random_bbox, mask_image
#from utils.logger import get_logger

### Config
config = {
    'dataset': "../Data",
    'resume': 0,
    'batch_size': 48,
    'image_shape': [256, 256, 1],
    'mask_shape': [128, 128],
    'mask_batch_same': True,
    'max_delta_shape': [32, 32],
    'margin': [0, 0],
    'discounted_mask': True,
    'spatial_discounting_gamma': 0.9,
    'random_crop': True,
    'mask_type': "hole", # hole | mosaic
    'mosaic_unit_size': 12,

    # Training parameters
    'expname': "benchmark",
    #cuda = True
    #gpu_ids = 0
    'n_cpu': 16, # Might be the same as num_workers #TODO come back after network implemented
    'num_workers': 4,
    'lr': 0.0001,
    'beta1': 0.5,
    'beta2': 0.9,
    'n_critic': 5,
    'niter': 500000,
    'print_iter': 100,
    'viz_iter': 1000,
    'viz_max_out': 16,
    'snapshot_save_iter': 5000,
    'seed': None,

    # Loss weight
    'coarse_l1_alpha': 1.2,
    'l1_loss_alpha': 1.2,
    'ae_loss_alpha': 1.2,
    'global_wgan_loss_alpha': 1.0,
    'gan_loss_alpha': 0.001,
    'wgan_gp_lambda': 10,

    # Network Parameters
    'input_dim': 1,
    'ngf': 32,
    'ndf': 64
}

##### Initialise

cuda = True if torch.cuda.is_available() else False
os.makedirs("out/images", exist_ok=True)
os.makedirs("out/saved_models", exist_ok=True)

## Set random seed to allow for training to be recreated

if config["seed"] is None:
    config["seed"] = random.randint(1, 10000)

seed = config["seed"]

print(f"Random seed used: {seed}")
random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)

##### Dataloader
## This is a very expensive way of implementing this, as all data is held in memory twice. # TODO implement a fix if time

# Dataloader for training
dataloader = DataLoader(
    Dataset(config["dataset"]),
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["n_cpu"],
)
# Dataloader for saving grid of samples every epoch
test_dataloader = DataLoader(
    Dataset(config["dataset"], mode="val"),
    batch_size=12,
    shuffle=True,
    num_workers=1,
)

trainer = Trainer(config)
print(trainer.netG)
print(trainer.localD)
print(trainer.globalD)
