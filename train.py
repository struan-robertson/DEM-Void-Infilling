#!/usr/bin/env python3
import os
import random
import time

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt

from trainer import Trainer
from data.dataset import Dataset
from utils.tools import random_bbox, mask_image, apply_colormap, make_grid

### Config
config = {
    'dataset': "../Data",
    'checkpoint_save_path': "out",
    'resume': 0,
    'batch_size': 12,
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
    'cuda': False,
    'n_cpu': 16,
    'lr': 0.0001,
    'beta1': 0.5,
    'beta2': 0.9,
    'n_critic': 5,
    'epochs': 500000,
    'print_iter': 5,
    'viz_iter': 10,
    'viz_max_out': 12,
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

cuda = config["cuda"]
os.makedirs(os.path.join(config["checkpoint_save_path"], "images"), exist_ok=True)
os.makedirs(os.path.join(config["checkpoint_save_path"], "saved_models"), exist_ok=True)

## Set random seed to allow for training to be recreated

if config["seed"] is None:
    config["seed"] = random.randint(1, 10000)

seed = config["seed"]

print(f"Random seed used: {seed}")
random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)

# Dataloader for training
train_loader = DataLoader(
    Dataset(config["dataset"]),
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["n_cpu"],
)

trainer = Trainer(config)
#print(trainer.netG)
#print(trainer.localD)
#print(trainer.globalD)

if cuda:
    trainer = trainer.cuda()

# Get the resume iteration to restart training
start_iteration = trainer.resume(config["resume"]) if config["resume"] else 1

iterable_train_loader = iter(train_loader)

time_count = time.time()

for iteration in range(start_iteration, config["epochs"] + 1): # TODO acc this isnt epochs, should change it ot epoch using the data set size
    # Not sure why this try block is here, TODO remove if possible
    try:
        ground_truth = next(iterable_train_loader)
    except StopIteration:
        iterable_train_loader = iter(train_loader)
        ground_truth = next(iterable_train_loader)

    # Prepare inputs
    bboxes = random_bbox(config, batch_size=ground_truth.size(0))
    x, mask = mask_image(ground_truth, bboxes, config)
    if cuda:
        x = x.cuda()
        mask = mask.cuda()
        ground_truth = ground_truth.cuda()

    #### Forward pass
    # Only compute generator loss after 'n_critic' iterations, usually 5 as defined in the Wasserstein GAN paper
    compute_g_loss = iteration % config["n_critic"] == 0
    losses, inpainted_result = trainer(x, bboxes, mask, ground_truth, compute_g_loss)

    #### Backward Pass
    # Update D
    trainer.optimizer_d.zero_grad()
    losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
    losses['d'].backward()

    # Update G
    if compute_g_loss:
        trainer.optimizer_g.zero_grad()
        losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                      + losses['ae'] * config['ae_loss_alpha'] \
                      + losses['wgan_g'] * config['gan_loss_alpha']
        losses['g'].backward()
        trainer.optimizer_g.step()

    # Has to come afterwards
    trainer.optimizer_d.step()

    log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
    if iteration % config['print_iter'] == 0:
        time_count = time.time() - time_count
        speed = config['print_iter'] / time_count
        speed_msg = f'speed: {speed} batches/s'
        time_count = time.time()

        message = 'Iter: %d/%d, ' % (iteration, config['epochs'])

        for k in log_losses:
            v = losses.get(k, 0.)
            message += '%s: %.6f, ' % (k, v)

        message += speed_msg
        print(message)

    if iteration % config['snapshot_save_iter'] == 0:
        trainer.save_model(os.path.join(config["checkpoint_save_path"], "saved_models"), iteration)


    if iteration % (config['viz_iter']) == 0:

            viz_max_out = config['viz_max_out']

            if x.size(0) > viz_max_out:
                viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out]], dim=1)
            else:
                viz_images = torch.stack([x, inpainted_result], dim=1)

            if x.size(0) > viz_max_out:
                viz_images = torch.cat((x[:viz_max_out].data, inpainted_result[:viz_max_out].data, ground_truth[:viz_max_out].data), -2)
            else:
                viz_images = torch.cat((x.data, inpainted_result.data, ground_truth.data), -2)

            viz_images = apply_colormap(viz_images)

            grid = make_grid(viz_images)

            plt.imsave(os.path.join(config["checkpoint_save_path"], f'images/{iteration}.png'), grid)
