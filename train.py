#!/usr/bin/env python3
import os
import random
import time

import torch
from torch.utils.data import DataLoader

import toml
import matplotlib.pyplot as plt

from trainer import Trainer
from dataset import Dataset
from tools import random_bbox, mask_image, apply_colormap, make_grid

with open("config.toml", "r") as file:
    config = toml.load(file)

##### Initialise

cuda = config["cuda"]
os.makedirs(os.path.join(config["checkpoint_save_path"], "images"), exist_ok=True)
os.makedirs(os.path.join(config["checkpoint_save_path"], "saved_models"), exist_ok=True)

## Set random seed to allow for training to be recreated

if "seed" in config:
    seed = config["seed"]
else:
    seed = random.randint(1, 10000)

print(f"Random seed used: {seed}")
random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)

# Dataloader for training
train_loader = DataLoader(
    Dataset(config),
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["n_cpu"],
)

trainer = Trainer(config)
# print(trainer.netG)
# print(trainer.localD)
# print(trainer.globalD)

if cuda:
    trainer = trainer.cuda()

# Get the resume iteration to restart training
start_epoch = (trainer.resume(config["checkpoint_save_path"], config["resume"]) if "resume" in config else 1)

iterable_train_loader = iter(train_loader)

time_count = time.time()

for epoch in range(config["epochs"]):
    # For correctly saving snapshots
    epoch += start_epoch

    # TODO maybe use iteration instead of epoch for saving snapshots and visualisations, as with v large dataset takes a long time for each epoch

    for iteration, ground_truth in enumerate(train_loader):
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
        losses, inpainted_result = trainer(
            x, bboxes, mask, ground_truth, compute_g_loss
        )

        #### Backward Pass
        # Update D
        trainer.optimizer_d.zero_grad()
        losses["d"] = losses["wgan_d"] + losses["wgan_gp"] * config["wgan_gp_lambda"]
        losses["d"].backward()

        # Update G
        if compute_g_loss:
            trainer.optimizer_g.zero_grad()
            losses["g"] = (
                losses["l1"] * config["l1_loss_alpha"]
                + losses["ae"] * config["ae_loss_alpha"]
                + losses["wgan_g"] * config["gan_loss_alpha"]
            )
            losses["g"].backward()
            trainer.optimizer_g.step()

        # Has to come afterwards
        trainer.optimizer_d.step()

        # TODO Epoch size doesnt change and can just be printed at the start
        # TODO Print start and end time

        log_losses = ["l1", "ae", "wgan_g", "wgan_d", "wgan_gp", "g", "d"]
        if epoch % config["print_iter"] == 0 and iteration == 0:
            time_count = time.time() - time_count
            speed = config["print_iter"] / time_count
            speed_msg = f"speed: {speed * 60} epochs/min"
            time_count = time.time()

            # message = 'Iter: %d/%d, ' % (iteration, config['epochs'])
            message = f"Epoch: {epoch}, Epoch Size: {train_loader.__len__()}, "

            for k in log_losses:
                v = losses.get(k, 0.0)
                message += "%s: %.6f, " % (k, v)

            message += speed_msg
            print(message)

        if epoch % config["snapshot_save_iter"] == 0 and iteration == 0:
            trainer.save_model(
                os.path.join(config["checkpoint_save_path"], "saved_models"), epoch
            )

        if epoch % (config["viz_iter"]) == 0 and iteration == 0:
            viz_max_out = config["viz_max_out"]

            # if x.size(0) > viz_max_out:
            #     viz_images = torch.stack([x[:viz_max_out,0], inpainted_result[:viz_max_out,0]], dim=1)
            # else:
            #     viz_images = torch.stack([x[:,0], inpainted_result[:,0]], dim=1)

            if x.size(0) > viz_max_out:
                viz_dem = torch.cat((x[:viz_max_out, 0].data, inpainted_result[:viz_max_out, 0].data, ground_truth[:viz_max_out, 0].data),-2)
                viz_slope = torch.cat((x[:viz_max_out, 1].data, inpainted_result[:viz_max_out, 1].data, ground_truth[:viz_max_out, 1].data),-2)
            else:
                viz_dem = torch.cat((x[:, 0].data, inpainted_result[:, 0].data, ground_truth[:, 0].data), -2)
                viz_slope = torch.cat((x[:, 1].data, inpainted_result[:, 1].data, ground_truth[:, 1].data), -2)

            viz_dem = apply_colormap(viz_dem, "terrain")
            viz_slope = apply_colormap(viz_slope, "viridis")

            dem_grid = make_grid(viz_dem)
            slope_grid = make_grid(viz_slope)

            plt.imsave(os.path.join(config["checkpoint_save_path"], f"images/{epoch}_dem.png"), dem_grid)
            plt.imsave(os.path.join(config["checkpoint_save_path"], f"images/{epoch}_slope.png"), slope_grid)
