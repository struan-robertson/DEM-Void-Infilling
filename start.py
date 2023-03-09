#!/usr/bin/env python3

dataset = 0
resume = 0 #TODO
batch_size = 48
image_shape = [256, 256, 1]
mask_shape = [128, 128]
mask_batch_same = True
max_delta_shape = [32, 32]
margin = [0, 0]
discounted_mask = True
spatial_discounting_gamma = 0.9
random_crop = True
mask_type = "hole" # hole | mosaic
mosaic_unit_size = 12

# Training parameters
expname = "benchmark"
cuda = True
gpu_ids = 0
num_workers = 4
lr = 0.0001
beta1 = 0.5
beta2 = 0.9
n_critic = 5
niter = 500000
print_iter = 100
viz_iter = 1000
viz_max_out = 16
snapshot_save_iter = 5000

# Loss weight
coarse_l1_alpha = 1.2
l1_loss_alpha = 1.2
ae_loss_alpha = 1.2
global_wgan_loss_alpha = 1.0
gan_loss_alpha = 0.001
wgan_gp_lambda = 10

# Network Parameters
netG_input_dim = 1
netG_ngf = 32

netD_input_dim = 1
netD_ndf = 64
