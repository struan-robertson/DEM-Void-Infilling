#!/usr/bin/env python3

import os

import torch.utils.data as data
import torch

from osgeo import gdal
import numpy as np

import torchvision.transforms as transforms

# Image transforms
def normalise(tensor):
    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

    return tensor

transforms_ = [
    transforms.ToTensor(),
    #transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
    transforms.Lambda(normalise),
    transforms.Normalize((0.5), (0.5)),
]


# Dataloader class

class Dataset(data.Dataset):
    def __init__(self, root, img_size=256, mask_size=128, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.tiles = self.tile(root, (img_size, img_size))

    def tile(self, dataset, kernel_size):

        dems = []

        for file in os.listdir(dataset):
            path = os.path.join(dataset, file)

            pds = gdal.Open(path)

            image = np.array(pds.ReadAsArray())

            img_height, img_width = image.shape
            tile_height, tile_width = kernel_size

            # If cant divide perfectly
            if (img_height % tile_height != 0 or img_width % tile_width != 0):
                new_height = img_height - (img_height % tile_height)
                new_width = img_width - (img_width % tile_width)

                image = image[:new_height, :new_width]

            tiles_high = img_height // tile_height
            tiles_wide = img_width // tile_width

            tiled_array = image.reshape(tiles_high,
                                        tile_height,
                                        tiles_wide,
                                        tile_width )

            tiled_array = tiled_array.swapaxes(1, 2)

            tiled_array = tiled_array.reshape(tiles_high * tiles_wide, tile_height, tile_width)

            dems.append(tiled_array)

            # GC should get these, but just to be safe
            pds = None
            tiled_array = None

        full = np.concatenate((*dems,))

        dems = None

        return full


    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):

        img = self.tiles[index % self.tiles.shape[0]]
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return self.tiles.shape[0]
