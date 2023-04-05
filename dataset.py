#!/usr/bin/env python3

import os
import time
import concurrent.futures

import torch.utils.data as data
import torch

from osgeo import gdal
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Dataloader class

# outside the class to make life simpler in terms of function signitures
def normalize(arr):
    normalized = (arr - arr.min()) / (arr.max() - arr.min())
    normalized = 2*normalized - 1
    return normalized

class Dataset(data.Dataset):
    def __init__(self, config):

        image_shape = config["image_shape"]
        self.kernel_size = (image_shape[0], image_shape[1])
        self.root = config["dataset"]
        self.n_cpu = config["n_cpu"]

        self.tiles = self.tile()

    def tile_thread(self, dataset, num):
        vectorized_normalise = np.vectorize(normalize, signature='(n,m)->(n,m)')

        pds = gdal.Open(dataset)
        geot = pds.GetGeoTransform()

        image = np.array(pds.ReadAsArray())

        img_height, img_width = image.shape
        tile_height, tile_width = self.kernel_size

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

        tiled_array = vectorized_normalise(tiled_array)

        # Slope
        cellsize = geot[1]
        px, py = np.gradient(tiled_array, cellsize, axis=(1,2))
        slope = np.arctan(np.sqrt(px ** 2 + py ** 2))
        slope = vectorized_normalise(slope)

        # RDLS
        windowed = sliding_window_view(tiled_array, (3,3), axis=(1,2)) # type: ignore
        rdls = np.ptp(windowed, axis=(3,4))
        rdls = np.pad(rdls, ((0,0), (1,1), (1,1)), mode='constant', constant_values=0)
        rdls = vectorized_normalise(rdls)

        all = np.stack((tiled_array, slope, rdls), axis=3)

        # H,W,C to C,H,W
        all = np.transpose(all, (0,3,1,2))

        file = os.path.basename(dataset)

        # V much all over the place since different DEMS take v different amounts of time to process, however gives a rough idea of where the proceesing is at
        print(f'loaded DEM {file}, number {num}')

        return all

    def tile(self):

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_cpu) as executor:

            futures = []

            i = 1

            # Slightly strange way to do it but used for counting files first
            files = os.listdir(self.root)
            n_files = len(files)

            for file in files:
                path = os.path.join(self.root, file)
                futures.append(executor.submit(self.tile_thread, path, f'{i}/{n_files}'))
                i += 1

            dems = [future.result() for future in concurrent.futures.as_completed(futures)]

        full = np.concatenate((*dems,))

        dems = None

        full = torch.from_numpy(full)

        end_time = int(time.time() - start_time)

        print(f'Loaded all DEMs in {end_time} seconds')

        return full

    def __getitem__(self, index):

        img = self.tiles[index % self.tiles.shape[0]]

        return img

    def __len__(self):
        return self.tiles.shape[0]
