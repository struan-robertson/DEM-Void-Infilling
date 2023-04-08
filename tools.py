import os
import torch
import numpy as np
import matplotlib.cm

import torch.nn.functional as F


# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding="same"):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ["same", "valid"]
    batch_size, channel, height, width = images.size()

    if padding == "same":
        images = same_padding(images, ksizes, strides, rates)
    elif padding == "valid":
        pass
    else:
        raise NotImplementedError(
            'Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(
                padding
            )
        )

    unfold = torch.nn.Unfold(
        kernel_size=ksizes, dilation=rates, padding=0, stride=strides
    )
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def random_bbox(config, batch_size):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config["image_shape"]
    h, w = config["mask_shape"]
    margin_height, margin_width = config["margin"]
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if config["mask_batch_same"]:
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
        bbox_list.append((t, l, h, w))
        bbox_list = bbox_list * batch_size
    else:
        for i in range(batch_size):
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))

    return torch.tensor(bbox_list, dtype=torch.int64)


def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[
            i,
            :,
            bbox[0] + delta_h : bbox[0] + bbox[2] - delta_h,
            bbox[1] + delta_w : bbox[1] + bbox[3] - delta_w,
        ] = 1.0
    return mask


def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t : t + h, l : l + w])
    return torch.stack(patches, dim=0)


def bbox2split(bboxes, height, width):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)

    min_delta = 20
    max_delta = 40

    for i in range(batch_size):
        bbox = bboxes[i]
        width = np.random.randint(min_delta, max_delta)

        mask[i, :, 0:256, bbox[1] + width : bbox[1] + bbox[3] - width] = 1.0

    return mask


def mask_image(x, bboxes, config, train=True):
    height, width, _ = config["image_shape"]
    max_delta_h, max_delta_w = config["max_delta_shape"]

    # Split between boxes and rectangles cutting the image in half
    if train:
        if np.random.randint(2) == 0:
            mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)
        else:
            mask = bbox2split(bboxes, height, width)
    else:
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        bbox = bboxes[0]
        mask[
            0, :, bbox[0] : (bbox[0] + bbox[2] + 1), bbox[1] : (bbox[1] + bbox[3] + 1)
        ] = 1

    if x.is_cuda:
        mask = mask.cuda()

    result = x * (1.0 - mask)

    return result, mask


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config["spatial_discounting_gamma"]
    height, width = config["mask_shape"]
    shape = [1, 1, height, width]
    if config["discounted_mask"]:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = max(
                    gamma ** min(i, height - i), gamma ** min(j, width - j)
                )
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 0)
    else:
        mask_values = np.ones(shape)
    spatial_discounting_mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    if config["cuda"]:
        spatial_discounting_mask_tensor = spatial_discounting_mask_tensor.cuda()
    return spatial_discounting_mask_tensor


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


# Get model list for resume
def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if "{:0>8d}".format(iteration) in model_name:
                return model_name
        raise ValueError("Not found models with this iteration")
    return last_model_name


def apply_colormap(tensor, colormap):
    cm = matplotlib.cm.get_cmap(colormap)

    tensor = tensor.cpu().detach().numpy()

    img = np.empty([tensor.shape[0], (256 * 3), 256, 4])

    for idx, sample in enumerate(tensor):
        norm = (sample + 1) / 2
        norm = norm * 255
        norm = norm.astype(np.int16)
        mapped = cm(norm)

        mapped = np.squeeze(mapped)
        img[idx] = mapped

    return img


def make_grid(d):
    r1 = np.concatenate((d[0], d[1], d[2], d[3], d[4], d[5]), axis=1)
    r2 = np.concatenate((d[6], d[7], d[8], d[9], d[10], d[11]), axis=1)

    return np.concatenate((r1, r2), axis=0)
