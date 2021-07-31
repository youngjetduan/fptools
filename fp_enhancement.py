"""
This file (fp_enhancement.py) is designed for:
    functions for fingerprint enhancement
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F


from .uni_image import intensity_normalization


def vectorize_orientation(ori, ang_stride=2, ang_range=180, sigma=5):
    new_shape = (1, -1,) + (1,) * (ori.ndim - 2)
    coord = torch.arange(ang_stride // 2, ang_range, ang_stride).view(*new_shape).type_as(ori) - ang_range // 2
    delta = (ori - coord).abs()
    delta = torch.min(delta, ang_range - delta)  # [0,180)
    vec = torch.exp(-(delta ** 2) / (2 * sigma ** 2))
    vec = vec / vec.sum(dim=1, keepdim=True)
    return vec


def gabor_bank(kernel_size=25, ang_stride=2, ang_range=180, sigma=4, Lambda=9, psi=0, gamma=1):
    grid_theta, grid_x, grid_y = np.meshgrid(
        np.arange(0, ang_range, ang_stride), np.arange(kernel_size), np.arange(kernel_size), indexing="ij"
    )
    grid_theta = -(grid_theta - ang_range // 2) * np.pi / 180.0
    grid_x = grid_x - kernel_size // 2
    grid_y = grid_y - kernel_size // 2

    cos_theta = np.cos(grid_theta)
    sin_theta = np.sin(grid_theta)
    x_theta = grid_y * sin_theta + grid_x * cos_theta
    y_theta = grid_y * cos_theta - grid_x * sin_theta
    # gabor filters
    exp_fn = np.exp(-0.5 * (x_theta ** 2 + gamma ** 2 * y_theta ** 2) / sigma ** 2)
    gb_cos = exp_fn * np.cos(2 * np.pi * x_theta / Lambda + psi)
    gb_sin = exp_fn * np.sin(2 * np.pi * x_theta / Lambda + psi)

    return gb_cos, gb_sin


def enhance(
    img,
    ori,
    seg,
    factor=8.0,
    kernel_size=25,
    ang_stride=2,
    ang_range=180,
    vo_sigma=5,
    gb_sigma=4,
    gb_lambda=9,
    gb_psi=0,
    gb_gamma=1,
    use_cuda=False,
):
    gb_cos, _ = gabor_bank(kernel_size, ang_stride, ang_range, gb_sigma, gb_lambda, gb_psi, gb_gamma)

    ori = np.where(seg > 0, ori, 91)

    img = torch.tensor(img)
    ori = torch.tensor(ori)
    gb_cos = torch.tensor(gb_cos)
    if use_cuda:
        img = img.cuda()
        ori = ori.cuda()
        gb_cos = gb_cos.cuda()

    img_real = F.conv2d(img[None, None], gb_cos[:, None].type_as(img))
    img_real = F.pad(img_real, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2))
    ori_peak = vectorize_orientation(ori[None, None], ang_stride, ang_range, vo_sigma)
    ori_peak = F.upsample_bilinear(ori_peak, scale_factor=factor)
    img_real = (img_real * ori_peak).sum(dim=1)

    try:
        img_real = img_real.squeeze().cpu().numpy()
    except:
        img_real = img_real.squeeze().numpy()

    img_real = np.where(zoom(seg, factor, order=0) > 0, img_real, 0)
    img_real = intensity_normalization(img_real)

    return img_real


if __name__ == "__main__":
    prefix = ""
