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
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F

from . import fp_orientation
from .uni_image import intensity_normalization


def steerable_enhance(img, ori, seg, sigma=2):
    if np.any(np.array(ori.shape) != np.array(img.shape)):
        ori = fp_orientation.zoom_orientation(ori, (1.0 * img.shape[0] / ori.shape[0], 1.0 * img.shape[1] / ori.shape[1]))
    if np.any(np.array(seg.shape) != np.array(img.shape)):
        seg = ndi.zoom(seg, (1.0 * img.shape[0] / ori.shape[0], 1.0 * img.shape[1] / ori.shape[1]), order=0)
    Ixx = ndi.gaussian_filter(img.astype(np.float32), sigma, order=(0, 2))
    Ixy = ndi.gaussian_filter(img.astype(np.float32), sigma, order=(1, 1))
    Iyy = ndi.gaussian_filter(img.astype(np.float32), sigma, order=(2, 0))
    sin_ori = np.sin(ori * np.pi / 180)
    cos_ori = np.cos(ori * np.pi / 180)
    J = cos_ori ** 2 * Ixx + sin_ori ** 2 * Iyy - 2 * sin_ori * cos_ori * Ixy
    return intensity_normalization(J * seg) * 255


def vectorize_orientation(ori, ang_stride=2, ang_range=180, sigma=5):
    new_shape = (1, -1,) + (
        1,
    ) * (ori.ndim - 2)
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


def gabor_enhance(
    img,
    ori,
    seg,
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

    if np.any(np.array(ori.shape) != np.array(img.shape)):
        ori = fp_orientation.zoom_orientation(ori, (1.0 * img.shape[0] / ori.shape[0], 1.0 * img.shape[1] / ori.shape[1]))
    if np.any(np.array(seg.shape) != np.array(img.shape)):
        seg = ndi.zoom(seg, (1.0 * img.shape[0] / ori.shape[0], 1.0 * img.shape[1] / ori.shape[1]), order=0)

    ori = np.where(seg > 0, ori, 91)

    img = torch.tensor(img.astype(np.float32))
    ori = torch.tensor(ori.astype(np.float32))
    gb_cos = torch.tensor(gb_cos.astype(np.float32))
    if use_cuda:
        img = img.cuda()
        ori = ori.cuda()
        gb_cos = gb_cos.cuda()

    img_real = F.conv2d(img[None, None], gb_cos[:, None].type_as(img))
    img_real = F.pad(img_real, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2))
    ori_peak = vectorize_orientation(ori[None, None], ang_stride, ang_range, vo_sigma)
    # ori_peak = F.upsample_bilinear(ori_peak, scale_factor=factor)
    img_real = (img_real * ori_peak).sum(dim=1)

    try:
        img_real = img_real.squeeze().cpu().numpy()
    except:
        img_real = img_real.squeeze().numpy()

    img_real = np.where(seg > 0, img_real, 0)
    img_real = intensity_normalization(img_real)

    return img_real


def compute_gradient_norm(img, eps=1e-8):
    Gx, Gy = np.gradient(img.astype(np.float32))
    out = np.sqrt(Gx ** 2 + Gy ** 2) + eps
    return out


def lowpass_filtering(img, L):
    img_fft = np.fft.fftshift(np.fft.fft2(img), axes=(-2, -1)) * L
    img_rec = np.fft.ifft2(np.fft.fftshift(img_fft, axes=(-2, -1)))
    img_rec = np.real(img_rec)
    return img_rec


def fast_cartoontexture(img, sigma=2.5, cmin=0.3, cmax=0.7, lim=20, eps=1e-8):
    H, W = img.shape[:2]
    grid_y, grid_x = np.meshgrid(np.linspace(-0.5, 0.5, H), np.linspace(-0.5, 0.5, W), indexing="ij")
    grid_radius = np.sqrt(grid_x ** 2 + grid_y ** 2) + eps

    L = 1.0 / (1 + (2 * np.pi * grid_radius * sigma) ** 4)

    grad_img1 = compute_gradient_norm(img, eps)
    grad_img1 = lowpass_filtering(grad_img1, L)

    img_low = lowpass_filtering(img, L)
    grad_img2 = compute_gradient_norm(img_low, eps)
    grad_img2 = lowpass_filtering(grad_img2, L)

    diff = grad_img1 - grad_img2
    flag = np.abs(grad_img1)
    diff = np.where(flag > 1, diff / flag.clip(eps, None), np.zeros_like(diff))
    weight = np.clip((diff - cmin) / (cmax - cmin), 0, 1)

    cartoon = weight * img_low + (1 - weight) * img
    texture = (img - cartoon + lim) * 255 / (2 * lim)
    texture = np.clip(texture, 0, 255)
    return cartoon, texture


if __name__ == "__main__":
    prefix = ""
