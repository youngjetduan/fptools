"""
This file (uni_image.py) is designed for:
    functions for image processing
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
import math
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
from scipy import ndimage as sndi


def affine_input_coordinate(y, scale=1.0, theta=0.0, trans_1=np.zeros(2), trans_2=np.zeros(2)):
    """get input coordinates (x) which are transformed to out_coord (y): y = s * (R(x + trans1) + trans2), x \in (M,2)"""
    theta = np.deg2rad(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    arr_shape = y.shape
    x = (y.reshape(-1, 2) / scale - trans_2[None]).dot(R) - trans_1[None]
    return x.reshape(*arr_shape)


def generate_grid(shape):
    grid = np.stack(np.meshgrid(*[np.arange(x) for x in shape], indexing="ij"))
    return grid


def calc_mass_center(img):
    grid = np.stack(np.meshgrid(*[np.arange(x) for x in img.shape], indexing="ij")).reshape(img.ndim, -1)
    center = (img.reshape(1, -1) * grid).sum(1) / img.sum().clip(1e-3, None)
    return center


def calc_seg_iou(input, target):
    smooth = 1e-3
    input_var = input.astype(np.float32)
    target_cal = 1.0 * (target > 0.5)
    input_cal = 1.0 * (input_var > 0.5)
    if target_cal.sum() == 0:
        return None

    intersect = input_cal * target_cal
    iou = (intersect.sum() + smooth) / (input_cal.sum() + target_cal.sum() - intersect.sum() + smooth)
    return iou


def intensity_centering(img, mask=None):
    """normalizing image intensity from (mean, std) to (0,1)

    Parameters:
        [None]
    Returns:
        [None]
    """
    if mask is not None:
        img = (img * 1.0 - img[mask].mean()) / img[mask > 0].std().clip(1e-6, None)
    else:
        img = (img * 1.0 - img.mean()) / img.std().clip(1e-6, None)

    return img


def intensity_normalization(img, mask=None, norm_type="min-max"):
    """map intensity to [0,1]

    Parameters:
        [None]
    Returns:
        [None]
    """
    if norm_type == "min-max":
        if mask is not None:
            img = (img * 1.0 - img[mask > 0].min()) / (img[mask > 0].max() - img[mask > 0].min()).clip(1e-6, None)
        else:
            img = (img * 1.0 - img.min()) / (img.max() - img.min()).clip(1e-6, None)
    elif norm_type == "mean-std":
        if mask is not None:
            img = (img * 1.0 - img[mask > 0].mean()) / img[mask > 0].std().clip(1e-6, None)
        else:
            img = (img * 1.0 - img.mean()) / img.std().clip(1e-6, None)
    img = img.clip(0, 1)
    return img


def shape_normalization(img, stride=8):
    """padding image to match the stride. Note that it's not always smaller than the previous.

    Parameters:
        [None]
    Returns:
        [None]
    """
    old_shape = np.array(img.shape[:2])
    tar_shape = old_shape // stride * stride
    img = np.pad(img, ((0, stride // 2), (0, stride // 2)))
    img = img[: tar_shape[0], : tar_shape[1]]
    return img


def croping_image(img, stride=8):
    """crop image to match stride. Note that it's always smaller than the previous.

    Parameters:
        [None]
    Returns:
        [None]
    """
    old_shape = np.array(img.shape[:2])
    new_shape = old_shape // stride * stride
    return img[: new_shape[0], : new_shape[1]]


def generate_heatmap(shape, kps, radius=3):
    kps_heatmap = np.zeros(shape)

    if len(kps):
        if kps.ndim == 1:
            kps = kps[None]

        if len(kps):
            if len(kps) > 2:
                kps = kps[:2]

            kps = np.rint(kps[:, :-1] / 8).astype(np.int)
            for ii in range(kps.shape[0]):
                kps_heatmap[kps[ii, 1], kps[ii, 0]] = 1
            kps_heatmap = np.exp(-sndi.distance_transform_edt(1 - kps_heatmap) ** 2 / 2 / radius ** 2)

    return kps_heatmap


def nextpow2(x):
    return int(math.ceil(math.log(x, 2)))


def LowpassFiltering(img, L):
    h, w = img.shape
    h2, w2 = L.shape

    img = cv2.copyMakeBorder(img, 0, h2 - h, 0, w2 - w, cv2.BORDER_CONSTANT, value=0)

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    img_fft = img_fft * L

    rec_img = np.fft.ifft2(np.fft.fftshift(img_fft))
    rec_img = np.real(rec_img)
    rec_img = rec_img[:h, :w]
    return rec_img


def compute_gradient_norm(input):
    input = input.astype(np.float32)
    h, w = input.shape

    Gx, Gy = np.gradient(input)
    out = np.sqrt(Gx * Gx + Gy * Gy) + 0.000001
    return out


def FastCartoonTexture(img, sigma=2.5, show=False):
    img = img.astype(np.float32)
    h, w = img.shape
    h2 = 2 ** nextpow2(h)
    w2 = 2 ** nextpow2(w)

    FFTsize = np.max([h2, w2])
    x, y = np.meshgrid(range(-int(FFTsize / 2), int(FFTsize / 2)), range(-int(FFTsize / 2), int(FFTsize / 2)))
    r = np.sqrt(x * x + y * y) + 0.0001
    r = r / FFTsize

    L = 1.0 / (1 + (2 * math.pi * r * sigma) ** 4)
    img_low = LowpassFiltering(img, L)

    gradim1 = compute_gradient_norm(img)
    gradim1 = LowpassFiltering(gradim1, L)

    gradim2 = compute_gradient_norm(img_low)
    gradim2 = LowpassFiltering(gradim2, L)

    diff = gradim1 - gradim2
    ar1 = np.abs(gradim1)
    diff[ar1 > 1] = diff[ar1 > 1] / ar1[ar1 > 1]
    diff[ar1 <= 1] = 0

    cmin = 0.3
    cmax = 0.7

    weight = (diff - cmin) / (cmax - cmin)
    weight[diff < cmin] = 0
    weight[diff > cmax] = 1

    u = weight * img_low + (1 - weight) * img
    temp = img - u
    lim = 20

    temp1 = (temp + lim) * 255 / (2 * lim)
    temp1[temp1 < 0] = 0
    temp1[temp1 > 255] = 255
    v = temp1
    if show:
        plt.imshow(v, cmap="gray")
        plt.show()
    return v


def histogram_match(img, ref):
    matched = exposure.match_histograms(img, ref)
    return matched


def histogram_match_distrib(img, tar_vals, tar_cdfs):
    _, img_cls, img_cnts = np.unique(img.ravel(), return_inverse=True, return_counts=True)
    img_cdfs = np.cumsum(img_cnts) / img.size
    interp_a_values = np.interp(img_cdfs, tar_cdfs, tar_vals)
    return interp_a_values[img_cls].reshape(img.shape)


def add_noise(img, scale=1.0, noise_type="gaussian", sigma=0.1):
    noise_shape = np.rint(np.array(img.shape[:2]) / scale).astype(int)
    if noise_type == "gaussian":
        noise = np.random.normal(0, sigma * img.max(), noise_shape)
        if scale != 1:
            noise = sndi.zoom(noise, (img.shape[0] * 1.0 / noise.shape[0], img.shape[1] * 1.0 / noise.shape[1]), order=1)

    return np.clip(img + noise, 0, 255)


if __name__ == "__main__":
    prefix = ""
