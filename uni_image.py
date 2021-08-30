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
from scipy.ndimage import distance_transform_edt


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


def intensity_normalization(img):
    """ map intensity to [0,1]
    
    Parameters:
        [None]
    Returns:
        [None]
    """
    img = (img * 1.0 - img.min()) / (img.max() - img.min()).clip(1e-6, None)
    return img


def shape_normalization(img, stride=8):
    """ padding image to match the stride. Note that it's not always smaller than the previous.
    
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
    """ crop image to match stride. Note that it's always smaller than the previous.
    
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
            kps_heatmap = np.exp(-distance_transform_edt(1 - kps_heatmap) ** 2 / 2 / radius ** 2)

    return kps_heatmap


if __name__ == "__main__":
    prefix = ""
