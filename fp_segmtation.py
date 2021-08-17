"""
This file (fp_segmtation.py) is designed for:
    functions for fingerprint segmentation
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
from skimage import morphology
from scipy.ndimage import gaussian_filter, uniform_filter, zoom
from skimage.segmentation import watershed
from skimage.filters import sobel
import torch
import torch.nn.functional as F


def ridge_coherence(img, win_size=8, stride=8):
    Gx, Gy = np.gradient(img.astype(np.float32))
    coh = np.sqrt((Gx ** 2 - Gy ** 2) ** 2 + 4 * Gx ** 2 * Gy ** 2)
    coh = (
        F.avg_pool2d(torch.tensor(coh)[None, None].float(), 2 * win_size - 1, stride=stride, padding=win_size - 1)
        .squeeze()
        .numpy()
    )
    return coh


def ridge_intensity(img, win_size=8, stride=8):
    coh = (
        F.avg_pool2d(torch.tensor(img)[None, None].float(), 2 * win_size - 1, stride=stride, padding=win_size - 1)
        .squeeze()
        .numpy()
    )
    return coh


def segmentation_clustering(samples):
    """ segmentation using cluster based method
    
    Parameters:
        samples: (N, M)
    Returns:
        [None]
    """
    pred_1 = samples[:, 0] > 20
    pred_2 = samples[:, 1] < samples[pred_1, 1].mean()
    return pred_1 | pred_2


def segmentation_postprocessing(seg, kernel_size=5):
    selem = np.ones((kernel_size, kernel_size))
    seg = morphology.remove_small_holes(seg, area_threshold=100)
    seg = morphology.remove_small_objects(seg, min_size=100)
    seg = morphology.binary_closing(seg.astype(np.bool), selem=selem)
    return seg


def segmentation_coherence(img, win_size=8, stride=8):
    # average pooling
    Gx, Gy = np.gradient(img.astype(np.float32))
    Gxx = uniform_filter(Gx ** 2, win_size / 3)
    Gyy = uniform_filter(Gy ** 2, win_size / 3)
    Gxy = uniform_filter(Gx * Gy, win_size / 3)
    coh = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2) / (Gxx + Gyy).clip(1e-6, None)
    print(coh.min(), coh.max())
    if stride != 1:
        coh = zoom(coh, 1.0 / stride, order=1)
    seg = coh > 0.5

    selem = np.ones((5, 5))
    seg = morphology.binary_closing(seg.astype(np.bool), selem=selem)
    seg = morphology.remove_small_holes(seg, area_threshold=1000 // stride)
    seg = morphology.remove_small_objects(seg, min_size=1000 // stride)
    return seg


def segmentation_watershed(img, markers, stride=8):
    img = zoom(img, 1.0 / stride, order=1)
    # markers = zoom(markers, 1.0 / stride, order=0)
    elevation_map = sobel(img)
    seg = watershed(elevation_map, markers) - 1
    # seg = morphology.remove_small_holes(seg, area_threshold=100)
    # seg = morphology.remove_small_objects(seg, min_size=100)
    return seg


if __name__ == "__main__":
    prefix = ""
