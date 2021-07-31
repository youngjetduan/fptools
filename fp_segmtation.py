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
    coh = np.sqrt((Gx ** 2 - Gy ** 2) ** 2 + 4 * Gx ** 2 * Gy ** 2)
    # coh = gaussian_filter(coh, sigma=win_size / 2)
    coh = zoom(coh, 1.0 / stride, order=1)
    # coh = gaussian_filter(coh, sigma=win_size / 4)
    # coh = gaussian_filter(coh, sigma=win_size / 4)
    # coh_tensor = torch.tensor(coh[None, None,]).float()
    # coh = F.avg_pool2d(coh_tensor, 2 * win_size - 1, stride=1, padding=win_size - 1)
    # coh = F.avg_pool2d(coh, 2 * win_size - 1, stride=stride, padding=win_size - 1).squeeze().numpy()
    seg = coh > 50

    # h, w = img.shape[:2]
    # h //= stride
    # w //= stride
    # blks = np.stack(np.split(img, h, axis=0), axis=-1)
    # blks = np.stack(np.split(blks, w, axis=1), axis=-1)
    # blks = blks.reshape(stride ** 2, h, w)
    # seg = np.std(blks, axis=0) > 0.1 * 255

    selem = np.ones((5, 5))
    seg = morphology.remove_small_holes(seg, area_threshold=100)
    seg = morphology.remove_small_objects(seg, min_size=100)
    seg = morphology.binary_closing(seg.astype(np.bool), selem=selem)
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
