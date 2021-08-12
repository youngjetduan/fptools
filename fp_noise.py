"""
This file (fp_noise.py) is designed for:
    add fingerprint noise
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
from skimage import morphology
from scipy.ndimage import distance_transform_edt, zoom

sys.path.append(osp.dirname(osp.abspath(__file__)))
from fp_segmtation import segmentation_coherence
from uni_image import intensity_normalization


def perlin(shape, frequency=5, seed=0):
    width = np.linspace(0, frequency * 3, shape[0], endpoint=False)
    height = np.linspace(0, frequency * 3, shape[1], endpoint=False)
    x, y = np.meshgrid(width, height)
    # permutation table
    # np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # integer part
    x_int = x.astype(int) % 256
    y_int = y.astype(int) % 256
    # fraction part
    x_frac = x - x_int
    y_frac = y - y_int
    # ease transitions with sigmoid-type function
    fade_x = fade(x_frac)
    fade_y = fade(y_frac)
    # noise components
    n00 = gradient(p[p[x_int] + y_int], x_frac, y_frac)
    n01 = gradient(p[p[x_int] + y_int + 1], x_frac, y_frac - 1)
    n11 = gradient(p[p[x_int + 1] + y_int + 1], x_frac - 1, y_frac - 1)
    n10 = gradient(p[p[x_int + 1] + y_int], x_frac - 1, y_frac)
    # combine noises
    x1 = lerp(n00, n10, fade_x)
    x2 = lerp(n01, n11, fade_x)
    return lerp(x1, x2, fade_y)


def lerp(a, b, x):
    return a + x * (b - a)


def fade(t):
    t_squared = t ** 2  # Time saver
    return (6 * t_squared - 15 * t + 10) * t * t_squared


def gradient(h, x, y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def sensor_noise(img, mask=None, do_wet=False, stride=8, pL=0.25, tB=20, k=3):
    # border noise
    if mask is None:
        mask = segmentation_coherence(img, win_size=16, stride=stride)
    mask = morphology.binary_dilation(mask, selem=np.ones([3, 3]))
    dist = distance_transform_edt(mask) * stride
    dist = zoom(dist, [img.shape[0] * 1.0 / mask.shape[0], img.shape[1] * 1.0 / mask.shape[1]], order=1)
    mask = dist > 0
    dist = np.where(dist > tB, 0, (tB - dist) / tB)
    p_border = pL * (1 + dist ** 3)
    # perlin noise
    p_perlin = 0
    for ii in [0, 1, 2]:
        p_perlin += 2.0 ** (-ii) * perlin(img.shape[::-1], frequency=2 ** ii, seed=809)
    p_perlin = intensity_normalization(p_perlin)

    # add blob noise
    img_n = img
    cur_p = np.random.random(p_perlin.shape)
    if do_wet:
        p_total = p_border * (1 + p_perlin ** k)
        p_total = intensity_normalization(p_total)
        blob_noise = (cur_p <= p_total) * mask
        blob_noise = morphology.binary_dilation(blob_noise)
        blob_noise = morphology.binary_closing(blob_noise)
        img_n = np.where(blob_noise | (1 - mask), 255, img_n)

        p_total = pL * (1 + (1 - p_perlin) ** k)
        p_total = intensity_normalization(p_total)
        blob_noise = (cur_p <= p_total) * mask
        blob_noise = morphology.binary_closing(blob_noise)
        img_n = np.where(blob_noise & mask, 0, img_n)
    else:
        p_total = p_border * (1 + p_perlin ** k)
        p_total = intensity_normalization(p_total * mask)
        # cur_p = np.random.random(p_total.shape)
        blob_noise = (cur_p <= 1.0 * p_total) * mask
        # blob_noise = morphology.binary_dilation(blob_noise)
        blob_noise = morphology.binary_closing(blob_noise)
        img_n = np.where(blob_noise | (1 - mask), 255, img_n)
        img_n = np.rint(np.minimum(255, img_n + 255 * p_perlin)).astype(np.uint8)
    img_n = intensity_normalization(img_n)

    return img_n, blob_noise  # intensity_normalization(blob_noise * 1)


def dryness(img, selem=np.ones([2, 2])):
    img_n = morphology.dilation(img, selem=selem)
    return img_n


def heavypress(img, selem=np.ones([3, 3])):
    img_n = morphology.erosion(img, selem=selem)
    return img_n


if __name__ == "__main__":
    prefix = ""
