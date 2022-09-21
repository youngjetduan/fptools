"""
This file (fp_noise.py) is designed for:
    add fingerprint noise
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys
import cv2
import random
import copy

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


def GaussianNoise(src, means, sigma, percentage):
    NoiseImg = src
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(1, src.shape[0] - 2)
        randY = random.randint(1, src.shape[1] - 2)
        NoiseImg[randX - 1 : randX + 1, randY - 1 : randY + 1] = NoiseImg[
            randX - 1 : randX + 1, randY - 1 : randY + 1
        ] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def crop_patch(mask, translation, rotation, rot_center):
    back_mask = np.ones_like(mask) * 255
    height, width = mask.shape[0], mask.shape[1]
    M = cv2.getRotationMatrix2D(rot_center, rotation, 1)
    M[:, 2] = M[:, 2] + translation
    back_mask = cv2.warpAffine(
        back_mask,
        M,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask[back_mask == 0] = 0
    return mask

def AddCut(
    mask,
    pCut=0,
    cut_param=[0, 50, 100, 200, -20, 20]
    ):
    if random.random() < pCut:
        rot_center = (mask.shape[1] // 2, mask.shape[0] // 2)
        translation = np.rint(
            [
                np.random.uniform(cut_param[0], cut_param[1]),
                np.random.uniform(cut_param[2], cut_param[3]),
            ]
            * np.random.choice([1, -1], 2)
        ).astype(np.int)
        rotation = random.randint(cut_param[4], cut_param[5])
        square_mask = np.ones_like(mask)
        cut_mask = crop_patch(square_mask, translation, rotation, rot_center)
    

        mask[cut_mask==0] = 0
    return mask


def AddBackgound(
    img,
    mask,
    bimg,
    back,
    pCut=0,
    cut_param=[0, 50, 100, 200, -20, 20],
    pBack=0.5,
    pInv=0.2,
    back_noise=[0.2, 0.4],
    pGauss=0.5,
    gaussianBlur_core=3,
    GaussianNoise_sigma=10,
):
    img = copy.deepcopy(img)
    bimg = copy.deepcopy(bimg)
    mask = copy.deepcopy(mask)
    back = copy.deepcopy(back)

    [hi, wi] = img.shape

    if random.random() < pBack:
        [hb, wb] = back.shape
        # invert color
        if random.random() < pInv:
            img = 255 - img

        # Scale range of random background superimposition: the larger the value, the greater the noise
        back_weight = random.uniform(back_noise[0], back_noise[1])

        # Randomly intercept background image block
        rand_x = random.randint(0, hb - hi)
        rand_y = random.randint(0, wb - wi)
        back_patch = back[rand_x : rand_x + hi, rand_y : rand_y + wi]
        img = back_weight * back_patch + (1 - back_weight) * img

        if random.random() < pGauss:
            # Gaussian blur: (3, 3), (5, 5), (7, 7) the larger the number, the more blurred
            img = cv2.GaussianBlur(img, (gaussianBlur_core, gaussianBlur_core), 3)
            # Point noise addition: the larger the third parameter, the stronger the noise
            img = GaussianNoise(img, 0, GaussianNoise_sigma, 0.1)

    # cut img
    if random.random() < pCut:
        rot_center = (img.shape[1] // 2, img.shape[0] // 2)
        translation = np.rint(
            [
                np.random.uniform(cut_param[0], cut_param[1]),
                np.random.uniform(cut_param[2], cut_param[3]),
            ]
            * np.random.choice([1, -1], 2)
        ).astype(np.int)
        rotation = random.randint(cut_param[4], cut_param[5])
        square_mask = np.ones_like(mask)
        cut_mask = crop_patch(square_mask, translation, rotation, rot_center)
    
        img[cut_mask==0] = 255
        bimg[cut_mask==0] = 255
        mask[cut_mask==0] = 0

    return img, mask,bimg


if __name__ == "__main__":
    prefix = ""
