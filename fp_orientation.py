"""
This file (fp_orientation.py) is designed for:
    functions for fingerprint orientation field
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, map_coordinates


def minus_orientation(ori, anchor):
    ori = ori - anchor
    ori = np.where(ori >= 90, ori - 180, ori)
    ori = np.where(ori < -90, ori + 180, ori)
    return ori


def zoom_orientation(ori, scale):
    cos_2ori = np.cos(ori * np.pi / 90)
    sin_2ori = np.sin(ori * np.pi / 90)
    cos_2ori = zoom(cos_2ori, scale, order=1)
    sin_2ori = zoom(sin_2ori, scale, order=1)
    ori = np.arctan2(sin_2ori, cos_2ori) * 90 / np.pi
    return ori


def transform_to_reference(arr, pose, tar_shape=None, order=0, cval=0, factor=8, is_ori=False):
    """ transform array to standard pose
    
    Parameters:
        [None]
    Returns:
        [None]
    """
    x, y, theta = pose
    tar_shape = np.array(arr.shape[-2:]) if tar_shape is None else tar_shape
    center = tar_shape // 2
    finger_center = np.array([y, x]).astype(np.float32) / factor
    sin_theta = np.sin(theta * np.pi / 180.0)
    cos_theta = np.cos(theta * np.pi / 180.0)
    mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    indices = np.stack(np.meshgrid(*[np.arange(x) for x in tar_shape], indexing="ij")).astype(np.float32)
    coord_indices = indices[-2:].reshape(2, -1)
    coord_indices = np.dot(mat, coord_indices - center[:, None]) + finger_center[:, None]
    indices[-2:] = coord_indices.reshape(2, *tar_shape)
    if is_ori:
        new_arr = arr + theta
        cos_2angle = map_coordinates(np.cos(2 * new_arr * np.pi / 180), indices, order=order, mode="nearest")
        sin_2angle = map_coordinates(np.sin(2 * new_arr * np.pi / 180), indices, order=order, mode="nearest")
        new_arr = np.arctan2(sin_2angle, cos_2angle) * 180 / np.pi / 2
    else:
        new_arr = map_coordinates(arr, indices, order=order, cval=cval)
    return new_arr


def transform_to_target(arr, pose, tar_shape=None, factor=8.0, angle=False, order=1):
    """ transform array to target pose
    
    Parameters:
        [None]
    Returns:
        [None]
    """
    x, y, theta = pose
    tar_shape = np.array(arr.shape[-2:]) if tar_shape is None else tar_shape
    center = np.array(arr.shape[-2:]) // 2
    finger_center = np.array([y, x]).astype(np.float32) / factor
    sin_theta = np.sin(theta * np.pi / 180.0)
    cos_theta = np.cos(theta * np.pi / 180.0)
    mat = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    indices = np.stack(np.meshgrid(*[np.arange(x) for x in tar_shape], indexing="ij")).astype(np.float32)
    coord_indices = indices[-2:].reshape(2, -1)
    coord_indices = np.dot(mat, coord_indices - finger_center[:, None]) + center[:, None]
    indices[-2:] = coord_indices.reshape(2, *tar_shape)
    if angle:
        new_arr = arr - theta
        cos_2angle = map_coordinates(np.cos(2 * new_arr * np.pi / 180), indices, order=1, mode="nearest")
        sin_2angle = map_coordinates(np.sin(2 * new_arr * np.pi / 180), indices, order=1, mode="nearest")
        new_arr = np.arctan2(sin_2angle, cos_2angle) * 180 / np.pi / 2
    else:
        new_arr = map_coordinates(arr, indices, order=order, mode="nearest")
    return new_arr


def draw_orientation(ax, ori, mask=None, factor=8, stride=32, color="lime", linewidth=1.5):
    """ draw orientation filed
    
    Parameters:
        [None]
    Returns:
        [None]
    """
    ori = ori * np.pi / 180
    for ii in range(stride // factor // 2, ori.shape[0], stride // factor):
        for jj in range(stride // factor // 2, ori.shape[1], stride // factor):
            if mask is not None and mask[ii, jj] == 0:
                continue
            x, y, o, r = jj, ii, ori[ii, jj], stride * 0.8
            ax.plot(
                [x * factor - 0.5 * r * np.cos(o), x * factor + 0.5 * r * np.cos(o)],
                [y * factor - 0.5 * r * np.sin(o), y * factor + 0.5 * r * np.sin(o)],
                "-",
                color=color,
                linewidth=linewidth,
            )


def draw_img_with_orientation(
    img, ori, save_path, factor=8, stride=16, cmap="gray", vmin=None, vmax=None, mask=None, color="lime", dpi=100
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_orientation(ax, ori, mask=mask, factor=factor, stride=stride, color=color, linewidth=dpi / 50)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.set_size_inches(img.shape[1] * 1.0 / dpi, img.shape[0] * 1.0 / dpi)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

