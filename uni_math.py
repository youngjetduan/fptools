"""
This file (uni_math.py) is designed for:
    functions for mathematical processing
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys
import os.path as osp
import numpy as np
from glob import glob
import scipy.linalg as slg
from scipy.spatial import distance
from scipy.spatial.transform import Rotation


def asStride(arr, sub_shape, stride):
    """Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    """
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape
    view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
    strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs


def poolingOverlap(mat, ksize, stride=None, method="max", pad=False):
    """Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    """

    m, n = mat.shape[:2]
    ky, kx = ksize
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, sy)
        nx = _ceil(n, sx)
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[: (m - ky) // sy * sy + ky, : (n - kx) // sx * sx + kx, ...]

    view = asStride(mat_pad, ksize, stride)

    if method == "max":
        result = np.nanmax(view, axis=(2, 3))
    else:
        result = np.nanmean(view, axis=(2, 3))

    return result


def distance_pc(pts_A, pts_B, ratio_sample=0.05, min_samples=1000, hemi=False):
    """ distance between two point cloud
    
    Parameters:
        hemi: if unilateral distance
    Returns:
        [None]
    """
    N_A = len(pts_A)
    N_B = len(pts_B)

    sample_A = max(min_samples, np.rint(ratio_sample * N_A).astype(int))
    sample_A = np.random.choice(np.arange(N_A), sample_A, replace=False).astype(int)
    sample_A = pts_A[sample_A]

    sample_B = max(min_samples, np.rint(ratio_sample * N_B).astype(int))
    sample_B = np.random.choice(np.arange(N_B), sample_B, replace=False).astype(int)
    sample_B = pts_B[sample_B]

    dist = distance.cdist(sample_A, sample_B)
    if hemi:
        error = dist.min(axis=1).mean()
    else:
        error = dist.min(axis=0).mean() + dist.min(axis=1).mean()
    return error


def estimate_affine_transform(pts_src, pts_tar):
    """ estimate affine transform based on 3 points only
    
    Parameters:
        [None]
    Returns:
        [None]
    """
    pts_src_c = pts_src.mean(axis=0)
    pts_tar_c = pts_tar.mean(axis=0)

    H = np.dot(pts_src.T - pts_src_c[:, None], pts_tar - pts_tar_c[None])
    U, S, V = slg.svd(H)
    R = np.dot(V, U.T)
    if slg.det(R) < 0:
        U, S, V = slg.svd(R)
        V[:, -1] *= -1
        R = np.dot(V, U.T)
    t = pts_tar_c - np.dot(R, pts_src_c)
    return R, t


def euler_to_R(euler_angle, is_deg=True):
    """ euler angle to rotation matrix
    
    Parameters:
        euler_angle: [roll, pitch, yaw]
    Returns:
        [None]
    """
    R = Rotation.from_euler("ZYX", [euler_angle[2], euler_angle[1], euler_angle[0]], degrees=is_deg).as_matrix()

    # if is_deg:
    #     roll = euler_angle[0] * np.pi / 180
    #     pitch = euler_angle[1] * np.pi / 180
    #     yaw = euler_angle[2] * np.pi / 180
    # R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    # R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    # R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    # R = np.dot(R_z, np.dot(R_y, R_x))
    # print(R)
    return R


def R_to_euler(R, is_deg=True):
    euler = Rotation.from_matrix(R).as_euler("ZYX", degrees=is_deg)[::-1]

    # roll = np.arctan2(R[2, 1], R[2, 2])
    # pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    # yaw = np.arctan2(R[1, 0], R[0, 0])
    # euler = np.array([roll, pitch, yaw])
    # if is_deg:
    #     euler = euler * 180 / np.pi
    # print("self", euler)

    return euler


if __name__ == "__main__":
    prefix = ""
