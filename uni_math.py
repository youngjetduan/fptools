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


def euler_to_R(roll, pitch, yaw, is_deg=True):
    """ euler angle to rotation matrix
    
    Parameters:
        [None]
    Returns:
        [None]
    """
    R = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=is_deg).as_matrix()

    # if is_deg:
    #     roll = roll * np.pi / 180
    #     pitch = pitch * np.pi / 180
    #     yaw = yaw * np.pi / 180
    # R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    # R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    # R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    # R = np.dot(R_z, np.dot(R_y, R_x))
    return R


if __name__ == "__main__":
    prefix = ""
