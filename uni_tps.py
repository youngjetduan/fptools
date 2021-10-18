"""
This file (uni_tps.py) is designed for:
    implementation of TPS using pytorch
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
import cv2
import scipy

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    pairwise_diff = input_points[:, None] - control_points[None]
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[..., 0] + pairwise_diff_square[..., 1]
    # fix numerical error for 0 * log(0), substitute all nan with 0
    repr_matrix = 0.5 * pairwise_dist * np.log(pairwise_dist.clip(1e-3, None))
    mask = (repr_matrix != repr_matrix) | np.isclose(pairwise_dist, 0)
    repr_matrix[mask] = 0
    return repr_matrix


def opencv_tps(img, source, target, mode=1, border_value=0):
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.setRegularizationParameter(0.01)

    source_ts = source[None]
    target_ts = target[None]
    matches = []
    for ii in range(len(source)):
        matches.append(cv2.DMatch(ii, ii, 0))
    tps.estimateTransformation(target_ts, source_ts, matches)
    if mode == 0:
        flags = cv2.INTER_NEAREST
    elif mode == 1:
        flags = cv2.INTER_LINEAR

    return tps.warpImage(img, flags=flags, borderValue=border_value)


def normalization(points, img_shape):
    points = points * 1.0 / (np.array(img_shape)[None, ::-1] - 1) * 2 - 1
    return points


def tps_module_numpy(src_cpts, tar_cpts):
    assert tar_cpts.ndim == 2
    assert tar_cpts.shape[1] == 2
    N = src_cpts.shape[0]
    src_cpts = src_cpts.astype(np.float32)
    tar_cpts = tar_cpts.astype(np.float32)

    # create padded kernel matrix
    src_cc_partial_repr = compute_partial_repr(src_cpts, src_cpts)
    forward_kernel = np.concatenate(
        (
            np.concatenate((src_cc_partial_repr, np.ones([N, 1]), src_cpts), axis=1),
            np.concatenate((np.ones([1, N]), np.zeros([1, 3])), axis=1),
            np.concatenate((src_cpts.T, np.zeros([2, 3])), axis=1),
        ),
        axis=0,
    )
    # compute mapping matrix
    Y = np.concatenate([tar_cpts, np.zeros([3, 2])], axis=0)  # (M+3,2)
    mapping_matrix = scipy.linalg.solve(forward_kernel, Y)
    return mapping_matrix


def tps_apply_transform(src_pts, src_cpts, mapping_matrix):
    """
    Parameters:
        src_pts: points to be transformed
        src_cpts: control points
    Returns:
        [None]
    """
    assert src_pts.ndim == 2
    src_pc_partial_repr = compute_partial_repr(src_pts, src_cpts)
    N = src_pts.shape[0]
    src_pts_repr = np.concatenate([src_pc_partial_repr, np.ones([N, 1]), src_pts], axis=1)
    tar_pts = np.matmul(src_pts_repr, mapping_matrix)
    return tar_pts


if __name__ == "__main__":
    prefix = ""
