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
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points, eps=0.02):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        self.target_height = target_height
        self.target_width = target_width
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr + eps * torch.eye(N))
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.pinverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat(
            [target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate], dim=1
        )

        # register precomputed matrices
        self.register_buffer("inverse_kernel", inverse_kernel)
        self.register_buffer("padding_matrix", torch.zeros(3, 2))
        self.register_buffer("target_coordinate_repr", target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat(
            [source_control_points, self.padding_matrix.type_as(source_control_points).expand(batch_size, -1, -1)], 1
        )
        mapping_matrix = torch.matmul(self.inverse_kernel.type_as(source_control_points), Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr.type_as(source_control_points), mapping_matrix)
        return source_coordinate.view(batch_size, self.target_height, self.target_width, -1)


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


def tps_grid_torch(img_shape, target, source, device="cpu"):
    source_ts = torch.tensor(normalization(source, img_shape)).double()
    target_ts = torch.tensor(normalization(target, img_shape)).double()

    # warped_grid = TPS()(target_ts.float(), source_ts.float(), img_shape[1], img_shape[0])
    tps = TPSGridGen(img_shape[0], img_shape[1], target_ts)
    warp_grid = tps(source_ts[None])
    return warp_grid.squeeze().detach().numpy()


def apply_deformation(img, warp_grid, mode="bilinear", padding_mode="border", device="cpu"):
    if img.ndim == 2:
        img_ts = torch.tensor(img)[None, None]
    elif img.ndim == 3:
        img_ts = torch.tensor(img)[None]
    else:
        raise ValueError(f"not supported image shape {img.shape}")

    if not torch.is_tensor(warp_grid):
        warp_grid = torch.tensor(warp_grid[None])
    assert warp_grid.size(-1) == 2

    img_ts = F.grid_sample(img_ts.double(), warp_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return img_ts.squeeze().detach().numpy()


if __name__ == "__main__":
    prefix = ""
