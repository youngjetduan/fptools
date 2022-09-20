'''
Descripttion: 
version: 
Author: Xiongjun Guan
Date: 2021-07-21 12:33:18
LastEditors: Xiongjun Guan
LastEditTime: 2021-08-14 17:52:57
'''

import os
import os.path as osp
import scipy.io as scio
import argparse
from Functions import Euler2Matrix, PoseCorrection, PointsFlattenByCircle, GenerateFlattenResult, DrawPCA_2D
import imageio
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import pptk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi",
                        "-d",
                        default=500,
                        type=int,
                        help="generated figure's dpi")
    parser.add_argument("--edge",
                        "-e",
                        default=30,
                        type=int,
                        help="blank edge's pixels")
    parser.add_argument("--brightness",
                        "-b",
                        default=0.6,
                        type=float,
                        help="Luminance coefficient of image")
    args = parser.parse_args()

    data_dir = "./data"
    save_dir = "./result"
    fname = "100_1_2.mat"
    save_name = None

    save_dir = osp.join(
        save_dir, 'dpi' + str(args.dpi) + '_' + 'b' + str(args.brightness))
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if save_name is None:
        save_name = fname.split('.')[0]

    data = scio.loadmat(osp.join(data_dir, fname))
    points = data["points"]
    depth = data["depth"]
    normals = data['normals']

    # # ---- test 'PoseCorrection' function ----
    # M = Euler2Matrix(0, 60, 0)
    # points = np.dot(points, M.T)
    # normals = np.dot(normals, M.T)
    # pptk.viewer(points)
    # DrawPCA_2D(points,0,1)
    # points,normals = PoseCorrection(points,normals)
    # pptk.viewer(points)
    # DrawPCA_2D(points,0,1)
    # plt.show()
    # # ----------------------------------------

    points, normals = PoseCorrection(points, normals)
    uv_points, jump_step = PointsFlattenByCircle(points, args.dpi)

    grid_fp, grid_gt = GenerateFlattenResult(points,
                                             depth,
                                             uv_points,
                                             args.edge,
                                             brightness=args.brightness,
                                             fill_value=255)

    imageio.imwrite(osp.join(save_dir, save_name + '.png'), grid_fp)
    scio.savemat(osp.join(save_dir, save_name + '.mat'), {
        'points': points,
        'normals': normals,
        'depth': depth,
        'grid_gt': grid_gt
    })
