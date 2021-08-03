"""
This file (fp_pose.py) is designed for:
    functions for fingerprint pose
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def draw_pose(ax, pose, length=100, color="blue"):
    x, y, theta = pose
    start = (x, y)
    end = (x - length * np.sin(theta * np.pi / 180.0), y - length * np.cos(theta * np.pi / 180.0))
    ax.plot(start[0], start[1], marker="o", color=color)
    ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=2, fc=color, ec=color)


def draw_img_with_pose(img, pose, save_path, cmap="gray", vmin=None, vmax=None, mask=None, length=100, color="blue"):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_pose(ax, pose, length=length, color=color)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    prefix = ""
