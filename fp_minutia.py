"""
This file (fp_minutia.py) is designed for:
    functions for minutiae
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
import matplotlib.pylab as plt


def draw_minutiae(ax, mnt_lst, R=15, arrow_length=15, color="red", linewidth=1.0):
    for mnt in mnt_lst:
        x, y, ori = mnt[:3]
        dx, dy = arrow_length * np.cos(-ori * np.pi / 180), arrow_length * np.sin(-ori * np.pi / 180)
        ax.scatter(x, y, marker="o", edgecolor=color, linewidths=linewidth)
        ax.plot([x, x + dx], [y, y + dy], "-", color=color, linewidth=linewidth)


def draw_minutia_on_finger(
    img, mnt_lst, save_path, cmap=None, vmin=None, vmax=None, R=10, arrow_length=15, color="red", linewidth=1.0
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_minutiae(ax, mnt_lst, R, arrow_length, color, linewidth)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    prefix = ""
