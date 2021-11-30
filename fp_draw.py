"""
This file (fp_draw.py) is designed for:
    functions for draw
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
import matplotlib.pylab as plt
from matplotlib.patches import ConnectionPatch

from .uni_io import mkdir


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


def draw_minutiae(ax, mnt_lst, arrow_length=15, color="red", linewidth=1.5):
    for mnt in mnt_lst:
        try:
            x, y, ori = mnt[:3]
            dx, dy = arrow_length * np.cos(ori * np.pi / 180), arrow_length * np.sin(ori * np.pi / 180)
            ax.scatter(x, y, marker="s", facecolors="none", edgecolor=color, linewidths=linewidth)
            ax.plot([x, x + dx], [y, y + dy], "-", color=color, linewidth=linewidth)
        except:
            x, y = mnt[:2]
            ax.scatter(x, y, marker="s", facecolors="none", edgecolor=color, linewidths=linewidth)


def draw_minutia_on_finger(
    img, mnt_lst, save_path, cmap="gray", vmin=None, vmax=None, arrow_length=15, color="red", linewidth=1.5
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_minutiae(ax, mnt_lst, arrow_length, color, linewidth)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    mkdir(osp.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def draw_minutiae_pair(
    ax, img1, img2, mnts1, mnts2, cmap="gray", vmin=None, vmax=None, markercolor="red", linecolor="green", linewidth=1.5
):
    img_shape1 = np.array(img1.shape[:2])
    img_shape2 = np.array(img2.shape[:2])
    img_height = max(img_shape1[0], img_shape2[0])
    img1 = np.pad(img1, ((0, img_height - img_shape1[0])), mode="edge")
    img2 = np.pad(img2, ((0, img_height - img_shape2[0])), mode="edge")
    img = np.concatenate((img1, img2), axis=1)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    mnts2 = mnts2.copy()
    mnts2[:, 0] += img1.shape[1]
    for ii in range(len(mnts1)):
        ax.scatter(
            mnts1[ii, 0], mnts1[ii, 1], marker="s", s=5, facecolors="none", edgecolor=markercolor, linewidths=linewidth,
        )
        ax.scatter(
            mnts2[ii, 0], mnts2[ii, 1], marker="s", s=5, facecolors="none", edgecolor=markercolor, linewidths=linewidth,
        )
        ax.plot(
            [mnts1[ii, 0], mnts2[ii, 0]],
            [mnts1[ii, 1], mnts2[ii, 1]],
            "-",
            color=linecolor,
            markersize=3,
            markerfacecolor="none",
        )
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()


def draw_minutiae_pair_on_finger(
    img1,
    img2,
    mnts1,
    mnts2,
    save_path,
    cmap="gray",
    vmin=None,
    vmax=None,
    markercolor="red",
    linecolor="green",
    linewidth=1.5,
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_minutiae_pair(
        ax, img1, img2, mnts1, mnts2, cmap, vmin, vmax, markercolor=markercolor, linecolor=linecolor, linewidth=linewidth,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    prefix = ""
