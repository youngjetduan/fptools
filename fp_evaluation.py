"""
This file (fp_evaluation.py) is designed for:
    functions for fingerprint evaluation
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def compute_cmc(score_mat, max_rank=30):
    """ compute rank array for cmc curve. Note that search names must the subset of file names!
    
    Parameters:
        score_mat: a dict contains {"score": ..., "search_names": ..., "file_names": ...}
    Returns:
        [None]
    """
    score_arr = score_mat["score"]
    search_names = np.array(score_mat["search_names"])
    file_names = np.array(score_mat["file_names"])

    score_arr = np.argsort(score_arr, axis=1)[:, ::-1]
    sorted_names = file_names[score_arr[:, :max_rank]]

    rank_arr = 1 * (sorted_names == search_names[:, None])
    rank_arr = np.cumsum(rank_arr, axis=1)
    rank_arr = rank_arr.sum(axis=0) * 1.0 / len(search_names)

    return rank_arr


def draw_cmc(ax, rank_arr, profile="k-s", linewidth=1.5, label=""):
    x = np.arange(len(rank_arr))
    ax.plot(x, rank_arr, profile, linewidth=linewidth, label=label)


if __name__ == "__main__":
    score_mat = {
        "score": np.random.random([10, 100]),
        "search_names": [f"{x}" for x in range(10)],
        "file_names": [f"{x}" for x in range(100)],
    }

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rank_arr = compute_cmc(score_mat, max_rank=30)
    draw_cmc(ax, rank_arr)

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("test_cmc.png", bbox_inches="tight")
    plt.close(fig)
