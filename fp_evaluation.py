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
from scipy.stats import norm
import matplotlib.pyplot as plt


def compute_roc(score_mat, num_steps=100):
    """ compute ROC curve.
    
    Parameters:
        score_mat: a dict contains {"genuine": ..., "impostor": ...}
    Returns:
        rate_arr: [false-positive-rate; true-positive-rate]
    """
    genuine_scores = score_mat["genuine"].astype(np.float32).flatten()
    impostor_scores = score_mat["impostor"].astype(np.float32).flatten()
    num_steps = min(num_steps, len(genuine_scores) + len(impostor_scores))

    threshs = np.concatenate(
        (
            np.linspace(
                min(genuine_scores.min(), impostor_scores.min()),
                max(genuine_scores.max(), impostor_scores.max()),
                num_steps // 2,
            ),
            np.random.choice(
                np.concatenate((genuine_scores, impostor_scores)), num_steps - num_steps // 2, replace=False
            ),
        )
    )
    threshs = np.sort(threshs)
    rate_arr = np.zeros([2, num_steps])
    for ii, c_th in enumerate(threshs):
        rate_arr[0, ii] = (impostor_scores >= c_th).sum() * 1.0 / len(impostor_scores)
        rate_arr[1, ii] = (genuine_scores >= c_th).sum() * 1.0 / len(genuine_scores)
    return rate_arr


def compute_cmc(score_mat, max_rank=30):
    """ compute rank array for cmc curve. Note that search names must the subset of file names!
    
    Parameters:
        score_mat: a dict contains {"score": ..., "search_names": ..., "file_names": ...}
    Returns:
        rank_arr: length equal to max_rank
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


def draw_roc(ax, rate_arr, profile="k-", linewidth=1.5, label=""):
    ax.plot(rate_arr[0], rate_arr[1], profile, linewidth=linewidth, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("FNR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve using Verifinger")


def draw_det(ax, rate_arr, profile="k-", linewidth=1.5, label=""):
    ax.plot(rate_arr[0], 1 - rate_arr[1], profile, linewidth=linewidth, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.grid("on", linestyle="--")
    ax.set_title("DET curve using Verifinger")


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
