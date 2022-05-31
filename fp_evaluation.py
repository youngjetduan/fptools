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
import seaborn


def draw_distribution(ax, scores, label=None, color="r", linestyle="-", linewidth=1.5):
    """draw matching score distribution

    Parameters:
        scores
    Returns:
        [None]
    """
    scores = scores.astype(np.float32).flatten()
    seaborn.kdeplot(scores, shade=False, label=label, color=color, linestyle=linestyle, linewidth=linewidth, ax=ax)


def draw_distribution2(ax, score_mat, linewidth=1.5):
    """draw matching score distribution

    Parameters:
        score_mat: a dict contains {"genuine": ..., "impostor": ...}
    Returns:
        [None]
    """
    genuine_scores = np.array(score_mat["genuine"]).astype(np.float32).flatten()
    impostor_scores = np.array(score_mat["impostor"]).astype(np.float32).flatten()

    seaborn.kdeplot(genuine_scores, shade=False, label="Genuine", linewidth=linewidth, ax=ax)
    ax2 = ax.twinx()
    seaborn.kdeplot(impostor_scores, shade=False, label="Impostor", linewidth=linewidth, ax=ax2)


def compute_roc(score_mat, num_steps=100, gms_only=False):
    """compute ROC curve.

    Parameters:
        score_mat: a dict contains {"genuine": ..., "impostor": ...}
        gms_only: generally for VeriFinger, only genunie scores are required
    Returns:
        rate_arr: [false-positive-rate; true-positive-rate]
    """
    genuine_scores = np.array(score_mat["genuine"]).astype(np.float32).flatten()

    if gms_only:
        num_steps = min(num_steps, len(genuine_scores))
        rate_arr = np.zeros([2, num_steps])

        rate_arr[0] = np.linspace(-8, 0, num_steps)
        threshs = -rate_arr[0] * 12
        for ii, c_th in enumerate(threshs):
            rate_arr[1, ii] = (genuine_scores >= c_th).sum() * 1.0 / len(genuine_scores)
        rate_arr[0] = 10 ** rate_arr[0]
    else:
        impostor_scores = np.array(score_mat["impostor"]).astype(np.float32).flatten()
        num_steps = min(num_steps, len(genuine_scores) + len(impostor_scores))
        rate_arr = np.zeros([2, num_steps])

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
        for ii, c_th in enumerate(threshs):
            rate_arr[0, ii] = (impostor_scores >= c_th).sum() * 1.0 / len(impostor_scores)
            rate_arr[1, ii] = (genuine_scores >= c_th).sum() * 1.0 / len(genuine_scores)

    return rate_arr


def compute_cmc(score_mat, max_rank=30):
    """compute rank array for cmc curve. Note that search names must the subset of file names!

    Parameters:
        score_mat: a dict contains {"score": ..., "genuine_scores": ...}
    Returns:
        rank_arr: length equal to max_rank
    """
    score_arr = score_mat["score"]
    genuine_scores = score_arr["genuine_scores"]

    idx_arr = (score_arr > genuine_scores.reshape(1, -1)).sum(-1)
    rank_arr = np.zeros(max_rank)
    N = len(genuine_scores)
    for ii in range(max_rank):
        rank_arr[ii] = (idx_arr <= ii).sum() * 1.0 / N

    # search_names = np.array(score_mat["search_names"])
    # file_names = np.array(score_mat["file_names"])

    # score_arr = np.argsort(score_arr, axis=1)[:, ::-1]
    # sorted_names = file_names[score_arr[:, :max_rank]]

    # rank_arr = 1 * (sorted_names == search_names[:, None])
    # rank_arr = np.cumsum(rank_arr, axis=1)
    # rank_arr = rank_arr.sum(axis=0) * 1.0 / len(search_names)

    return rank_arr


def draw_roc(ax, rate_arr, profile="k-", linewidth=1.5, label=""):
    ax.plot(rate_arr[0], rate_arr[1], profile, linewidth=linewidth, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("FNR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve using Verifinger")


def draw_cmc(ax, rank_arr, profile="k-s", linewidth=1.5, label=""):
    x = np.arange(len(rank_arr))
    ax.plot(x, rank_arr, profile, linewidth=linewidth, label=label)


def draw_det(ax, rate_arr, profile="k-", linewidth=2, label=""):
    ax.plot(rate_arr[0], 1 - rate_arr[1], profile, linewidth=linewidth, label=label)


def normalize_det_lines(ax, title="DET curve using Verifinger", xlim=[0, 1], ylim=[0, 1]):
    ax.set_xscale("log")
    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.set_title(title)
    ax.grid(which="minor", axis="both", linestyle="--", linewidth=0.5)

    ax.legend(loc="best")

    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])


def normalize_det_log_lines(ax, title="DET curve using Verifinger", xlim=[1e-4, 1], ylim=[1e-3, 1]):

    # EER line
    points = np.array([-4, 1])
    ax.plot(points, points, linewidth=1, linestyle="-", color="k")

    # draw wide grid lines
    for ii in range(-3, 1):
        y = np.array([np.power(10, 1.0 * ii), np.power(10, 1.0 * ii)])
        x = np.array([np.power(10, -5.0), np.power(10, 0)])
        ax.plot(x, y, linewidth=1, linestyle="-", color="k")
    for jj in range(-4, 1):
        y = np.array([np.power(10, -4.0), np.power(10, 0)])
        x = np.array([np.power(10, 1.0 * jj), np.power(10, 1.0 * jj)])
        ax.plot(x, y, linewidth=1, linestyle="-", color="k")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.set_title(title)
    ax.grid(which="minor", axis="both", linestyle="--", linewidth=0.5)

    ax.legend(loc="lower left")

    ax.text(np.power(10, -3.95), np.power(10, -0.75), "FMR10000", color="darkred", rotation=90)
    ax.text(np.power(10, -2.95), np.power(10, -0.75), "FMR1000", color="darkred", rotation=90)
    ax.text(np.power(10, -1.95), np.power(10, -0.75), "FMR100", color="darkred", rotation=90)
    ax.text(np.power(10, -0.95), np.power(10, -0.75), "EER", color="darkred", rotation=45)

    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])


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
