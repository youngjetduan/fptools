"""
This file (fp_indexing.py) is designed for:
    refer to Su's MCC(BMC)-indexing
Copyright (c) 2022, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
import tqdm
import pickle
from scipy import io as sio, spatial as ssp
import functools
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from fptools import fp_mcc, uni_io


def transform_points(pts, trans=np.zeros(2), theta=0):
    theta_rad = np.deg2rad(theta)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])
    new_pts = np.zeros_like(pts)
    new_pts[:, :2] = np.dot(pts[:, :2] - trans[None], R.T)
    new_pts[:, 2:] = pts[:, 2:]
    new_pts[:, 2] = (pts[:, 2] + theta + 180) % 360 - 180

    return new_pts


def rectify_minutiae(input):
    mcc_path, pose_path = input
    try:
        mcc_des = fp_mcc.load_mcc_feature(mcc_path)
        indices = np.where(mcc_des["flag"] > 0)[0]
        n_minu = len(indices)
        if n_minu == 0:
            return np.zeros((0, 3))

        try:
            pose = np.loadtxt(pose_path, delimiter=",")
        except:
            pose = np.loadtxt(pose_path)
        minu_align = transform_points(mcc_des["mnt"][indices, :3], trans=pose[:2], theta=pose[2])
        return minu_align
    except:
        return np.zeros((0, 3))


def flatting_mcc(input, params):
    ii, mcc_path = input
    try:
        mcc_des = fp_mcc.load_mcc_feature(mcc_path)
        indices = np.where(mcc_des["flag"] > 0)[0]
        n_minu = len(indices)
        if n_minu == 0:
            return np.zeros((0, 5 + params.n_bits))

        minutiae = mcc_des["mnt"][indices, :3]
        vector = mcc_des["cm"]
        return np.concatenate((np.ones((len(indices), 1)) * ii, indices[:, None], minutiae, vector), axis=1)
    except Exception as ex:
        print(ex)
        return np.zeros((0, 5 + params.n_bits))


def create_indexing_table(mcc_names, save_path, funcs, n_proc=8):
    params = fp_mcc.MCCParameters(is_binary=True)

    with Pool(processes=n_proc) as tp:
        mcc_array = list(
            tqdm.tqdm(tp.imap(functools.partial(flatting_mcc, params=params), enumerate(mcc_names)), total=len(mcc_names))
        )
    mcc_array = np.concatenate(mcc_array, axis=0)
    minu_indices = mcc_array[:, :2]
    minutiae = mcc_array[:, 2:5]
    mcc_array = mcc_array[:, 5:].astype(bool)

    bin2dec_mat = 2 ** np.arange(funcs.shape[1])[::-1].astype(params.key_dtype)

    hash_tables = [{} for _ in range(params.n_func)]
    for ff in range(params.n_func):
        keys = mcc_array[:, funcs[ff] - 1]
        key_vals = keys.dot(bin2dec_mat)
        indices = np.where(keys.sum(axis=1) >= params.n_minpc)[0]
        key_vals = key_vals[indices]
        hash_tables[ff] = {k: indices[key_vals == k] for k in np.unique(key_vals)}

    uni_io.mkdir(osp.dirname(save_path))
    with open(save_path, "wb") as fp:
        pickle.dump(
            {
                "funcs": funcs,
                "hash_table": hash_tables,
                "gallery_size": len(mcc_names),
                "gallery_name": [osp.basename(x).replace(".mat", "") for x in mcc_names],
                "minu_idx": minu_indices.astype(int),
                "minu": minutiae,
                "param": params,
            },
            fp,
        )


def minu_mcc_indexing(input, gallery_minu, gallery_minu_idx, hash_table, gallery_size, coef, params):
    # cur_minu, cur_key_vals = input
    cur_minu, cur_key_vals = input[:3], input[3:]
    scores = np.zeros(gallery_size)

    q_cands = [hash_table[f][cur_key_vals[f]] for f in range(len(hash_table)) if cur_key_vals[f] in hash_table[f]]
    if len(q_cands) == 0:
        return scores
    q_cands = np.concatenate(q_cands)
    match_minu = gallery_minu[q_cands]

    loc_diff = np.linalg.norm(match_minu[:, :2] - cur_minu[None, :2], axis=1)
    rot_diff = (match_minu[:, 2] - cur_minu[2] + 180) % 360 - 180
    diff_mask = (loc_diff < params.delta_xy) & (np.abs(rot_diff) < params.delta_theta)
    q_cands = q_cands[diff_mask]
    if len(q_cands) == 0:
        return scores

    q_cands_unique, q_cands_counts = np.unique(q_cands, return_counts=True)
    q_cands_indices = gallery_minu_idx[q_cands_unique, 0]
    tpl = np.unique(q_cands_indices)
    for tt in range(len(tpl)):
        tmp_counts = q_cands_counts[q_cands_indices == tpl[tt]]
        scores[tpl[tt]] = scores[tpl[tt]] + tmp_counts.max() ** coef

    return scores


def apply_mcc_indexing(input, gallery_table, params):
    mcc_path, pose_path = input

    funcs = gallery_table["funcs"]
    hash_table = gallery_table["hash_table"]
    gallery_size = gallery_table["gallery_size"]
    gallery_minu_idx = gallery_table["minu_idx"].astype(int)
    gallery_minu = gallery_table["minu"]
    gallery_params = gallery_table["param"]

    coef = gallery_params.p / gallery_params.n_select

    try:
        mcc_des = fp_mcc.load_mcc_feature(mcc_path)
        indices = np.where(mcc_des["flag"] > 0)[0]
        n_minu = len(indices)
        if n_minu == 0:
            return 0

        minutiae = mcc_des["mnt"][indices, :3]
        if pose_path is not None:
            try:
                pose = np.loadtxt(pose_path, delimiter=",")
            except:
                pose = np.loadtxt(pose_path)
            minutiae = transform_points(minutiae, trans=pose[:2], theta=pose[2])

        vector = mcc_des["cm"]
        query_mcc = np.concatenate((minutiae, vector), axis=1)
    except:
        return 0

    q_minu = query_mcc[:, :3]
    q_mcc = query_mcc[:, 3:].astype(bool)

    bin2dec_mat = 2 ** np.arange(funcs.shape[1])[::-1].astype(gallery_params.key_dtype)
    q_keys = q_mcc[:, funcs - 1]
    q_key_vals = q_keys.dot(bin2dec_mat)

    # scores = [
    #     minu_mcc_indexing(x, gallery_minu, gallery_minu_idx, hash_table, gallery_size, coef, params)
    #     for x in zip(q_minu, q_key_vals)
    # ]
    # scores = np.stack(scores, axis=0).sum(axis=0)

    arr = np.concatenate((q_minu, q_key_vals), axis=1)
    scores = np.apply_along_axis(
        minu_mcc_indexing,
        axis=1,
        arr=arr,
        gallery_minu=gallery_minu,
        gallery_minu_idx=gallery_minu_idx,
        hash_table=hash_table,
        gallery_size=gallery_size,
        coef=coef,
        params=params,
    )
    scores = scores.sum(axis=0)

    # scores = np.zeros(gallery_size)
    # for cur_ii in range(len(q_minu)):
    #     cur_minu = q_minu[cur_ii]
    #     cur_key_vals = q_key_vals[cur_ii]

    #     q_cands = [hash_table[f][cur_key_vals[f]] for f in range(len(hash_table)) if cur_key_vals[f] in hash_table[f]]
    #     if len(q_cands) == 0:
    #         continue
    #     q_cands = np.concatenate(q_cands)
    #     match_minu = gallery_minu[q_cands]

    #     loc_diff = np.linalg.norm(match_minu[:, :2] - cur_minu[None, :2], axis=1)
    #     rot_diff = (match_minu[:, 2] - cur_minu[2] + 180) % 360 - 180
    #     diff_mask = (loc_diff < params.delta_xy) & (np.abs(rot_diff) < params.delta_theta)
    #     q_cands = q_cands[diff_mask]
    #     if len(q_cands) == 0:
    #         continue

    #     q_cands_unique, q_cands_counts = np.unique(q_cands, return_counts=True)
    #     q_cands_indices = gallery_minu_idx[q_cands_unique, 0]
    #     tpl = np.unique(q_cands_indices)
    #     for tt in range(len(tpl)):
    #         tmp_counts = q_cands_counts[q_cands_indices == tpl[tt]]
    #         scores[tpl[tt]] = scores[tpl[tt]] + tmp_counts.max() ** coef

    scores = scores / len(q_minu) / (gallery_params.n_func ** coef)
    return scores
