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
import time
import pickle
from ctypes import cdll
from scipy import io as sio, spatial as ssp, sparse
import functools
from multiprocessing import Pool

# server 27
# neu_dir = "/mnt/data5/fptools/Verifinger"
# server 33
# neu_dir = "/mnt/data1/dyj"
neu_dir = np.loadtxt(osp.join(osp.dirname(osp.abspath(__file__)), "neu_dir.txt"), str).tolist()

cdll.LoadLibrary(osp.join(neu_dir, "boost", "lib", "libboost_python37.so"))
cdll.LoadLibrary(osp.join(neu_dir, "boost", "lib", "libboost_numpy37.so"))
sys.path.append(osp.join(neu_dir, "MCCIndexing"))

import mcc_indexing as _mcc_indexing
from fptools import fp_mcc, uni_io, GlobalManager as glm


class MCCParameter(_mcc_indexing.MCCParameter):
    def __init__(
        self,
        pm_nbit=312,
        pm_nbitbyte=39,
        pm_minpc=2,
        pm_p=30,
        pm_h=24,
        pm_l=32,
        delta_xy=200,
        delta_a=40,
        hash_path="hashfuncs_local.txt",
    ) -> None:
        super().__init__(
            pm_nbit, pm_nbitbyte, pm_minpc, pm_p, pm_h, pm_l, delta_xy, delta_a, osp.join(osp.dirname(__file__), hash_path)
        )


def mcc_indexing(
    gallery_mcc_paths,
    query_mcc_paths,
    gallery_pose_paths=None,
    query_pose_paths=None,
    delta_xy=200,
    delta_a=40,
    n_proc=8,
    hash_path="hashfuncs_local.txt",
):
    bmc_params = fp_mcc.MCCParameters(is_binary=True)

    # gallery
    with Pool(n_proc) as tp:
        gallery_mccs = list(
            tqdm.tqdm(
                tp.imap(functools.partial(flatting_mcc, params=bmc_params, is_pack=True), enumerate(gallery_mcc_paths)),
                total=len(gallery_mcc_paths),
            )
        )
    if gallery_pose_paths is not None:
        with Pool(n_proc) as tp:
            gallery_mccs = list(
                tqdm.tqdm(
                    tp.imap(rectify_minutiae_only, zip(gallery_mccs, gallery_pose_paths)), total=len(gallery_mcc_paths)
                )
            )
    gallery_mccs = np.concatenate(gallery_mccs, axis=0)
    gallery_ids = gallery_mccs[:, :2].astype(np.uint32)
    gallery_minus = gallery_mccs[:, 2:5].astype(np.int16)
    gallery_mccs = gallery_mccs[:, 5:].astype(np.uint8)
    gallery_N = len(gallery_mcc_paths)
    gallery_minus[:, 2] = gallery_minus[:, 2] % 360

    # query
    with Pool(n_proc) as tp:
        query_mccs = list(
            tqdm.tqdm(
                tp.imap(functools.partial(flatting_mcc, params=bmc_params, is_pack=True), enumerate(query_mcc_paths)),
                total=len(query_mcc_paths),
            )
        )
    if query_pose_paths is not None:
        with Pool(n_proc) as tp:
            query_mccs = list(
                tqdm.tqdm(tp.imap(rectify_minutiae_only, zip(query_mccs, query_pose_paths)), total=len(query_mcc_paths))
            )
    query_mccs = np.concatenate(query_mccs, axis=0)
    query_ids = query_mccs[:, :2].astype(np.uint32)
    query_minus = query_mccs[:, 2:5].astype(np.int16)
    query_mccs = query_mccs[:, 5:].astype(np.uint8)
    query_N = len(query_mcc_paths)
    query_minus[:, 2] = query_minus[:, 2] % 360

    index_params = MCCParameter(delta_xy=delta_xy, delta_a=delta_a, hash_path=hash_path)
    scores = _mcc_indexing._mcc_indexing(
        gallery_N, gallery_ids, gallery_minus, gallery_mccs, query_N, query_ids, query_minus, query_mccs, index_params
    )
    return scores.reshape(query_N, gallery_N)


def transform_points(pts, trans=np.zeros(2), theta=0, trans2=np.zeros(2)):
    # theta has been saved as negative direction in pose file
    theta_rad = np.deg2rad(theta)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])
    new_pts = np.zeros_like(pts)
    new_pts[:, :2] = np.dot(pts[:, :2] - trans[None], R.T) + trans2
    new_pts[:, 2:] = pts[:, 2:]
    new_pts[:, 2] = (pts[:, 2] + theta) % 360

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


def rectify_minutiae_only(input):
    mcc_des, pose_path = input
    try:
        if len(mcc_des) == 0:
            return mcc_des

        try:
            pose = np.loadtxt(pose_path, delimiter=",")
        except:
            pose = np.loadtxt(pose_path)
        mcc_des[:, 2:5] = transform_points(mcc_des[:, 2:5], trans=pose[:2], theta=pose[2])
        return mcc_des
    except:
        return mcc_des


def flatting_mcc(input, params, is_pack=False):
    ii, mcc_path = input
    try:
        mcc_des = fp_mcc.load_mcc_feature(mcc_path)
        indices = np.where(mcc_des["flag"] > 0)[0]
        n_minu = len(indices)
        if n_minu == 0:
            return np.zeros((0, 5 + params.n_bits)) if not is_pack else np.zeros((0, 5 + np.ceil([params.n_bits / 8])))

        minutiae = mcc_des["mnt"][indices, :3]
        if is_pack:
            vector = np.stack([np.packbits(x) for x in mcc_des["cm"][indices].astype(bool)], axis=0)
        else:
            vector = mcc_des["cm"][indices]
        return np.concatenate((np.ones((len(indices), 1)) * ii, np.arange(n_minu)[:, None], minutiae, vector), axis=1)
    except Exception as ex:
        print(ex)
        return np.zeros((0, 5 + params.n_bits)) if not is_pack else np.zeros((0, 5 + np.ceil(params.n_bits / 8).astype(int)))


def construct_hash_table(input, n_minpc=2):
    keys, key_vals = input
    mask = keys.sum(axis=1) >= n_minpc
    unique_key_vals = np.unique(key_vals)
    mask_array = sparse.csr_matrix([(key_vals * mask) == k for k in unique_key_vals])

    return {k: ii for ii, k in enumerate(unique_key_vals)}, mask_array


def create_indexing_table(mcc_names, save_path, funcs, n_proc=8):
    params = fp_mcc.MCCParameters(is_binary=True)

    with Pool(n_proc) as tp:
        mcc_array = list(
            tqdm.tqdm(tp.imap(functools.partial(flatting_mcc, params=params), enumerate(mcc_names)), total=len(mcc_names))
        )
    mcc_array = np.concatenate(mcc_array, axis=0)
    minu_indices = mcc_array[:, :2]
    minutiae = mcc_array[:, 2:5]
    mcc_array = mcc_array[:, 5:].astype(bool)

    bin2dec_mat = 2 ** np.arange(funcs.shape[1])[::-1].astype(params.key_dtype)

    keys = mcc_array[:, funcs - 1].transpose(1, 0, 2)
    key_vals = keys.dot(bin2dec_mat)
    with Pool(n_proc) as tp:
        hash_tables = list(
            tqdm.tqdm(
                tp.imap(functools.partial(construct_hash_table, n_minpc=params.n_minpc), zip(keys, key_vals)),
                total=params.n_func,
            )
        )

    # hash_tables = [{} for _ in range(params.n_func)]
    # for ff in range(params.n_func):
    #     keys = mcc_array[:, funcs[ff] - 1]
    #     mask = keys.sum(axis=1) >= params.n_minpc
    #     key_vals = keys.dot(bin2dec_mat)
    #     unique_key_vals = np.unique(key_vals)
    #     mask_array = sparse.csr_matrix([(key_vals * mask) == k for k in unique_key_vals])
    #     hash_tables[ff] = [{k: ii for ii, k in enumerate(unique_key_vals)}, mask_array]
    #     print(f"constructing hash table for {ff}-th function")

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


def minu_mcc_indexing(input, coef, params):
    gallery_table = glm.get_value("gallery_table")

    # end = time.time()
    # cur_minu, cur_key_vals = input
    cur_minu, cur_key_vals = input[:3], input[3:]

    counts = 0
    for f in range(len(gallery_table["hash_table"])):
        if cur_key_vals[f] in gallery_table["hash_table"][f][0]:
            counts += gallery_table["hash_table"][f][1][gallery_table["hash_table"][f][0][cur_key_vals[f]]].A[0] * 1
    mask = counts > 0
    if not isinstance(mask, np.ndarray) or mask.sum() == 0:
        return np.zeros(gallery_table["gallery_size"])

    match_minu = gallery_table["minu"][mask]
    loc_diff = np.linalg.norm(match_minu[:, :2] - cur_minu[None, :2], axis=1)
    rot_diff = (match_minu[:, 2] - cur_minu[2] + 180) % 360 - 180
    diff_mask = (loc_diff < params.delta_xy) & (np.abs(rot_diff) < params.delta_theta)
    mask[mask] = diff_mask
    if diff_mask.sum() == 0:
        return np.zeros(gallery_table["gallery_size"])

    counts = counts[mask]
    gallery_idx = gallery_table["minu_idx"][mask, 0]
    scores = counts[:, None] * (gallery_idx[:, None] == np.arange(gallery_table["gallery_size"])[None])
    scores = scores.max(axis=0) ** coef

    # print(time.time() - end)

    return scores


def apply_mcc_indexing(input, params):
    mcc_path, pose_path = input
    gallery_table = glm.get_value("gallery_table")

    # funcs = gallery_table["funcs"]
    # hash_table = gallery_table["hash_table"]
    # gallery_size = gallery_table["gallery_size"]
    # gallery_minu_idx = gallery_table["minu_idx"].astype(int)
    # gallery_minu = gallery_table["minu"]
    # gallery_params = gallery_table["param"]

    coef = gallery_table["param"].p / gallery_table["param"].n_select

    try:
        mcc_des = fp_mcc.load_mcc_feature(mcc_path)
        indices = np.where(mcc_des["flag"] > 0)[0]
        n_minu = len(indices)
        if n_minu == 0:
            return np.zeros(gallery_table["gallery_size"])

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
        return np.zeros(gallery_table["gallery_size"])

    q_minu = query_mcc[:, :3]
    q_mcc = query_mcc[:, 3:].astype(bool)

    bin2dec_mat = 2 ** np.arange(gallery_table["funcs"].shape[1])[::-1].astype(gallery_table["param"].key_dtype)
    q_keys = q_mcc[:, gallery_table["funcs"] - 1]
    q_key_vals = q_keys.dot(bin2dec_mat)

    arr = np.concatenate((q_minu, q_key_vals), axis=1)
    scores = np.apply_along_axis(
        minu_mcc_indexing,
        axis=1,
        arr=arr,
        coef=coef,
        params=params,
    )
    scores = scores.sum(axis=0)

    # scores = np.zeros(gallery_table["gallery_size"])
    # for cur_ii in range(len(q_minu)):
    #     cur_minu = q_minu[cur_ii]
    #     cur_key_vals = q_key_vals[cur_ii]

    #     counts = 0
    #     for f in range(len(gallery_table["hash_table"])):
    #         if cur_key_vals[f] in gallery_table["hash_table"][f][0]:
    #             counts += gallery_table["hash_table"][f][1][gallery_table["hash_table"][f][0][cur_key_vals[f]]].A[0] * 1
    #     mask = counts > 0
    #     if not isinstance(mask, np.ndarray) or mask.sum() == 0:
    #         continue

    #     match_minu = gallery_table["minu"][mask]
    #     loc_diff = np.linalg.norm(match_minu[:, :2] - cur_minu[None, :2], axis=1)
    #     rot_diff = (match_minu[:, 2] - cur_minu[2] + 180) % 360 - 180
    #     diff_mask = (loc_diff < params.delta_xy) & (np.abs(rot_diff) < params.delta_theta)
    #     mask[mask] = diff_mask
    #     if diff_mask.sum() == 0:
    #         continue

    #     counts = counts[mask]
    #     gallery_idx = gallery_table["minu_idx"][mask, 0]
    #     tmp = counts[:, None] * (gallery_idx[:, None] == np.arange(gallery_table["gallery_size"])[None])
    #     scores += tmp.max(axis=0) ** coef

    scores = scores / len(q_minu) / (gallery_table["param"].n_func ** coef)
    return scores
