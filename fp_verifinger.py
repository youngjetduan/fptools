"""
This file (fp_verifinger.py) is designed for:
    functions for verifinger api
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
from ctypes import cdll
import subprocess
import warnings
import yaml
import socket

neu_lst = yaml.safe_load(open(osp.join(osp.dirname(osp.abspath(__file__)), "neu_dir.yaml"), "r"))
neu_dir = neu_lst[socket.gethostname()]

cdll.LoadLibrary(osp.join(neu_dir, "boost", "lib", "libboost_python37.so"))
cdll.LoadLibrary(osp.join(neu_dir, "boost", "lib", "libboost_numpy37.so"))
sys.path.append(osp.join(neu_dir, "Verifinger"))

# from _verifinger import _verifinger, _fingerprint_matching_single, _quality_score, _minutia_extraction
import _verifinger

sys.path.append(osp.dirname(osp.abspath(__file__)))
from uni_io import mkdir


def save_pair_file(fname, score, pairs):
    mkdir(osp.dirname(fname))
    with open(fname, "w") as fp:
        fp.write("# score, num_pairs, pairs\n")
        fp.write(f"{score:.3f}\n")
        fp.write(f"{len(pairs)}\n")
        for c_pair in pairs:
            fp.write(f"{int(c_pair[0])} {int(c_pair[1])}\n")


def load_pair_file(fname):
    warnings.filterwarnings("error")
    try:
        score = np.loadtxt(fname, skiprows=1, max_rows=1)
        pairs = np.loadtxt(fname, skiprows=3, dtype=int)
        return score.item(), pairs
    except UserWarning:
        return 0, np.array([])


def load_neu_dat(fname):
    cores = []
    deltas = []
    mnts = []
    with open(fname, "rb") as fp:
        nums = np.fromfile(fp, np.int32, 3)
        for _ in range(nums[0]):
            pt = np.fromfile(fp, np.int16, 2)
            theta = np.fromfile(fp, np.int32, 1)
            cores.append([pt[0], pt[1], theta[0], 1])
        for _ in range(nums[1]):
            pt = np.fromfile(fp, np.int16, 2)
            theta = np.fromfile(fp, np.int32, 3)
            deltas.append([pt[0], pt[1], theta[0], theta[1], theta[2], 2])
        for _ in range(nums[2]):
            pt = np.fromfile(fp, np.int16, 2)
            mtype = np.fromfile(fp, np.int32, 1)
            theta = np.fromfile(fp, np.ubyte, 4).astype(np.float32) * 180 / 128
            mnts.append([pt[0], pt[1], theta[0], mtype[0]])

    cores = np.array(cores)
    deltas = np.array(deltas)
    mnts = np.array(mnts)
    return cores, deltas, mnts


def pts_normalization(arr):
    if len(arr):
        arr = arr[None] if arr.ndim == 1 else arr
    return arr


def load_minutiae_complete(fname, return_header=False):
    num_core = np.loadtxt(fname, skiprows=2, max_rows=1).astype(int)
    num_delta = np.loadtxt(fname, skiprows=3, max_rows=1).astype(int)
    num_minu = np.loadtxt(fname, skiprows=4, max_rows=1).astype(int)

    if num_core:
        core_arr = np.loadtxt(fname, skiprows=5, max_rows=num_core)
        core_arr = pts_normalization(core_arr)
    else:
        core_arr = np.zeros((0, 4))
    if num_delta:
        delta_arr = np.loadtxt(fname, skiprows=5 + num_core, max_rows=num_delta)
        delta_arr = pts_normalization(delta_arr)
    else:
        delta_arr = np.zeros((0, 6))
    if num_minu:
        mnt_arr = np.loadtxt(fname, skiprows=5 + num_core + num_delta)
        mnt_arr = pts_normalization(mnt_arr)
    else:
        mnt_arr = np.zeros((0, 4))

    if return_header:
        header = np.loadtxt(fname, max_rows=2).astype(int)
        return core_arr, delta_arr, mnt_arr, header
    else:
        return core_arr, delta_arr, mnt_arr


def load_minutiae(fname, return_header=False):
    """load minutiae file

    Parameters:
        [None]
    Returns:
        mnt_array[, img_size(width, height)]
    """
    try:
        num_sp = np.loadtxt(fname, skiprows=2, max_rows=2)
        mnt_arr = np.loadtxt(fname, skiprows=5 + num_sp.sum().astype(int))

        mnt_arr = pts_normalization(mnt_arr)

        if return_header:
            header = np.loadtxt(fname, max_rows=2).astype(int)
            return mnt_arr, header
        else:
            return mnt_arr
    except:
        if return_header:
            return None, None
        else:
            return None


def load_singular(fname, return_header=False):
    num_core = np.loadtxt(fname, skiprows=2, max_rows=1).astype(int)
    num_delta = np.loadtxt(fname, skiprows=3, max_rows=1).astype(int)
    core_arr = np.loadtxt(fname, skiprows=5, max_rows=num_core)
    delta_arr = np.loadtxt(fname, skiprows=5 + num_core, max_rows=num_delta)

    core_arr = pts_normalization(core_arr)
    delta_arr = pts_normalization(delta_arr)

    if return_header:
        header = np.loadtxt(fname, max_rows=2).astype(int)
        return core_arr, delta_arr, header
    else:
        return core_arr, delta_arr


def save_minutiae(fname, img_size, core_arr=None, delta_arr=None, kps_arr=None):
    """save minutiae and singular points to '.mnt' file

    Parameters:
        img_shpae: [width, height]
    Returns:
        [None]
    """
    if not osp.isdir(osp.dirname(fname)):
        os.makedirs(osp.dirname(fname))

    core_arr = [] if core_arr is None else core_arr
    delta_arr = [] if delta_arr is None else delta_arr
    kps_arr = [] if kps_arr is None else kps_arr
    with open(fname, "w") as fp:
        fp.write(f"{img_size[0]}\n{img_size[1]}\n")
        fp.write(f"{len(core_arr)}\n{len(delta_arr)}\n{len(kps_arr)}\n")
        for c_core in core_arr:
            fp.write(" ".join([f"{int(x)}" for x in c_core]))
            fp.write("\n")
        for c_delta in delta_arr:
            fp.write(" ".join([f"{int(x)}" for x in c_delta]))
            fp.write("\n")
        for c_mnt in kps_arr:
            fp.write(" ".join([f"{int(x)}" for x in c_mnt]))
            fp.write("\n")


class Verifinger(_verifinger._verifinger):
    def __init__(self):
        super(Verifinger, self).__init__()
        self._initialize_license()

    def fingerprint_matching_single(self, search_dir, search_name, gallery_dir, gallery_name):
        """Note that, the number of minutiae in 'gallery_name' is higher than 'search_name'

        Parameters:
            [None]
        Returns:
            score [, pairs]
        """
        return self._fingerprint_matching_single(search_dir, search_name, gallery_dir, gallery_name)

    def fingerprint_matching_batch(self, search_paths, gallery_paths, thread_num=8):
        """Note that, the number of minutiae in 'gallery_name' is higher than 'search_name'
        DO NOT USE THIS FUNCTION!

        Parameters:
            [None]
        Returns:
            scores: N_search, N_gallery
        """
        return self._fingerprint_matching_batch(search_paths, gallery_paths, thread_num)

    def quality_score(self, img_dir, img_name, img_format="png"):
        """calculate fingerprint quality score, [0, 100], -1 means quality score can not be calculated

        Parameters:
            [None]
        Returns:
            quality_score
        """
        score = self._quality_score(img_dir, img_name, img_format)
        score = score if score <= 100 else -1
        return score

    def minutia_extraction(self, img_dir, img_name, feat_dir, feat_name="", img_format="png", mnt_format="ISO"):
        """set feat_name as "" if you prefer it has the same name as img_name
        The Minutia File format:
        img_width
        img_height
        num_of_core
        num_of_delta
        num_of_minutia
        [ core_point_detail ] * num_of_core
        [ delta_point_detail ] * num_of_delta
        [ minutia_point_detail ] * num_of_minutia

        Parameters:
            [None]
        Returns:
            quality_score
        """
        if len(feat_name) == 0:
            feat_name = img_name
        score = self._minutia_extraction(img_dir, img_name, feat_dir, feat_name, img_format, mnt_format)
        score = score if score <= 100 else -1
        return score

    def binary_extraction(self, img_dir, img_name, bin_dir, bin_name="", img_format="png"):
        """set bin_name as "" if you prefer it has the same name as img_name

        Parameters:
            [None]
        Returns:
            [None]
        """
        if len(bin_name) == 0:
            bin_name = img_name
        return self._binary_extraction(img_dir, img_name, bin_dir, bin_name, img_format)

    def skeleton_extraction(self, img_dir, img_name, skl_dir, skl_name="", img_format="png"):
        """set skl_name as "" if you prefer it has the same name as img_name

        Parameters:
            [None]
        Returns:
            [None]
        """
        if len(skl_name) == 0:
            skl_name = img_name
        return self._skeleton_extraction(img_dir, img_name, skl_dir, skl_name, img_format)

    def __del__(self):
        self._exit()


def fingerprint_matching_single(search_path, gallery_path):
    """Note that, the number of minutiae in 'gallery_name' is higher than 'search_name'

    Parameters:
        [None]
    Returns:
        score [, pairs]
    """
    return _verifinger._fingerprint_matching_single(
        osp.dirname(search_path), osp.basename(search_path), osp.dirname(gallery_path), osp.basename(gallery_path)
    )


def quality_score(img_path, img_format="png"):
    """calculate fingerprint quality score, [0, 100], -1 means quality score can not be calculated

    Parameters:
        [None]
    Returns:
        quality_score
    """
    score = _verifinger._quality_score(osp.dirname(img_path), osp.basename(img_path), img_format)
    score = score if score <= 100 else -1
    return score


def minutia_extraction(img_path, feat_path, img_format="png", mnt_format="ISO"):
    """set feat_name as "" if you prefer it has the same name as img_name
    The Minutia File format:
    img_width
    img_height
    num_of_core
    num_of_delta
    num_of_minutia
    [ core_point_detail ] * num_of_core
    [ delta_point_detail ] * num_of_delta
    [ minutia_point_detail ] * num_of_minutia

    Parameters:
        [None]
    Returns:
        quality_score
    """
    score = _verifinger._minutia_extraction(
        osp.dirname(img_path),
        osp.basename(img_path),
        osp.dirname(feat_path),
        osp.basename(feat_path),
        img_format,
        mnt_format,
    )
    score = score if score <= 100 else -1
    return score


def binary_extraction(img_path, bin_path, img_format="png"):
    """set bin_name as "" if you prefer it has the same name as img_name

    Parameters:
        [None]
    Returns:
        [None]
    """
    return _verifinger._binary_extraction(
        osp.dirname(img_path), osp.basename(img_path), osp.dirname(bin_path), osp.basename(bin_path), img_format
    )


def skeleton_extraction(img_path, skl_path, img_format="png"):
    """set skl_name as "" if you prefer it has the same name as img_name

    Parameters:
        [None]
    Returns:
        [None]
    """
    return _verifinger._skeleton_extraction(
        osp.dirname(img_path), osp.basename(img_path), osp.dirname(skl_path), osp.basename(skl_path), img_format
    )


def fingerprint_matching_single_minuonly(
    query_size, query_mnts, gallery_size, gallery_mnts, query_resolution=500, gallery_resolution=500
):
    """use minutiae only for fingerprint matching

    Parameters:
        [None]
    Returns:
        score [, pairs]
    """
    query_size = np.array(query_size) if not isinstance(query_size, np.ndarray) else query_size
    gallery_size = np.array(gallery_size) if not isinstance(gallery_size, np.ndarray) else gallery_size
    return _verifinger._fingerprint_matching_single_only(
        query_size.astype(np.int32),
        query_mnts.astype(np.int32),
        gallery_size.astype(np.int32),
        gallery_mnts.astype(np.int32),
        int(query_resolution),
        int(gallery_resolution),
    )


def minutia_extraction_test(img_path, feat_path, img_format="png", mnt_format="ISO"):
    """set feat_name as "" if you prefer it has the same name as img_name
    The Minutia File format:
    img_width
    img_height
    num_of_core
    num_of_delta
    num_of_minutia
    [ core_point_detail ] * num_of_core
    [ delta_point_detail ] * num_of_delta
    [ minutia_point_detail ] * num_of_minutia

    Parameters:
        [None]
    Returns:
        quality_score
    """
    score = _verifinger._minutia_extraction_test(
        osp.dirname(img_path),
        osp.basename(img_path),
        osp.dirname(feat_path),
        osp.basename(feat_path),
        img_format,
        mnt_format,
    )
    score = score if score <= 100 else -1
    return score
