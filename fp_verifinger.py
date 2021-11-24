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

# server 27
# neu_dir = "/mnt/data5/fptools/Verifinger"
# server 33
# neu_dir = "/mnt/data1/dyj"
neu_dir = np.loadtxt(osp.join(osp.dirname(osp.abspath(__file__)), "neu_dir.txt"), str).tolist()

cdll.LoadLibrary(osp.join(neu_dir, "boost", "lib", "libboost_python37.so"))
cdll.LoadLibrary(osp.join(neu_dir, "boost", "lib", "libboost_numpy37.so"))
sys.path.append(osp.join(neu_dir, "Verifinger"))

from _verifinger import _verifinger


def save_pair_file(fname, score, pairs):
    with open(fname, "w") as fp:
        fp.write("# score, num_pairs, pairs\n")
        fp.write(f"{score:.3f}\n")
        fp.write(f"{len(pairs)}\n")
        for c_pair in pairs:
            fp.write(f"{c_pair[0]} {c_pair[1]}\n")


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
    core_arr = np.loadtxt(fname, skiprows=5, max_rows=num_core)
    delta_arr = np.loadtxt(fname, skiprows=5 + num_core, max_rows=num_delta)
    mnt_arr = np.loadtxt(fname, skiprows=5 + num_core + num_delta)

    core_arr = pts_normalization(core_arr)
    delta_arr = pts_normalization(delta_arr)
    mnt_arr = pts_normalization(mnt_arr)

    if return_header:
        header = np.loadtxt(fname, max_rows=2).astype(int)
        return core_arr, delta_arr, mnt_arr, header
    else:
        return core_arr, delta_arr, mnt_arr


def load_minutiae(fname, return_header=False):
    num_sp = np.loadtxt(fname, skiprows=2, max_rows=2)
    mnt_arr = np.loadtxt(fname, skiprows=5 + num_sp.sum().astype(int))

    mnt_arr = pts_normalization(mnt_arr)

    if return_header:
        header = np.loadtxt(fname, max_rows=2).astype(int)
        return mnt_arr, header
    else:
        return mnt_arr


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
            fp.write(" ".join([f"{x:.2f}" for x in c_core]))
            fp.write("\n")
        for c_delta in delta_arr:
            fp.write(" ".join([f"{x:.2f}" for x in c_delta]))
            fp.write("\n")
        for c_mnt in kps_arr:
            fp.write(" ".join([f"{x:.2f}" for x in c_mnt]))
            fp.write("\n")


class Verifinger(_verifinger):
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

        Parameters:
            [None]
        Returns:
            scores: N_search, N_gallery
        """
        return self._fingerprint_matching_batch(search_paths, gallery_paths, thread_num)

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
            [None]
        """
        if len(feat_name) == 0:
            feat_name = img_name
        return self._minutia_extraction(img_dir, img_name, feat_dir, feat_name, img_format, mnt_format)

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
