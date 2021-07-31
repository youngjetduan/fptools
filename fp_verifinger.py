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

neu_dir = "/mnt/data5/fptools/Verifinger"

cdll.LoadLibrary(osp.join(neu_dir, "boost", "lib", "libboost_python37.so"))
cdll.LoadLibrary(osp.join(neu_dir, "boost", "lib", "libboost_numpy37.so"))
sys.path.append(osp.join(neu_dir, "Verifinger"))

from _verifinger import _verifinger


def load_minutiae(fname, return_hearder=False):
    sp = np.loadtxt(fname, skiprows=2, max_rows=2)
    kps = np.loadtxt(fname, skiprows=5 + sp.sum().astype(int))
    if return_hearder:
        header = np.loadtxt(fname, max_rows=2).astype(np.int)
        return kps, header
    else:
        return kps


class Verifinger(_verifinger):
    def __init__(self):
        super(Verifinger, self).__init__()
        self._initialize_license()

    def fingerprint_matching_single(self, search_dir, search_name, gallery_dir, gallery_name):
        """ 
        
        Parameters:
            [None]
        Returns:
            score [, pairs]
        """
        return self._fingerprint_matching_single(search_dir, search_name, gallery_dir, gallery_name)

    def fingerprint_matching_batch(self, search_paths, gallery_paths, thread_num=8):
        """ 
        
        Parameters:
            [None]
        Returns:
            scores: N_search, N_gallery
        """
        return self._fingerprint_matching_batch(search_paths, gallery_paths, thread_num)

    def minutia_extraction(self, img_dir, img_name, feat_dir, feat_name="", img_format="png", mnt_format="ISO"):
        """ set feat_name as "" if you prefer it has the same name as img_name
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
        """ set bin_name as "" if you prefer it has the same name as img_name
        
        Parameters:
            [None]
        Returns:
            [None]
        """
        if len(bin_name) == 0:
            bin_name = img_name
        return self._binary_extraction(img_dir, img_name, bin_dir, bin_name, img_format)

    def skeleton_extraction(self, img_dir, img_name, skl_dir, skl_name="", img_format="png"):
        """ set skl_name as "" if you prefer it has the same name as img_name
        
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
