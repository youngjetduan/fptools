"""
This file (uni_io.py) is designed for:
    functions for file/io
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
import shutil
import imageio
import scipy.io as sio


def mkdir(path):
    if not osp.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError as err:
            pass


def re_mkdir(path):
    if osp.isdir(path):
        shutil.rmtree(path)

    try:
        os.makedirs(path)
    except FileExistsError as err:
        pass


def imwrite(path, img):
    mkdir(osp.dirname(path))
    imageio.imwrite(path, np.rint(img).astype(np.uint8))


def matwrite(path, mat):
    mkdir(osp.dirname(path))
    sio.savemat(path, mat, do_compression=True)


if __name__ == "__main__":
    prefix = ""
