"""
Descripttion: Python version converted from the 'base' code (Matlab version)
version: 
Author: Xiongjun Guan
Date: 2021-09-01 11:36:50
LastEditors: Xiongjun Guan
LastEditTime: 2021-09-01 11:48:07
"""

import os
import os.path as osp
import cv2
import shutil
import numpy as np
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))
from BasePhase import ExtractPhaseFeature


def ExtractFeature(img_path, feature_path, ftitle, tool, ext="png"):
    """Extract features (binary,skeleton, minutiae, phase)
       The prefixes of these features should be:
            fingerprint - None
            binary      - 'b'
            minutiae    - 'mf'
            skeleton    - 't'
            phase       - 'phase'

    Args:
        img_path ([type]): image path
        feature_path ([type]): save path
        ftitle ([type]): image's title
        tool ([type]): verifinger
        ext (str, optional): image's format. Defaults to "png".
    """
    if not osp.exists(feature_path):
        os.makedirs(feature_path)
    if not osp.exists(osp.join(feature_path, ftitle + "." + ext)):
        shutil.copy(
            osp.join(img_path, ftitle + "." + ext),
            osp.join(feature_path, ftitle + "." + ext),
        )

    tool.minutia_extraction(img_path, ftitle, feature_path, "mf" + ftitle, ext)
    tool.binary_extraction(img_path, ftitle, feature_path, "b" + ftitle, ext)
    tool.skeleton_extraction(img_path, ftitle, feature_path, "t" + ftitle, ext)
    ExtractPhaseFeature(feature_path, feature_path, ftitle, "phase_", ext, save=True)

    return


def PadImage(img_path, feature_path, ftitle1, ftitle2, ext="png"):
    """Make two fingerprints the same size and save them in feature_path

    Args:
        img_path ([type]): image path
        feature_path ([type]): save path
        ftitle1 ([type]): image1's title
        ftitle2 ([type]): image2's title
        ext (str, optional): image's format. Defaults to "png".
    """
    if not osp.exists(feature_path):
        os.makedirs(feature_path)

    im1 = cv2.imread(osp.join(img_path, ftitle1 + "." + ext), cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(osp.join(img_path, ftitle2 + "." + ext), cv2.IMREAD_GRAYSCALE)

    # pad size
    row = max(im1.shape[0], im2.shape[0])
    col = max(im1.shape[1], im2.shape[1])

    if im1.shape == im2.shape:
        shutil.copy(
            osp.join(img_path, ftitle1 + "." + ext),
            osp.join(feature_path, ftitle1 + "." + ext),
        )
        shutil.copy(
            osp.join(img_path, ftitle2 + "." + ext),
            osp.join(feature_path, ftitle2 + "." + ext),
        )
    else:
        im1_pad = np.pad(
            im1,
            ((0, row - im1.shape[0]), (0, col - im1.shape[1])),
            mode="constant",
            constant_values=255,
        )
        cv2.imwrite(osp.join(feature_path, ftitle1 + "." + ext), im1_pad)
        im2_pad = np.pad(
            im2,
            ((0, row - im2.shape[0]), (0, col - im2.shape[1])),
            mode="constant",
            constant_values=255,
        )
        cv2.imwrite(osp.join(feature_path, ftitle2 + "." + ext), im2_pad)

    return


def PadFeature(
    feature_path, feature_tmp_path, ftitle1, ftitle2, ext="png", need_pad=True
):
    """Make two fingerprints and corresponding features the same size and save them in feature_tmp_path

    Args:
        feature_path ([type]): origin features' path.The prefixes of these features should be:
            fingerprint - None
            binary      - 'b'
            minutiae    - 'mf'
            skeleton    - 't'
        feature_tmp_path ([type]): save path
        ftitle1 ([type]): fp1's name
        ftitle2 ([type]): fp2's name
        ext ([type]): image's format
        need_pad ([type]): copy files to tmp dir if it is False
    """
    if not osp.exists(feature_tmp_path):
        os.makedirs(feature_tmp_path)

    if need_pad is False:
        shutil.copy(
            osp.join(feature_path, "mf" + ftitle1 + ".mnt"),
            osp.join(feature_tmp_path, "mf" + ftitle1 + ".mnt"),
        )
        shutil.copy(
            osp.join(feature_path, "mf" + ftitle2 + ".mnt"),
            osp.join(feature_tmp_path, "mf" + ftitle2 + ".mnt"),
        )
        shutil.copy(
            osp.join(feature_path, ftitle1 + "." + ext),
            osp.join(feature_tmp_path, ftitle1 + "." + ext),
        )
        shutil.copy(
            osp.join(feature_path, ftitle2 + "." + ext),
            osp.join(feature_tmp_path, ftitle2 + "." + ext),
        )
        shutil.copy(
            osp.join(feature_path, "b" + ftitle1 + "." + ext),
            osp.join(feature_tmp_path, "b" + ftitle1 + "." + ext),
        )
        shutil.copy(
            osp.join(feature_path, "b" + ftitle2 + "." + ext),
            osp.join(feature_tmp_path, "b" + ftitle2 + "." + ext),
        )
        shutil.copy(
            osp.join(feature_path, "t" + ftitle1 + "." + ext),
            osp.join(feature_tmp_path, "t" + ftitle1 + "." + ext),
        )
        shutil.copy(
            osp.join(feature_path, "t" + ftitle2 + "." + ext),
            osp.join(feature_tmp_path, "t" + ftitle2 + "." + ext),
        )
        return

    im1 = cv2.imread(osp.join(feature_path, ftitle1 + "." + ext), cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(osp.join(feature_path, ftitle2 + "." + ext), cv2.IMREAD_GRAYSCALE)

    # pad size
    row = max(im1.shape[0], im2.shape[0])
    col = max(im1.shape[1], im2.shape[1])

    # image
    im1_pad = np.pad(
        im1,
        ((0, row - im1.shape[0]), (0, col - im1.shape[1])),
        mode="constant",
        constant_values=255,
    )
    cv2.imwrite(osp.join(feature_tmp_path, ftitle1 + "." + ext), im1_pad)
    im2_pad = np.pad(
        im2,
        ((0, row - im2.shape[0]), (0, col - im2.shape[1])),
        mode="constant",
        constant_values=255,
    )
    cv2.imwrite(osp.join(feature_tmp_path, ftitle2 + "." + ext), im2_pad)

    # skeleton
    tim1 = cv2.imread(
        osp.join(feature_path, "t" + ftitle1 + "." + ext), cv2.IMREAD_GRAYSCALE
    )
    tim1_pad = np.pad(
        tim1,
        ((0, row - im1.shape[0]), (0, col - im1.shape[1])),
        mode="constant",
        constant_values=255,
    )
    cv2.imwrite(osp.join(feature_tmp_path, "t" + ftitle1 + "." + ext), tim1_pad)
    tim2 = cv2.imread(
        osp.join(feature_path, "t" + ftitle2 + "." + ext), cv2.IMREAD_GRAYSCALE
    )
    tim2_pad = np.pad(
        tim2,
        ((0, row - im2.shape[0]), (0, col - im2.shape[1])),
        mode="constant",
        constant_values=255,
    )
    cv2.imwrite(osp.join(feature_tmp_path, "t" + ftitle2 + "." + ext), tim2_pad)

    # binary
    bim1 = cv2.imread(
        osp.join(feature_path, "b" + ftitle1 + "." + ext), cv2.IMREAD_GRAYSCALE
    )
    bim1_pad = np.pad(
        bim1,
        ((0, row - im1.shape[0]), (0, col - im1.shape[1])),
        mode="constant",
        constant_values=255,
    )
    cv2.imwrite(osp.join(feature_tmp_path, "b" + ftitle1 + "." + ext), bim1_pad)
    bim2 = cv2.imread(
        osp.join(feature_path, "b" + ftitle2 + "." + ext), cv2.IMREAD_GRAYSCALE
    )
    bim2_pad = np.pad(
        bim2,
        ((0, row - im2.shape[0]), (0, col - im2.shape[1])),
        mode="constant",
        constant_values=255,
    )
    cv2.imwrite(osp.join(feature_tmp_path, "b" + ftitle2 + "." + ext), bim2_pad)

    # minutiae
    shutil.copy(
        osp.join(feature_path, "mf" + ftitle1 + ".mnt"),
        osp.join(feature_tmp_path, "mf" + ftitle1 + ".mnt"),
    )
    shutil.copy(
        osp.join(feature_path, "mf" + ftitle2 + ".mnt"),
        osp.join(feature_tmp_path, "mf" + ftitle2 + ".mnt"),
    )

    return
