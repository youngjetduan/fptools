"""
Descripttion: demo for phase matching (Zhe Cui)
version:
Author: Xiongjun Guan
Date: 2021-08-21 11:49:21
LastEditors: Xiongjun Guan
LastEditTime: 2021-09-08 10:18:39
"""

# from base.uni_io import imwrite
import os
import os.path as osp
import scipy.io as scio
import argparse
import sys
import cv2
import time

sys.path.append(osp.dirname(osp.abspath(__file__)))
from BasePhase import PhaseRegistration
from BaseFeature import PadImage, ExtractFeature
from Base import ShowRegisteredImg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ftitle1", "-f1", default=None, type=str, help="file 1's title"
    )
    parser.add_argument(
        "--ftitle2", "-f2", default=None, type=str, help="file 2's title"
    )

    parser.add_argument(
        "--need_pad_image",
        "-p",
        default=True,
        type=bool,
        help="do feature extraction and save results in feature path",
    )
    parser.add_argument(
        "--need_feature_extraction",
        "-f",
        default=True,
        type=bool,
        help="do feature extraction and save results in feature path",
    )
    parser.add_argument(
        "--need_time",
        "-t",
        default=True,
        type=bool,
        help="Time consuming for calculating phase registration (regardless of the time for pre extracting features)",
    )
    args = parser.parse_args()

    if args.ftitle1 is None:
        args.ftitle1 = "l0_0"

    if args.ftitle2 is None:
        args.ftitle2 = "l0_1"

    ftitle1 = args.ftitle1
    ftitle2 = args.ftitle2
    need_feature_extraction = args.need_feature_extraction
    need_pad_image = args.need_pad_image
    need_time = args.need_time

    img_path = "../data/img/"
    feature_path = "../data/feature/"
    tmp_path = "../data/tmp/"
    res_path = "../data/res/"
    ext = "png"

    tmp_title = ftitle1 + "_with_" + ftitle2
    feature_tmp_path = osp.join(tmp_path, tmp_title, "pad_feature")
    init_tmp_path = osp.join(tmp_path, tmp_title, "init")

    init_prefix = "init_"
    phase_prefix = "phase_"

    # --------------- feature extraction : begin --------------- #
    # verifinger
    from fp_verifinger import Verifinger

    verifinger = Verifinger()

    if need_feature_extraction:
        if need_pad_image:
            PadImage(img_path, feature_tmp_path, ftitle1, ftitle2, ext)
            feature_path = feature_tmp_path
            ExtractFeature(feature_path, feature_path, ftitle1, verifinger, ext)
            ExtractFeature(feature_path, feature_path, ftitle2, verifinger, ext)
        else:
            ExtractFeature(img_path, feature_path, ftitle1, verifinger, ext)
            ExtractFeature(img_path, feature_path, ftitle2, verifinger, ext)
    if need_pad_image:
        feature_path = feature_tmp_path
    # --------------- feature extraction : end --------------- #

    if need_time:
        start_time = time.time()

    # --------------- phase match : begin --------------- #
    img_phase, dx, dy = PhaseRegistration(
        feature_path,
        init_tmp_path,
        ftitle1,
        ftitle2,
        verifinger,
        ext,
        init_prefix,
        phase_prefix,
    )
    if img_phase is None:
        return
    if not osp.exists(res_path):
        os.makedirs(res_path)
    img_phase2_title = phase_prefix + ftitle1 + "_" + ftitle2
    cv2.imwrite(osp.join(res_path, img_phase2_title + "." + ext), img_phase)
    scio.savemat(
        osp.join(res_path, img_phase2_title + ".mat"),
        {
            "ftitle1": ftitle1,
            "ftitle2": ftitle2,
            "dx": dx,
            "dy": dy,
        },
    )
    # --------------- phase match : end--------------- #

    if need_time:
        end_time = time.time()
        print("time: ", end_time - start_time)

    # --------------- compare effects : begin --------------- #
    bimg1 = cv2.imread(
        osp.join(feature_path, "b" + ftitle1 + "." + ext), cv2.IMREAD_GRAYSCALE
    )
    img_deformed_title = init_prefix + ftitle1 + "_" + ftitle2
    bimg2_init = cv2.imread(
        osp.join(init_tmp_path, "b" + img_deformed_title + "." + ext),
        cv2.IMREAD_GRAYSCALE,
    )
    res_title = "phase_" + ftitle1 + "_" + ftitle2
    verifinger.binary_extraction(res_path, res_title, res_path, "b" + res_title, ext)
    bimg2_res = cv2.imread(
        osp.join(res_path, "b" + res_title + "." + ext), cv2.IMREAD_GRAYSCALE
    )

    im = ShowRegisteredImg(bimg1, bimg2_init)
    cv2.imwrite(
        osp.join(res_path, "ShowRegistration_init.png"),
        cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
    )
    im = ShowRegisteredImg(bimg1, bimg2_res)
    cv2.imwrite(
        osp.join(res_path, "ShowRegistration_res.png"),
        cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
    )
    # --------------- compare effects : end--------------- #