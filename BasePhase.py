"""
Descripttion: Python version converted from the 'base' code (Matlab version)
version:
Author: Xiongjun Guan
Date: 2021-08-18 16:09:59
LastEditors: Xiongjun Guan
LastEditTime: 2021-08-21 20:52:46
"""
from numpy.core.fromnumeric import nonzero
import skimage
from skimage import measure
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata
import scipy.io as scio
import os
import os.path as osp
import numpy as np
import cv2
import copy
import sys
from scipy import stats
import time

sys.path.append(osp.dirname(osp.abspath(__file__)))
from uni_tps import opencv_tps, tps_apply_transform, tps_module_numpy
from fp_mcc import MCC
from BaseDirPed import FillHoleDir, ComputePeriod_Dir
from Base import (
    FillHole,
    DetectSP,
    GetBranchCuts,
    FindCurve2,
    NormalizeMinuDir,
    RidgeFilterComplex,
)
from fptools.fp_verifinger import load_minutiae
from fptools.fp_sift import regist_sift



ROUND_EPS = 0.00001  # for the problem of np.round() in '0.5'


def SIFTRegistration(feature_path, ftitle1, ftitle2, ext="png"):
    img1 = cv2.imread(osp.join(feature_path, ftitle1 +
                      "." + ext), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(osp.join(feature_path, ftitle2 +
                      "." + ext), cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape

    img_deformed, pts1, pts2 = regist_sift(img1, img2)

    return img_deformed


def PhaseRegistration(
    feature_path,
    tmp_path,
    ftitle1,
    ftitle2,
    tool,
    ext="png",
    init_prefix="init_",
    phase_prefix="phase_",
    initFunc="VeriFinger",
    threshNum=8,
    RANSAC_th = None,
):
    """_summary_

    Args:
        feature_path (_type_): _description_
        tmp_path (_type_): _description_
        ftitle1 (_type_): _description_
        ftitle2 (_type_): _description_
        tool (_type_): VeriFinger
        ext (_type_): _description_. Defaults to "png".
        init_prefix (str, optional): Prefix name of init data. Defaults to "init_".
        phase_prefix (str, optional): Prefix name of phase data. Defaults to "phase_".
        initFunc (str, optional): Matching method. Defaults to "VeriFinger".
        threshNum (int, optional): Minimum number of matching points if 'initFunc' type is 'VeriFinger'. Defaults to 8.

    Returns:
        _type_: _description_
    """

    # -------------------------------------------------------- #
    # ----------------------- tps init ----------------------- #
    # -------------------------------------------------------- #
    img1 = cv2.imread(osp.join(feature_path, ftitle1 +
                      "." + ext), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(osp.join(feature_path, ftitle2 +
                      "." + ext), cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape

    # t1 = time.time()
    if initFunc is "VeriFinger":
        MINU1 = load_minutiae(osp.join(feature_path, "mf" + ftitle1 + ".mnt"))
        MINU2 = load_minutiae(osp.join(feature_path, "mf" + ftitle2 + ".mnt"))
        score, init_minu_pairs = tool.fingerprint_matching_single(
            feature_path, "mf" + ftitle1, feature_path, "mf" + ftitle2
        )
        if RANSAC_th is not None:
            H, mask = cv2.estimateAffinePartial2D(MINU1[init_minu_pairs[:, 0], 0:2], MINU2[init_minu_pairs[:, 1], 0:2], method=cv2.RANSAC, ransacReprojThreshold=20.0)
            mask = mask.reshape((-1,))
            init_minu_pairs = init_minu_pairs.take(np.where(mask==1)[0],0)
        if init_minu_pairs.shape[0] < threshNum:
            # print(
            #     "[%s] - [%s] Error: Insufficient number of matching points !\n"
            #     % (ftitle1, ftitle2)
            # )
            # return None, -1, -1
            raise ValueError('Insufficient number of matching points !')

        p_init = np.hstack(
            (MINU1[init_minu_pairs[:, 0], 0:2],
             MINU2[init_minu_pairs[:, 1], 0:2])
        )

        if not osp.exists(tmp_path):
            os.makedirs(tmp_path)

        
    elif initFunc is "SIFT":
        img_deformed, pts1, pts2 = regist_sift(img1, img2)
        MINU1 = np.ones((pts1.shape[0], 4))
        MINU2 = np.ones((pts2.shape[0], 4))
        MINU1[:, 0:2] = pts1
        MINU2[:, 0:2] = pts2
        init_minu_pairs = np.array(
            [np.arange(1, MINU1.shape[0]), np.arange(1, MINU1.shape[0])]).T
        p_init = np.hstack(
            (MINU1[init_minu_pairs[:, 0], 0:2],
             MINU2[init_minu_pairs[:, 1], 0:2])
        )

    else:
        # print(
        #     "[%s] - [%s] Error: %s No such init function type !\n"
        #     % (ftitle1, ftitle2,initFunc)
        # )
        # return None, -1, -1
        raise ValueError('No such init function type !')
    
    # t2 = time.time()
    # print("init registration: {}s".format(t2-t1))

    img_deformed = opencv_tps(
        img2,
        MINU2[init_minu_pairs[:, 1], 0:2],
        MINU1[init_minu_pairs[:, 0], 0:2],
        mode=1,
        border_value=255,
    )
    img_deformed_title = init_prefix + ftitle1 + "_" + ftitle2
    if not osp.exists(tmp_path):
        os.makedirs(tmp_path)
    cv2.imwrite(osp.join(tmp_path, img_deformed_title + "." + ext), img_deformed)

    # -------------------------------------------------------------------------------------------------- #
    # ----------------------- extract feature of init result (tps wrapped image) ----------------------- #
    # -------------------------------------------------------------------------------------------------- #
    tool.binary_extraction(
        tmp_path, img_deformed_title, tmp_path, "b" + img_deformed_title, ext
    )
    # tool.skeleton_extraction(
    #     tmp_path, img_deformed_title, tmp_path, "t" + img_deformed_title, ext
    # )
    ExtractPhaseFeature(
        tmp_path,
        tmp_path,
        img_deformed_title,
        prefix=phase_prefix,
        ext="png",
        save=False,
    )
    # t3 = time.time()
    # print("feature extraction 2: {}s".format(t3-t2))


    # ----------------------------------------------------- #
    # ----------------------- phase ----------------------- #
    # ----------------------------------------------------- #
    # load ftitle1 info
    phase_1_mat = scio.loadmat(
        osp.join(feature_path, phase_prefix + ftitle1 + ".mat"))
    PHASE1ORIGIN = phase_1_mat["m_phase"]
    UNWRAPPEDDIR1 = phase_1_mat["m_unwrappedDir"]
    PED1 = phase_1_mat["m_ped"]
    PED1[PED1 == 0] = np.inf
    MASK1 = phase_1_mat["m_mask0"]
    MASK1[MASK1 != 1] = 0

    # load init result info
    phase_init_mat = scio.loadmat(
        osp.join(tmp_path, phase_prefix + img_deformed_title + ".mat")
    )
    PHASE2ORIGIN = phase_init_mat["m_phase"]
    UNWRAPPEDDIR2 = phase_init_mat["m_unwrappedDir"]
    MASK2 = phase_init_mat["m_mask0"]
    MASK2[MASK2 != 1] = 0

    anchorPts = MINU1[init_minu_pairs[:, 0], :]
    MASK = np.zeros_like(MASK1)
    MASK[(MASK1 == 1) & (MASK2 == 1)] = 1

    # deal with phase reversion due to reverse in direction
    diffDir = UNWRAPPEDDIR1 - UNWRAPPEDDIR2
    diffDir[MASK == 0] = 0
    diffDir = NormalizeMinuDir(diffDir)
    dirMask = np.abs(diffDir) < 90
    PHASE2ORIGIN[dirMask == 0] = -PHASE2ORIGIN[dirMask == 0]
    phasediffRaw = PHASE2ORIGIN - PHASE1ORIGIN
    phasediffRaw[MASK == 0] = 0

    # t1 = time.time()
    ret, phaseDiff, PHASE_MASK, reliability = PhaseUnwrap(phasediffRaw, MASK)
    # t2 = time.time()
    # print("phase unwrap: ",t2-t1)

    if ret:
        # find start point of unwrapping
        minuType = np.hstack(
            (
                MINU1[init_minu_pairs[:, 0], 3].reshape((-1, 1)),
                MINU2[init_minu_pairs[:, 1], 3].reshape((-1, 1)),
            )
        )

        # t1 = time.time()
        startPts, phaseDiff, vis = find_unwrap_start_point2(
            anchorPts, minuType, PHASE_MASK, phasediffRaw, phaseDiff
        )
        # t2 = time.time()
        # print("find_unwrap_start_point2: ",t2-t1)

        for i in range(0, len(vis)):
            if vis[i] == 0:
                PHASE_MASK[PHASE_MASK == i + 1] = 0

        # sample on valid mask region
        blockSize = 20
        # compute distortion
        px, py, xx, yy = ComputeDistortion(
            phaseDiff, PED1, UNWRAPPEDDIR1, blockSize, PHASE_MASK
        )

        if xx.shape[0] > 2:
            xx = xx.reshape((-1, 1))
            yy = yy.reshape((-1, 1))
            p_phase = np.hstack(
                (
                    xx + px[yy, xx].reshape((-1, 1)),
                    yy + py[yy, xx].reshape((-1, 1)),
                    xx,
                    yy,
                )
            )
            suc = True
        
        else:
            # print(
            #     "[%s] - [%s] Error: No valid point after distortion field smooth \n"
            #     % (ftitle1, ftitle2)
            # )
            # suc = False
            # return None, -1, -1
            raise ValueError('No valid point after distortion field smooth !')
        
    else:
        # suc = False
        # return None, -1, -1
        raise ValueError('Dense Registration error !')
    
    # t4 = time.time()
    # print("phase registration: {}s".format(t4-t3))

    # ---------------------------------------------------------- #
    # ----------------------- distortion ----------------------- #
    # ---------------------------------------------------------- #
    # t1 = time.time()
    h, w = img1.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    origin_xy = np.hstack((x, y))

    # init_tps_matrix = tps_module_numpy(p_init[:, 2:], p_init[:, 0:2])
    # init_xy = tps_apply_transform(origin_xy, p_init[:, 2:], init_tps_matrix)
    # phase_tps_matrix = tps_module_numpy(p_phase[:, 2:], p_phase[:, 0:2])
    # phase_xy = tps_apply_transform(init_xy, p_phase[:, 2:], phase_tps_matrix)
    phase_tps_matrix = tps_module_numpy(p_phase[:, 2:], p_phase[:, 0:2])
    phase_xy = tps_apply_transform(origin_xy, p_phase[:, 2:], phase_tps_matrix)

    dx = (phase_xy[:, 0] - origin_xy[:, 0]).reshape((-1, 1))
    dy = (phase_xy[:, 1] - origin_xy[:, 1]).reshape((-1, 1))

    # img2_tmp = griddata(
    #     np.hstack((x + dx, y + dy)),
    #     img2.reshape((-1, 1)),
    #     np.hstack((x, y)),
    #     method="nearest",
    # ).reshape((h, w))
    img2_tmp = griddata(
        np.hstack((x + dx, y + dy)),
        img_deformed.reshape((-1, 1)),
        np.hstack((x, y)),
        method="nearest",
        fill_value=255
    ).reshape((h, w))
    # t2 = time.time()
    # print("distortion: ",t2-t1)

    # t5 = time.time()
    # print("grid warp: {}s".format(t5-t4))

    return img2_tmp, dx, dy

def PhaseRegistrationWithoutInit(
            bimg_dir,
            minu_dir,
            phase_dir,
            ftitle1,
            ftitle2,
            minu_title,
            phase_prefix,
            ):

    ExtractPhaseFeature(
        bimg_dir,
        phase_dir,
        ftitle1,
        prefix=phase_prefix,
        ext="png",
        save=False,
        bimg_prefix='',
    )
    ExtractPhaseFeature(
        bimg_dir,
        phase_dir,
        ftitle2,
        prefix=phase_prefix,
        ext="png",
        save=False,
        bimg_prefix='',
    )

    phase_1_mat = scio.loadmat(
        osp.join(phase_dir, phase_prefix + ftitle1 + ".mat"))
    PHASE1ORIGIN = phase_1_mat["m_phase"]
    UNWRAPPEDDIR1 = phase_1_mat["m_unwrappedDir"]
    PED1 = phase_1_mat["m_ped"]
    PED1[PED1 == 0] = np.inf
    MASK1 = phase_1_mat["m_mask0"]
    MASK1[MASK1 != 1] = 0

    phase_2_mat = scio.loadmat(
        osp.join(phase_dir, phase_prefix + ftitle2 + ".mat"))
    PHASE2ORIGIN = phase_2_mat["m_phase"]
    UNWRAPPEDDIR2 = phase_2_mat["m_unwrappedDir"]
    MASK2 = phase_2_mat["m_mask0"]
    MASK2[MASK2 != 1] = 0

    minu_data = scio.loadmat(osp.join(minu_dir,minu_title+'.mat'))
    minu1 = minu_data['minu1']
    minu2 = minu_data['minu2']
    anchorPts = minu2
    MASK = np.zeros_like(MASK1)
    MASK[(MASK1 == 1) & (MASK2 == 1)] = 1

    # deal with phase reversion due to reverse in direction
    diffDir = UNWRAPPEDDIR1 - UNWRAPPEDDIR2
    diffDir[MASK == 0] = 0
    diffDir = NormalizeMinuDir(diffDir)
    dirMask = np.abs(diffDir) < 90
    PHASE2ORIGIN[dirMask == 0] = -PHASE2ORIGIN[dirMask == 0]
    phasediffRaw = PHASE2ORIGIN - PHASE1ORIGIN
    phasediffRaw[MASK == 0] = 0

    # t1 = time.time()
    ret, phaseDiff, PHASE_MASK, reliability = PhaseUnwrap(phasediffRaw, MASK)
    # t2 = time.time()
    # print("phase unwrap: ",t2-t1)

    if ret:
        # find start point of unwrapping
        minuType = np.hstack(
            (
                minu1[:, 3].reshape((-1, 1)),
                minu2[:, 3].reshape((-1, 1)),
            )
        )

        # t1 = time.time()
        startPts, phaseDiff, vis = find_unwrap_start_point2(
            anchorPts, minuType, PHASE_MASK, phasediffRaw, phaseDiff
        )
        # t2 = time.time()
        # print("find_unwrap_start_point2: ",t2-t1)

        for i in range(0, len(vis)):
            if vis[i] == 0:
                PHASE_MASK[PHASE_MASK == i + 1] = 0

        # sample on valid mask region
        blockSize = 20
        # compute distortion
        px, py, xx, yy = ComputeDistortion(
            phaseDiff, PED1, UNWRAPPEDDIR1, blockSize, PHASE_MASK
        )

        if xx.shape[0] > 2:
            xx = xx.reshape((-1, 1))
            yy = yy.reshape((-1, 1))
            p_phase = np.hstack(
                (
                    xx + px[yy, xx].reshape((-1, 1)),
                    yy + py[yy, xx].reshape((-1, 1)),
                    xx,
                    yy,
                )
            )
            suc = True
        
        else:
            # print(
            #     "[%s] - [%s] Error: No valid point after distortion field smooth \n"
            #     % (ftitle1, ftitle2)
            # )
            # suc = False
            # return None, -1, -1
            raise ValueError('No valid point after distortion field smooth !')
        
    else:
        # suc = False
        # return None, -1, -1
        raise ValueError('Dense Registration error !')
    

    # ---------------------------------------------------------- #
    # ----------------------- distortion ----------------------- #
    # ---------------------------------------------------------- #
    img1 = cv2.imread(osp.join(bimg_dir,ftitle1+'.png'),0)
    h, w = img1.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    origin_xy = np.hstack((x, y))

    # init_tps_matrix = tps_module_numpy(p_init[:, 2:], p_init[:, 0:2])
    # init_xy = tps_apply_transform(origin_xy, p_init[:, 2:], init_tps_matrix)
    # phase_tps_matrix = tps_module_numpy(p_phase[:, 2:], p_phase[:, 0:2])
    # phase_xy = tps_apply_transform(init_xy, p_phase[:, 2:], phase_tps_matrix)
    phase_tps_matrix = tps_module_numpy(p_phase[:, 2:], p_phase[:, 0:2])
    phase_xy = tps_apply_transform(origin_xy, p_phase[:, 2:], phase_tps_matrix)

    dx = (phase_xy[:, 0] - origin_xy[:, 0]).reshape((-1, 1))
    dy = (phase_xy[:, 1] - origin_xy[:, 1]).reshape((-1, 1))

    
    # img2_tmp = griddata(
    #     np.hstack((x + dx, y + dy)),
    #     img_deformed.reshape((-1, 1)),
    #     np.hstack((x, y)),
    #     method="nearest",
    #     fill_value=255
    # ).reshape((h, w))

    return dx, dy


def ExtractPhaseFeature(
    feature_path, res_path, ftitle, prefix="phase", ext="png", save=True,bimg_prefix="b"
):
    """Extract phase information and save
    Args:
        feature_path ([type]): Binary/skeleton image path
        res_path ([type]): Save path
        ftitle ([type]): Fingerprint' title
        prefix (str, optional): Prefix of result. Defaults to "phase".
        ext (str, optional): Save format of images. Defaults to 'png'.
    """
    # timg = cv2.imread(
    #     osp.join(feature_path, "t" + ftitle + "." + ext), cv2.IMREAD_GRAYSCALE
    # )
    bimg = cv2.imread(
        osp.join(feature_path, bimg_prefix + ftitle + "." + ext), cv2.IMREAD_GRAYSCALE
    )
    # m_dir, m_ped, m_mask0 = ComputePeriod_Dir(timg)
    # t1 = time.time()
    m_dir, m_ped, m_mask0 = ComputePeriod_Dir(bimg)
    # t2 = time.time()
    # print("feature extraction - compute ped_dir: {}s".format(t2-t1))
    # t1 = time.time()
    m_phase, m_unwrappedDir, m_mask1, m_xn, m_yn, m_sps = ComputePhase(
        bimg, m_dir, m_ped, m_mask0
    )
    # t2 = time.time()
    # print("feature extraction - compute phase: {}s".format(t2-t1))
    scio.savemat(
        osp.join(res_path, prefix + ftitle + ".mat"),
        {
            "m_phase": m_phase,
            "m_unwrappedDir": m_unwrappedDir,
            "m_mask0": m_mask0,
            "m_mask1": m_mask1,
            "m_dir": m_dir,
            "m_ped": m_ped,
            "m_xn": m_xn,
            "m_yn": m_yn,
            "m_sps": m_sps,
        },
    )
    # print("Extracting phase from " + ftitle + " successfully!")
    return m_phase, m_unwrappedDir, m_mask0, m_mask1, m_dir, m_ped, m_xn, m_yn, m_sps


def ComputePhase(img, DIR, PED, MASK):
    """compute phase
    Args:
        img ([type]): [description]
        DIR ([type]): [description]
        PED ([type]): [description]
        MASK ([type]): [description]
    Returns:
        PHASE: Phase
        DIR2: Direction
        MASK2: Mask
        xn,yn: Branch cuts
        sps:  [x y type direction] of singular points
    """
    if np.max(img) > 1:
        I = (img / 255).astype(np.double)
    else:
        I = img.astype(np.double)

    DIR[MASK == 0] = 91
    PED[MASK == 0] = 0
    DIR2, MASK2, xn, yn, sps = UnwrapOrientationField(DIR)
    MASK2[MASK == 0] = 0

    DIR2 = NormalizeMinuDir(-DIR2)

    orient = DIR2 * np.pi / 180
    freq = np.divide(1, (PED + np.spacing(1)))
    freq[PED == 0] = 0  # Only enhance hole
    freq[freq > 0] = np.round(ROUND_EPS + freq[freq > 0] * 100) / 100

    k = 0.3
    Ir, Ii = RidgeFilterComplex(I, orient, freq, k, k)

    Z = Ir + 1j * Ii
    Intensity = np.abs(Z)
    PHASE = np.angle(Z)
    PHASE[Intensity < 0.00001] = 2 * np.pi  # invalid


    return PHASE, DIR2, MASK2, xn, yn, sps


def UnwrapOrientationField(D1):
    """Unwrap orientation field
    Args:
        D1 ([type]): direction image
    Returns:
        D2 : Unwrapped direction image
        mask : segmentation mask (0 background, 1 foreground, 2 branch cuts)
        xnn, ynn : branch cuts
        sps : [x y type direction] of singular points
    """
    copy_D1 = copy.deepcopy(D1)
    D1 = 91 * np.ones((D1.shape[0] + 2, D1.shape[1] + 2))
    D1[1:-1, 1:-1] = copy_D1

    # constants
    dx = np.array([0, 1, 0, -1])
    dy = np.array([-1, 0, 1, 0])

    h, w = D1.shape

    #  Fill hole (not connected to border)
    MASK = FillHole((D1 != 91).astype(np.int))
    D1 = FillHoleDir(D1, MASK)

    # detect singular points
    sps = DetectSP(D1, (D1 != 91).astype(np.int), 2, 1)

    # Find branch cuts
    xnn, ynn = GetBranchCuts(D1, sps)

    if len(xnn) > 0:
        xn = copy.deepcopy(xnn[-1]).astype(np.int)
        yn = copy.deepcopy(ynn[-1]).astype(np.int)
    else:
        xn = []
        yn = []

    MASK = (D1 != 91)
    distmap = distance_transform_edt(MASK.astype(np.int))
    yy, xx = np.unravel_index(np.argsort(-distmap.flatten()), distmap.shape)

    FLAG = np.zeros(
        D1.shape
    )  # 0 background, 1 foreground, 2 branch cuts, 3 in queue and unwrapped
    FLAG[D1 != 91] = 1
    FLAG[yn, xn] = 2
    mask = copy.deepcopy(FLAG[1:-1, 1:-1])

    D2 = copy.deepcopy(D1)
    for ii in range(0, len(yy)):
        y = yy[ii]
        x = xx[ii]
        if FLAG[y, x] != 1:
            continue
        # Keep orientation unchanged
        D2[y, x] = D1[y, x]
        FLAG[y, x] = 3  # Change flag
        q = np.zeros((h * w, 2))
        q[0, :] = np.array([x, y])  # push into queue
        qid = 0
        qlen = 0
        while qid <= qlen:
            x1 = int(q[qid, 0])
            y1 = int(q[qid, 1])
            qid = qid + 1
            for k in range(0, 4):
                x2 = int(x1 + dx[k])
                y2 = int(y1 + dy[k])
                if FLAG[y2, x2] == 1:
                    # Upwrap, change flag and push into queue
                    dif = D2[y1, x1] - D1[y2, x2]
                    D2[y2, x2] = D1[y2, x2] + \
                        np.round(ROUND_EPS + dif / 180) * 180
                    FLAG[y2, x2] = 3
                    qlen = qlen + 1
                    q[qlen, 0:2] = np.array([x2, y2])

    if len(xnn) > 0:
        # Unwrap branch cuts
        # to do: if a branch cut is thick, iterative unwrap is needed.
        for i in range(0, len(xnn) - 1):
            cxn = xnn[i]
            cyn = ynn[i]
            start = int(np.round(ROUND_EPS + len(cxn) / 2))
            for j in range(start, len(cxn)):
                dif = D2[cyn[j], cxn[j]] - D2[cyn[j - 1], cxn[j - 1]]
                if np.abs(np.mod(dif, 360) - 180) < 90:
                    D2[cyn[j], cxn[j]] = D2[cyn[j],
                                            cxn[j]] - np.sign(dif) * 180

            for j in range(start - 2, -1, -1):
                dif = D2[cyn[j], cxn[j]] - D2[cyn[j + 1], cxn[j + 1]]
                if np.abs(np.mod(dif, 360) - 180) < 90:
                    D2[cyn[j], cxn[j]] = D2[cyn[j],
                                            cxn[j]] - np.sign(dif) * 180

        # for each branch cut, there should be another one along with the
        # original one, which has a 180 degree difference, together, they form the
        # border of branch cut
        xnn = xnn[0:-1]
        ynn = ynn[0:-1]
        ll = len(xnn)
        fail_ind = []
        for i in range(0, ll):
            cx = xnn[i]
            cy = ynn[i]

            # extend the front and end of the line by 1 pixel
            ecx = np.hstack((cx[0] + cx[0] - cx[1], cx,
                            cx[-1] + cx[-1] - cx[-2]))
            ecy = np.hstack((cy[0] + cy[0] - cy[1], cy,
                            cy[-1] + cy[-1] - cy[-2]))

            im = np.zeros_like(D2, dtype=np.int)
            im[cy, cx] = 1
            im = skimage.morphology.dilation(im, np.ones((3, 3)))
            arrind = np.nonzero(im)
            ind1 = np.ravel_multi_index(arrind, im.shape)
            tmpmask = (ecx >= 1) & (ecx <= w) & (ecy >= 1) & (ecy <= h)
            ind2 = np.setdiff1d(
                ind1, np.ravel_multi_index(
                    (ecy[tmpmask], ecx[tmpmask]), D2.shape)
            )
            im = np.zeros_like(D2, dtype=np.int)
            ind2_y, ind2_x = np.unravel_index(ind2, D2.shape)
            im[ind2_y, ind2_x] = 1
            tmpmask = np.ones_like(D2, dtype=np.int)
            tmpmask[1:-1, 1:-1] = 0
            im[tmpmask] = 0
            I, cnt = measure.label(im, connectivity=1, return_num=True)

            if cnt == 1:
                fail_ind.append(i)
                continue
            ind1 = I == 1
            ind2 = I == 2
            I = np.ones_like(D2)
            I[ind1] = 0
            ind1 = FindCurve2(I, cx[0], cy[0])
            I = np.ones_like(D2)
            I[ind2] = 0
            ind2 = FindCurve2(I, cx[0], cy[0])

            y = ind1[1, int(
                np.round(ROUND_EPS + (1.1 + ind1.shape[1]) / 2)) - 1]
            x = ind1[0, int(
                np.round(ROUND_EPS + (1.1 + ind1.shape[1]) / 2)) - 1]
            flag = 0
            dx2 = np.array([0, 1, 0, -1, 1, 1, -1, -1])
            dy2 = np.array([-1, 0, 1, 0, 1, -1, 1, -1])
            for k in range(0, 8):
                xx = dx2[k] + x
                yy = dy2[k] + y
                if xx < 0 or xx >= w or yy < 0 or yy >= h:
                    continue
                if D2[yy, xx] == 91:
                    continue
                if np.abs(np.mod(D2[y, x] - D2[yy, xx], 360) - 180) < 50:
                    flag = 1
                    break

            xnn = np.hstack((xnn, np.empty((1,), dtype=object)))
            ynn = np.hstack((ynn, np.empty((1,), dtype=object)))
            if flag == 1:
                xnn[-1] = ind1[0, :]
                ynn[-1] = ind1[1, :]
            else:
                xnn[-1] = ind2[0, :]
                ynn[-1] = ind2[1, :]

        # xnn[fail_ind] = np.array([])
        # ynn[fail_ind] = np.array([])
        # xnn = np.hstack((xnn, np.empty((1,), dtype=object)))
        # ynn = np.hstack((ynn, np.empty((1,), dtype=object)))
        # xnn[-1] = np.hstack(xnn[0:-1])
        # ynn[-1] = np.hstack(ynn[0:-1])

        if xnn.shape[0] > 0:
            xnn_list = []
            ynn_list = []
            for i in range(xnn.shape[0]):
                if i not in fail_ind and xnn[i].shape[0] > 0:
                    xnn_list.append(xnn[i])
                    ynn_list.append(ynn[i])
            if len(xnn_list) > 0:
                xnn_list.append([])
                ynn_list.append([])
                xnn = np.array(xnn_list, dtype=object)
                ynn = np.array(ynn_list, dtype=object)
                xnn[-1] = np.hstack(xnn[0:-1])
                ynn[-1] = np.hstack(ynn[0:-1])
            else:
                xnn = np.array([])
                ynn = np.array([])

        D2[D1 == 91] = 1000  # invalid value
        D2 = D2[1:-1, 1:-1]
        for i in range(0, len(xnn)):
            xnn[i] = xnn[i] - 1
            ynn[i] = ynn[i] - 1

    else:
        D2[D1 == 91] = 1000  # invalid value
        D2 = D2[1:-1, 1:-1]

    return D2, mask, xnn, ynn, sps


def PhaseUnwrap(phase, mask, threshold=200.0):
    """Herrez M A, Burton D R, Lalor M J, et al. Fast two-dimensional phase-unwrapping
       algorithm based on sorting by reliability following a noncontinuous path[J]. Applied Optics, 2002, 41(35): 7437-7444.
    Args:
        phase ([type]): [description]
        mask ([type]): [description]
        threshold (float, optional): [description]. Defaults to 200.0.
    Returns:
        ret : True for success, False for failed
        phaseNew2 : unwrapped phase
        phaseMaskNew: mask
        R: reliability value for every pixel in mask region
    """

    def diffPhase(phase1, phase2):
        res = phase1 - phase2
        if res > np.pi:
            res -= 2 * np.pi
        elif res <= -np.pi:
            res += 2 * np.pi
        return res

    h, w = mask.shape
    bmask = (mask > 0).astype(np.int8)

    # calculate reliability value for every pixel in mask region
    R = np.zeros_like(phase)
    yy, xx = np.nonzero(bmask)
    for i in range(0, len(yy)):
        if (
            (yy[i] + 1 >= h) or (yy[i] - 1 <
                                 0) or (xx[i] + 1 > w) or (xx[i] - 1 < 0)
        ):  # image border
            R[yy[i], xx[i]] = 0
        elif (
            np.sum(bmask[yy[i] - 1: yy[i] + 2, xx[i] - 1: xx[i] + 2]) != 9
        ):  # mask border
            R[yy[i], xx[i]] = 0
        else:
            H = diffPhase(phase[yy[i], xx[i] - 1], phase[yy[i], xx[i]]) - diffPhase(
                phase[yy[i], xx[i]], phase[yy[i], xx[i] + 1]
            )
            V = diffPhase(phase[yy[i] - 1, xx[i]], phase[yy[i], xx[i]]) - diffPhase(
                phase[yy[i], xx[i]], phase[yy[i] + 1, xx[i]]
            )
            D1 = diffPhase(
                phase[yy[i] - 1, xx[i] - 1], phase[yy[i], xx[i]]
            ) - diffPhase(phase[yy[i], xx[i]], phase[yy[i] + 1, xx[i] + 1])
            D2 = diffPhase(
                phase[yy[i] + 1, xx[i] - 1], phase[yy[i], xx[i]]
            ) - diffPhase(phase[yy[i], xx[i]], phase[yy[i] - 1, xx[i] + 1])
            tmp = H * H + V * V + D1 * D1 + D2 * D2
            if tmp == 0:
                R[yy[i], xx[i]] = np.inf
            else:
                R[yy[i], xx[i]] = 1.0 / tmp

    # calculate edge reliablity in mask region
    edgeR = np.zeros((len(yy) * 2, 5))
    cnt = 0
    for i in range(0, len(yy)):
        if yy[i] + 1 < h and bmask[yy[i] + 1, xx[i]]:
            edgeR[cnt, :] = np.array(
                [xx[i], yy[i], xx[i], yy[i] + 1,
                    R[yy[i], xx[i]] + R[yy[i] + 1, xx[i]]]
            )
            cnt += 1
        if xx[i] + 1 < w and bmask[yy[i], xx[i] + 1]:
            edgeR[cnt, :] = np.array(
                [xx[i], yy[i], xx[i] + 1, yy[i],
                    R[yy[i], xx[i]] + R[yy[i], xx[i] + 1]]
            )
            cnt += 1

    edgeR = edgeR[0:cnt, :]

    idx_tmp = np.argsort(edgeR[:, 4])
    edgeR = np.flipud(edgeR[idx_tmp, :])

    # unwrap phase
    curLabel = 1
    label = np.zeros_like(phase)
    labelCnt = np.zeros((edgeR.shape[0],))
    labelIdx_y = np.empty((edgeR.shape[0],), dtype=object)
    labelIdx_y.fill(np.array([]))
    labelIdx_x = np.empty((edgeR.shape[0],), dtype=object)
    labelIdx_x.fill(np.array([]))
    phaseNew = np.zeros_like(phase)
    for edgeCnt in range(0, edgeR.shape[0]):
        if edgeR[edgeCnt, 4] <= threshold:
            break
        pt1x = int(edgeR[edgeCnt, 0])
        pt1y = int(edgeR[edgeCnt, 1])
        pt2x = int(edgeR[edgeCnt, 2])
        pt2y = int(edgeR[edgeCnt, 3])
        label1 = int(label[pt1y, pt1x])
        label2 = int(label[pt2y, pt2x])
        if label1 > 0 and label2 == label1:
            continue
        elif label1 == 0 and label2 == 0:
            phaseNew[pt1y, pt1x] = phase[pt1y, pt1x]
            diff = phase[pt2y, pt2x] - phase[pt1y, pt1x]
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff <= -np.pi:
                diff += 2 * np.pi
            phaseNew[pt2y, pt2x] = phaseNew[pt1y, pt1x] + diff
            label[pt1y, pt1x] = curLabel
            label[pt2y, pt2x] = curLabel
            labelCnt[curLabel] += 2
            labelIdx_y[curLabel] = np.array([pt1y, pt2y])
            labelIdx_x[curLabel] = np.array([pt1x, pt2x])
            curLabel += 1
        elif label1 > 0 and label2 > 0:
            if labelCnt[label1] > labelCnt[label2]:  # merge label2 to label1
                diff = phase[pt2y, pt2x] - phase[pt1y, pt1x]
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff <= -np.pi:
                    diff += 2 * np.pi
                diff -= phaseNew[pt2y, pt2x] - phaseNew[pt1y, pt1x]
                phaseNew[labelIdx_y[label2], labelIdx_x[label2]] += diff
                label[labelIdx_y[label2], labelIdx_x[label2]] = label1
                labelCnt[label1] += labelCnt[label2]
                labelIdx_y[label1] = np.hstack(
                    (labelIdx_y[label1], labelIdx_y[label2]))
                labelIdx_x[label1] = np.hstack(
                    (labelIdx_x[label1], labelIdx_x[label2]))
                labelCnt[label2] = 0
                labelIdx_y[label2] = np.array([])
                labelIdx_x[label2] = np.array([])
            else:
                diff = phase[pt1y, pt1x] - phase[pt2y, pt2x]
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff <= -np.pi:
                    diff += 2 * np.pi
                diff -= phaseNew[pt1y, pt1x] - phaseNew[pt2y, pt2x]
                phaseNew[labelIdx_y[label1], labelIdx_x[label1]] += diff
                label[labelIdx_y[label1], labelIdx_x[label1]] = label2
                labelCnt[label2] += labelCnt[label1]
                labelIdx_y[label2] = np.hstack(
                    (labelIdx_y[label2], labelIdx_y[label1]))
                labelIdx_x[label2] = np.hstack(
                    (labelIdx_x[label2], labelIdx_x[label1]))
                labelCnt[label1] = 0
                labelIdx_y[label1] = np.array([])
                labelIdx_x[label1] = np.array([])
        elif label1 > 0:
            diff = phase[pt2y, pt2x] - phase[pt1y, pt1x]
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff <= -np.pi:
                diff += 2 * np.pi

            phaseNew[pt2y, pt2x] = phaseNew[pt1y, pt1x] + diff
            label[pt2y, pt2x] = label1
            labelCnt[label1] += 1
            labelIdx_y[label1] = np.hstack((labelIdx_y[label1], pt2y))
            labelIdx_x[label1] = np.hstack((labelIdx_x[label1], pt2x))
        else:
            diff = phase[pt1y, pt1x] - phase[pt2y, pt2x]
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff <= -np.pi:
                diff += 2 * np.pi

            phaseNew[pt1y, pt1x] = phaseNew[pt2y, pt2x] + diff
            label[pt1y, pt1x] = label2
            labelCnt[label2] += 1
            labelIdx_y[label2] = np.hstack((labelIdx_y[label2], pt1y))
            labelIdx_x[label2] = np.hstack((labelIdx_x[label2], pt1x))

    # unwrapping failed
    if curLabel == 1:
        return False, 0, 0, 0

    # sort area of components in unwrapped phase
    sortIdx2 = np.argsort(-labelCnt)
    areaNum = 10
    phaseNew2 = np.zeros_like(phase)
    phaseMaskNew = np.zeros_like(phase)

    for j in range(0, min(areaNum, curLabel - 1)):
        t = sortIdx2[j]
        if labelIdx_x[t].shape[0] > 0:
            phaseNew2[labelIdx_y[t], labelIdx_x[t]] = phaseNew[
                labelIdx_y[t], labelIdx_x[t]
            ]
            phaseMaskNew[labelIdx_y[t], labelIdx_x[t]] = j + 1

    return True, phaseNew2, phaseMaskNew, R


def find_unwrap_start_point2(anchorPts, minuType, PHASE_MASK, phasediffRaw, phaseDiff):
    h, w = PHASE_MASK.shape
    region_size_thresh = 2000
    startPts = np.zeros((0, 2))
    areaNum = 10
    sz = 7
    border = 2
    unwrapPointThresh = 1.5
    stdThresh = 0.5
    vis = np.zeros((areaNum,))
    borderMask = np.ones((2 * sz + 1, 2 * sz + 1))
    borderMask[border:-border, border:-border] = 0
    borderY, borderX = np.nonzero(borderMask)
    CX = np.array([sz + 1, sz + 1, 1])

    tmp = phasediffRaw > np.pi
    phasediffRaw[tmp] = phasediffRaw[tmp] - 2 * np.pi
    tmp = phasediffRaw <= -np.pi
    phasediffRaw[tmp] = phasediffRaw[tmp] + 2 * np.pi
    recAnchorPts = np.zeros(
        (anchorPts.shape[0], 5)
    )  # 1. label, 2.fit score, 3.offset, 4. x, 5. y

    for i in range(0, anchorPts.shape[0]):
        if minuType[i, 0] != minuType[i, 1]:  # good start point should be the same type
            continue
        anchorPts = anchorPts.astype(np.int16)
        cx = anchorPts[i, 0]
        cy = anchorPts[i, 1]
        if cx + sz >= w or cx - sz < 0 or cy + sz >= h or cy - sz < 0:
            continue
        top = cy - sz
        down = cy + sz
        left = cx - sz
        right = cx + sz
        mask = PHASE_MASK[top: down + 1, left: right + 1]
        phasediffraw = phasediffRaw[top: down + 1, left: right + 1]
        phaseunwrap = phaseDiff[top: down + 1, left: right + 1]
        vmask = mask.flatten()
        vmask = vmask[vmask != 0]
        if vmask.shape[0] == 0:
            continue

        # find the labels of mask
        # if mask contain multiple labels belonging to separate components, find
        # the most freqent one in labels
        label = int(stats.mode(vmask)[0][0])

        if vis[label - 1] != 0:
            continue

        if len(np.nonzero(vmask == label)[0]) < 20:  # small valid region
            continue

        cur_mask = mask == label
        tmp = phaseunwrap[cur_mask]
        if np.max(tmp) - np.min(tmp) >= 2 * np.pi:  # big variation
            continue

        mask1 = mask[borderY, borderX] == label
        if np.sum(mask1) < 9:
            continue

        xx = borderX[mask1].reshape((-1, 1))
        yy = borderY[mask1].reshape((-1, 1))
        X = np.hstack((xx + 1, yy + 1, np.ones((xx.shape[0], 1))))
        z = phasediffraw[yy, xx]
        weight = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), z)
        zp = np.dot(X, weight)
        err = np.sum(np.power(zp - z, 2)) / len(zp)
        centerVal = np.dot(CX, weight)[0]
        if abs(centerVal) > unwrapPointThresh:
            continue

        recAnchorPts[i, 0] = label
        recAnchorPts[i, 1] = err
        minidx = np.argmin(np.abs(zp - z))
        sx = xx[minidx, 0]
        sy = yy[minidx, 0]
        soffset = phasediffraw[sy, sx] - phaseunwrap[sy, sx]
        recAnchorPts[i, 2] = soffset
        sx = left + sx
        sy = top + sy
        recAnchorPts[i, 3] = sx
        recAnchorPts[i, 4] = sy

    for i in range(1, areaNum + 1):
        tmpAnchorPts = recAnchorPts[recAnchorPts[:, 0] == i, :]
        if tmpAnchorPts.shape[0] > 0:
            tmpAnchorPts = tmpAnchorPts[tmpAnchorPts[:, 1] < 0.05, :]
            if tmpAnchorPts.shape[0] > 0:
                idx = np.argsort(tmpAnchorPts[:, 2])
                tidx = idx[int(np.round((1 + len(idx)) / 2 + ROUND_EPS)) - 1]
                vis[i - 1] = 1
                startPts = np.vstack(
                    (startPts, np.array(
                        [tmpAnchorPts[tidx, 3], tmpAnchorPts[tidx, 4]]))
                )
                ind = PHASE_MASK == i
                phaseDiff[ind] = phaseDiff[ind] + tmpAnchorPts[tidx, 2]

    cosPhasediffRaw = np.cos(phasediffRaw)
    mQuality = stdfilt(cosPhasediffRaw, 3)
    badArea = (mQuality > stdThresh).astype(np.int16)
    goodArea = 1 - cv2.dilate(badArea, np.ones((9, 9)))
    meanCosPhasediffRaw = cv2.boxFilter(
        cosPhasediffRaw, -1, (7, 7), borderType=cv2.BORDER_REPLICATE
    )

    phase_mask_cnt = np.zeros((10,))
    for iii in range(1, 11):
        phase_mask_cnt[iii - 1] = np.sum(PHASE_MASK == iii)

    for i_region in range(1, 11):
        if (
            vis[i_region - 1] == 0 and phase_mask_cnt[i_region -
                                                      1] > region_size_thresh
        ):  # ignore small areas
            tmpgoodArea = (PHASE_MASK == i_region) & goodArea
            curAreaVals = meanCosPhasediffRaw[tmpgoodArea]
            if curAreaVals.shape[0] == 0:
                continue
            ind = np.argmax(curAreaVals)
            yy, xx = np.nonzero(tmpgoodArea)
            if xx.shape[0] == 0:
                continue
            sx = xx[ind]
            sy = yy[ind]
            m_offset = phasediffRaw[sy, sx] - phaseDiff[sy, sx]
            ind = PHASE_MASK == i_region
            phaseDiff[ind] = phaseDiff[ind] + m_offset
            startPts = np.vstack((startPts, np.array([sx, sy])))
            vis[i_region] = 1

    return startPts, phaseDiff, vis


def stdfilt(arr, nhood=3):
    """MATLAB stdfilt
    Args:
        arr ([type]): 2D array
    Returns:
        J: arr after standard filter
    """
    h = np.ones((nhood, nhood))
    n = h.sum()
    n1 = n - 1
    c1 = cv2.filter2D(arr ** 2, -1, h / n1, borderType=cv2.BORDER_REFLECT)
    c2 = cv2.filter2D(
        arr, -1, h, borderType=cv2.BORDER_REFLECT) ** 2 / (n * n1)
    J = np.sqrt(np.maximum(c1 - c2, 0))
    J[np.isnan(J)] = 0
    return J


def ComputeDistortion(phaseDiff, PED1, UNWRAPPEDDIR1, blkSize, PHASE_MASK):
    """Calculate the pixel by pixel displacement based on direction and phase difference
    Args:
        phaseDiff ([type]): Phase difference
        PED1 ([type]): Direction
        UNWRAPPEDDIR1 ([type]): Unwrapped direction
        blkSize ([type]): Grid size
        PHASE_MASK ([type]): Phase mask
    Returns:
        px,py: pixel distortion
        xx2,yy2: grid points in ROI
    """
    h, w = phaseDiff.shape
    unwrappedGdt1 = UNWRAPPEDDIR1 + 90
    offsets = np.divide(
        phaseDiff, (np.spacing(1) + np.divide(2 * np.pi, PED1)))
    px = np.multiply(np.cos(unwrappedGdt1 * np.pi / 180), offsets)
    py = np.multiply(np.sin(unwrappedGdt1 * np.pi / 180), offsets)
    mask = PHASE_MASK != 0
    px[mask == 0] = np.nan
    py[mask == 0] = np.nan
    stdx = stdfilt(px)
    badAreax = (stdx > 1).astype(np.int16)
    badAreax = cv2.dilate(badAreax, np.ones((9, 9)))
    stdy = stdfilt(py)
    badAreay = (stdy > 1).astype(np.int16)
    badAreay = cv2.dilate(badAreay, np.ones((9, 9)))
    badArea = badAreax | badAreay
    mask[badArea == 1] = 0
    rnd = int(np.round(blkSize / 2 + ROUND_EPS)) - 1
    xx, yy = np.meshgrid(np.arange(rnd, w, blkSize),
                         np.arange(rnd, h, blkSize))
    xx = xx.flatten()
    yy = yy.flatten()
    xx2 = xx[mask[yy, xx] == 1]
    yy2 = yy[mask[yy, xx] == 1]

    return px, py, xx2, yy2
