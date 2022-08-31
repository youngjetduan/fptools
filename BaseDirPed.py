"""
Descripttion: Python version converted from the 'base' code (Matlab version)
version: 
Author: Xiongjun Guan
Date: 2021-08-20 17:20:18
LastEditors: Xiongjun Guan
LastEditTime: 2021-08-22 12:15:22
"""

import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import math
import numpy as np
import copy
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))
from Base import MakeSameSize, MedianFilter, FillHole

ROUND_EPS = 0.00001  # for the problem of np.round() in '0.5'


def ComputePeriod_Dir(skel):
    """Calculate the period image, direction image and segmentation image of fingerprint

    Args:
        skel ([type]): skeleton image

    Returns:
        DIR: direction image
        PED: period image
        MASK: segmentation image
    """
    max_period = 20
    max_half_period = max_period / 2
    blksize = 8
    T = skel
    D = distance_transform_edt(T)
    D[D > max_half_period] = max_half_period
    MASK = ((distance_transform_edt(D < max_half_period)) > max_half_period).astype(int)

    DIR, _ = ComputeDir(D, 1, 21)
    DIR[MASK == 0] = 91
    blk_MASK = MASK[int(blksize / 2) :: blksize, int(blksize / 2) :: blksize]

    P = ComputePeriod(
        D.astype(np.double),
        DIR[int(blksize / 2) :: blksize, int(blksize / 2) :: blksize],
        blksize,
    )

    P[blk_MASK == 0] = 0

    P = MedianFilter(P, blk_MASK, 3)
    P = cv2.resize(P, (0, 0), fx=blksize, fy=blksize, interpolation=cv2.INTER_NEAREST)
    P = MakeSameSize(P, DIR.shape[0], DIR.shape[1], 0)
    P[MASK == 0] = 0
    P[P < 0] = 0
    MASK = P > 0
    DIR[MASK == 0] = 91

    PED = copy.deepcopy(P)
    MASK = ((DIR != 91) & (PED > 0)).astype(np.int)

    # find hole
    MASK = FillHole(MASK)

    # fill hole in Ridge Flow Map
    DIR[MASK == 0] = 91
    DIR = DIR.astype(np.double)
    DIR = FillHoleDir(DIR, MASK)

    # Fill hold in Ridge Wavelength Map
    PED[MASK == 0] = 0
    PED = PED.astype(np.double)
    PED = FillHolePed(PED, MASK)

    return DIR, PED, MASK


def ComputePeriod(I, DIR, BLOCK_SIZE):
    """ "Compute blockwise period image

    Args:
        I ([type]): grayscale fingerprint image
        DIR ([type]): blockwise direction image, [-90, 90]
        BLOCK_SIZE ([type]): block size

    Returns:
        P: blockwise period image
    """
    [h, w] = I.shape
    [bh, bw] = DIR.shape
    P = np.zeros((bh, bw))

    def ProjectNormal(m, n):
        H = 8
        L = 40
        invalid_gray = 1000

        cx = np.floor(n * BLOCK_SIZE + BLOCK_SIZE / 2 + 1)
        cy = np.floor(m * BLOCK_SIZE + BLOCK_SIZE / 2 + 1)
        xc = np.floor(cx - np.sin(np.deg2rad(DIR[m, n])) * np.arange(-L / 2, L / 2 + 1))
        yc = np.floor(cy - np.cos(np.deg2rad(DIR[m, n])) * np.arange(-L / 2, L / 2 + 1))

        #  project
        f = np.ones((1 + L)) * invalid_gray
        # valid_f = []
        for i in range(0, len(xc)):
            if xc[i] < 1 or yc[i] < 1 or xc[i] > w or yc[i] > h:
                continue
            xs = (
                np.floor(
                    xc[i]
                    + np.cos(np.deg2rad(DIR[m, n])) * np.arange(-H / 2, H / 2 + 1) * 3
                ).astype(np.int)
                - 1
            )
            ys = (
                np.floor(
                    yc[i]
                    - np.sin(np.deg2rad(DIR[m, n])) * np.arange(-H / 2, H / 2 + 1) * 3
                ).astype(np.int)
                - 1
            )
            idx = np.nonzero((xs >= 0) & (ys >= 0) & (xs <= w - 1) & (ys <= h - 1))
            if len(idx[0]) > 2:
                f[i] = np.mean(I[ys[idx], xs[idx]])
                # valid_f.append(f[i])

        sf1 = np.ones((len(f) - 6)) * invalid_gray
        for i in range(3, len(f) - 3):
            sf1[i - 3] = (
                f[i] * 0.2080
                + (f[i - 1] + f[i + 1]) * 0.1861
                + (f[i - 2] + f[i + 2]) * 0.1334
                + (f[i - 3] + f[i + 3]) * 0.0765
            )

        sf = np.ones((len(f) - 6,)) * invalid_gray
        for i in range(0, len(sf1)):
            if (
                sf1[i] == invalid_gray
                or (i > 0 and sf1[i - 1] == invalid_gray)
                or (i + 1 < len(sf1) and sf1[i + 1] == invalid_gray)
            ):
                sf[i] = invalid_gray
            else:
                sf[i] = sf1[i]

        bStart = 0
        iStatus = 0
        iFirstType = 0
        idx = []

        for i in range(1, len(sf) - 1):
            if sf[i] == invalid_gray:
                if bStart == 1:
                    break
                else:
                    continue
            if sf[i] > sf[i - 1] and sf[i] > sf[i + 1]:
                if iStatus == 0:
                    iFirstType = 1
                elif iStatus == 1:
                    mid_idx = np.floor((idx[-1] + i) / 2)
                    idx.append(mid_idx)
                iStatus = 1
                idx.append(i)
            elif sf[i] < sf[i - 1] and sf[i] < sf[i + 1]:
                if iStatus == 0:
                    iFirstType = -1
                elif iStatus == -1:
                    mid_idx = np.floor((idx[-1] + i) / 2)
                    idx.append(mid_idx)
                iStatus = -1
                idx.append(i)

        # fPeakMean = 0
        # fPeakVar = 0
        iPeakNum = 0
        # fValleyMean = 0
        # fValleyVar = 0
        # iValleyNum = 0
        fPeakDistMean = 0
        # fPVDistMean = 0
        # fPeakDistVar = 0
        # fPVDistVar = 0
        iStatus = iFirstType
        for i in range(0, len(idx)):
            if iStatus == 1:
                iPeakNum = iPeakNum + 1
                # fPeakMean = fPeakMean + sf[idx[i]]
                # fPeakVar = fPeakVar + sf[idx[i]]*sf[idx[i]]
                if i > 1:
                    fPeakDistMean = fPeakDistMean + idx[i] - idx[i - 2]
                    # fPeakDistVar = fPeakDistVar + (idx[i] - idx[i-2]) * (idx[i] - idx[i-2])
            # else:
            #     iValleyNum = iValleyNum + 1
            #     fValleyMean = fValleyMean + sf[idx[i]]
            #     fValleyVar = fValleyVar + sf[idx[i]]*sf[idx[i]]
            # if i<len(idx)-1:
            #     fPVDistMean = fPVDistMean + idx[i+1] - idx[i]
            #     fPVDistVar = fPVDistVar + (idx[i+1] - idx[i]) * (idx[i+1] - idx[i])
            iStatus = -iStatus

        if iPeakNum >= 2:
            fPeakDistMean = fPeakDistMean / (iPeakNum - 1)
            # fPeakDistVar = fPeakDistVar/(iPeakNum-1) - fPeakDistMean*fPeakDistMean
        else:
            fPeakDistMean = 0
            # fPeakDistVar = 0

        # if len(idx)>=2:
        #     fPVDistMean = fPVDistMean/(len(idx)-1)
        #     fPVDistVar = fPVDistVar/(len(idx)-1) - fPVDistMean*fPVDistMean
        # else:
        #     fPVDistMean = 0
        #     fPVDistVar = 0

        # if iPeakNum>0:
        #     fPeakMean = fPeakMean/iPeakNum
        #     fPeakVar = np.sqrt(fPeakVar/iPeakNum - fPeakMean*fPeakMean)
        # else:
        #     fPeakVar = 0

        # if iValleyNum>0:
        #     fValleyMean = fValleyMean/iValleyNum
        #     fValleyVar = np.sqrt(fValleyVar/iValleyNum - fValleyMean*fValleyMean)
        # else:
        #     fValleyVar = 0

        # if iPeakNum>0 and iValleyNum>0:
        #     fAmplitude = fPeakMean - fValleyMean
        # else:
        #     fAmplitude = 0

        return fPeakDistMean

    for m in range(0, bh):
        for n in range(0, bw):
            if DIR[m, n] != 91:
                P[m, n] = ProjectNormal(m, n)

    return P


def ComputeDir(I, BLK_SIZE, smoothsize=None):
    """Compute blockwise direction image using gradient method.
    % Problems:
    %   coh may be large in some background block
    %   e.x.,D:/fvc2004/image/bmpDB1_B/101_1.bmp
    %
    % Reference:
    %   A. M. Bazen and S. H. Gerez, 'Systematic Methods for the
    %   Computation of the Directional Fields and Singular Points of Fingerprints',
    %   IEEE PAMI, vol. 24, no. 7, pp. 905-919, 2002.
    %
    % Jianjiang Feng
    % 2007-3

    Args:
        I ([type]): grayscale fingerprint image
        BLK_SIZE ([type]): block size
        smoothsize ([type], optional): [description]. Defaults to None.

    Returns:
        DIR: blockwise direction image, [-90, 90]
        COH: coherence, with the same size as DIR, [0, 1]
    """
    height, width = I.shape
    vBlockNum = math.ceil(height / BLK_SIZE)
    hBlockNum = math.ceil(width / BLK_SIZE)
    DIR = np.zeros((vBlockNum, hBlockNum))
    COH = np.zeros((vBlockNum, hBlockNum))

    # gradient
    dI = I.astype(np.double)
    Gy, Gx = np.gradient(dI)
    Gy = -Gy

    # double the angle
    Gsx = np.power(Gx, 2) - np.power(Gy, 2)
    Gsy = 2 * np.multiply(Gx, Gy)
    coh = np.sqrt(np.power(Gsx, 2) + np.power(Gsy, 2))

    if smoothsize is None:
        smoothsize = 16

    Gsx = cv2.blur(Gsx, (smoothsize, smoothsize))
    Gsy = cv2.blur(Gsy, (smoothsize, smoothsize))
    coh = cv2.blur(coh, (smoothsize, smoothsize))

    coh = np.divide(np.sqrt(np.power(Gsx, 2) + np.power(Gsy, 2)), (coh + np.spacing(1) + 1e-8))

    if BLK_SIZE > 1:
        for by in range(1, vBlockNum + 1):
            for bx in range(1, hBlockNum + 1):
                x1 = (bx - 1) * BLK_SIZE + 1
                x2 = bx * BLK_SIZE
                y1 = (by - 1) * BLK_SIZE + 1
                y2 = by * BLK_SIZE

                # border block may be small
                if x2 > width:
                    x2 = width
                    x1 = width - BLK_SIZE + 1
                if y2 > height:
                    y2 = height
                    y1 = height - BLK_SIZE + 1

                tempGsy = Gsy[y1 - 1 : y2, x1 - 1 : x2]
                tempGsx = Gsx[y1 - 1 : y2, x1 - 1 : x2]

                DIR[by - 1, bx - 1] = (
                    np.atan2(np.sum(tempGsy), np.sum(np.sum(tempGsx))) / 2
                )
                tempCOH = coh[y1 - 1 : y2, x1 - 1 : x2]
                COH[by, bx] = np.mean(tempCOH)
    else:
        DIR = np.arctan2(Gsy, Gsx) / 2
        COH = coh

    INDEX = np.nonzero(DIR > 0)
    DIR += np.pi / 2
    DIR[INDEX] -= np.pi
    DIR = np.round(ROUND_EPS + np.rad2deg(DIR))

    return DIR, COH


def FillHoleDir(D1, MASK):
    """Fit the missed orientation elements in hole of the orientation field

    Args:
        D1 ([type]): direction image
        MASK ([type]): 0: background, 1: reliable block, 2: block to be fitted

    Returns:
        [type]: fitted direction image
    """
    D2 = copy.deepcopy(D1)
    h, w = D1.shape
    D1 = np.deg2rad(D1)
    r, c = np.nonzero(MASK == 2)
    dx = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
    dy = np.array([-1, -1, -1, 0, 1, 1, 1, 0])
    ratio = np.zeros((8,))
    ratio[0:7:2] = 2
    ratio[1:8:2] = 1
    for k in range(0, len(r)):
        weight = 0
        cosv = 0
        sinv = 0
        for m in range(0, 8):
            lenx = w
            if dx[m] > 0:
                lenx = w - c[k] - 1
            elif dx[m] < 0:
                lenx = c[k]
            leny = h
            if dy[m] > 0:
                leny = h - r[k] - 1
            elif dy[m] < 0:
                leny = r[k]
            mlen = min(lenx, leny)

            r1 = r[k] + dy[m] * np.arange(1, (mlen + 1))
            c1 = c[k] + dx[m] * np.arange(1, (mlen + 1))

            idx2 = np.nonzero(MASK[r1, c1] == 1)[0]

            if idx2.shape[0] > 0:
                idx2 = idx2[0]
                tempWeight = 1 / (ratio[m] * (idx2 + 1) * (idx2 + 1))
                weight = weight + tempWeight
                cosv = cosv + tempWeight * np.cos(2 * D1[r1[idx2], c1[idx2]])
                sinv = sinv + tempWeight * np.sin(2 * D1[r1[idx2], c1[idx2]])

        D2[r[k], c[k]] = np.arctan2(sinv, cosv) * 90 / np.pi

    return D2


def FillHolePed(Y1, MASK):
    """Fit the missed period elements in hole of the orientation field

    Args:
        Y1 ([type]): period image
        MASK ([type]): 0: background, 1: reliable block, 2: block to be fitted

    Returns:
        [type]: fitted period image
    """
    Y2 = copy.deepcopy(Y1)
    h, w = Y1.shape
    r, c = np.nonzero(MASK == 2)
    dx = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
    dy = np.array([-1, -1, -1, 0, 1, 1, 1, 0])
    ratio = np.zeros((8,))
    ratio[0:7:2] = 2
    ratio[1:8:2] = 1
    for k in range(0, len(r)):
        weight = 0
        sumv = 0
        for m in range(0, 8):
            lenx = w
            if dx[m] > 0:
                lenx = w - c[k] - 1
            elif dx[m] < 0:
                lenx = c[k]
            leny = h
            if dy[m] > 0:
                leny = h - r[k] - 1
            elif dy[m] < 0:
                leny = r[k]
            mlen = min(lenx, leny)

            r1 = r[k] + dy[m] * np.arange(1, (mlen + 1))
            c1 = c[k] + dx[m] * np.arange(1, (mlen + 1))

            idx2 = np.nonzero(MASK[r1, c1] == 1)[0]

            if idx2.shape[0] > 0:
                idx2 = idx2[0]
                tempWeight = 1 / (ratio[m] * (idx2 + 1) * (idx2 + 1))
                weight = weight + tempWeight
                sumv = sumv + tempWeight * Y1[r1[idx2], c1[idx2]]

        Y2[r[k], c[k]] = sumv / weight

    return Y2
