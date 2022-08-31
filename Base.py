"""
Descripttion: Python version converted from the 'base' code (Matlab version)
version:2.0
Author: Xiongjun Guan
Date: 2021-08-23 19:59:57
LastEditors: Xiongjun Guan
LastEditTime: 2021-08-25 18:02:39
"""

import cv2
from numpy.lib import RankWarning
from scipy.ndimage import distance_transform_edt
from skimage import measure
import numpy as np
import copy
import os.path as osp
from scipy.ndimage import zoom, rotate
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

ROUND_EPS = 0.00001  # for the problem of np.round() in '0.5'


def MakeSameSize(X, h1, w1, fillval):
    """MakeSameSize - Image size is not exactly same after minify and magnify, so modify it
    Args:
        X ([type]): Origin image
        h1 ([type]): height
        w1 ([type]): weight
        fillval ([type]): Fill value
    Returns:
        Y: Filled image
    """
    Y = copy.deepcopy(X)
    h2, w2 = X.shape

    if h2 > h1:
        Y = Y[0:h1, :]
    elif h2 < h1:
        rows = h1 - h2
        Y = np.vstack((Y, fillval * np.ones((rows, w2))))

    if w2 > w1:
        Y = Y[:, 0:w1]
    elif w2 < w1:
        cols = w1 - w2
        Y = np.hstack((Y, fillval * np.ones((h1, cols))))

    return Y


def MedianFilter(I1, MASK, r):
    """Median filter the area where mask = 1
    Args:
        I1 ([type]): 2-D array
        MASK ([type]):  1 object, 0 background
        r ([type]): Filter radius
    Returns:
        I2: Filtered array
    """
    h, w = I1.shape
    I2 = copy.deepcopy(I1)
    DX, DY = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    DX = np.reshape(DX, (-1, 1), order="F")
    DY = np.reshape(DY, (-1, 1), order="F")
    y, x = np.nonzero(MASK > 0)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    for i in range(0, len(y)):
        y2 = y[i] + DY
        x2 = x[i] + DX
        mask = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
        y2 = y2[mask]
        x2 = x2[mask]
        mask = MASK[y2, x2] > 0
        y2 = y2[mask]
        x2 = x2[mask]
        vals = I1[y2, x2]
        I2[y[i], x[i]] = np.median(vals)

    return I2


def FillHole(A):
    """Fill holes which are not connected to the border
    Args:
        A ([type]): 1 object, 0 background
    Raises:
        Exception: A.shape is smaller than 3x3
    Returns:
        B: 1 object, 0 background, 2 hole
    """
    B = copy.deepcopy(A)
    h, w = A.shape

    if h < 3 or w < 3:
        raise Exception("It does not work for matrix smaller than 3x3!")

    bx = np.concatenate(
        (np.arange(0, w), (w - 1) * np.ones((h,)), np.arange(0, w), np.zeros((h,))),
        axis=0,
    ).astype(np.int)
    by = np.concatenate(
        (np.zeros((w,)), np.arange(0, h), (h - 1) * np.ones((w,)), np.arange(0, h)),
        axis=0,
    ).astype(np.int)

    L, num = measure.label((A == 0), connectivity=1, return_num=True)

    for k in range(1, num + 1):
        if np.nonzero(L[by, bx] == k)[0].shape[0] == 0:
            B[L == k] = 2

    return B


def DetectSP(DIR, MASK, layer_num, bComputeDir=False):
    """Detect singular points using PointCare method
    Args:
        DIR ([type]): direction image
        MASK ([type]): 1 object, 0 background
        layer_num ([type]): num of resize layers
        bComputeDir (bool, optional): [description]. Defaults to False.
    Returns:
        sps: [x y type direction] of singular points
    """
    # ---------------------------------------
    def ComputeCoreDirection(sp, DIR, MASK):
        """Core direction
        Args:
            sp ([type]): [x,y,type] of singular point
            DIR ([type]): direction image
            MASK ([type]):  1 object, 0 background
        Returns:
            alphas: [description]
            score: [description]
        """
        r = 30
        step = 1
        [X, Y] = np.meshgrid(np.arange(-r, r + 1, step), np.arange(-r, r + 1, step))
        refCosv = np.true_divide(
            -Y, np.sqrt(np.power(X, 2) + np.power(Y, 2)) + np.spacing(1)
        )
        refSinv = np.true_divide(
            X, np.sqrt(np.power(X, 2) + np.power(Y, 2)) + np.spacing(1)
        )
        x1 = sp[0] - r
        y1 = sp[1] - r
        x2 = sp[0] + r
        y2 = sp[1] + r
        delta_x1 = max(0, -x1)
        delta_x2 = max(0,x2-(DIR.shape[1] - 1))
        delta_y1 = max(0, -y1)
        delta_y2 = max(0, y2 - (DIR.shape[0] - 1))

        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(DIR.shape[1] - 1, x2))
        y2 = int(min(DIR.shape[0] - 1, y2))
        mask = MASK[y1 : (y2 + 1), x1 : (x2 + 1)]
        refCosv = refCosv[delta_y1 : 2*r+1-delta_y2, delta_x1 : 2*r+1-delta_x2]
        refSinv = refSinv[delta_y1 : 2*r+1-delta_y2, delta_x1 : 2*r+1-delta_x2]
        mask[sp[1] - y1, sp[0] - x1] = 0
        dirs = DIR[y1 : (y2 + 1), x1 : (x2 + 1)]
        cosv = np.cos(dirs[mask == 1] * np.pi / 90)
        sinv = np.sin(dirs[mask == 1] * np.pi / 90)
        Z = cosv + 1j * sinv
        refZ = refCosv[mask == 1] + 1j * refSinv[mask == 1]
        s = np.multiply(Z, refZ)
        alpha = np.arctan2(np.sum(np.imag(s)), np.sum(np.real(s))) * 180 / np.pi - 90
        score = np.abs(np.mean(s))

        #  find the most symmetric direction
        max_sym = 0.6
        temp_alpha = alpha
        for ddir in range(-20, 24, 4):
            sym = ComputeSymmetry(
                DIR, sp[0], sp[1], NormalizeRidgeDir(ddir + temp_alpha + 90)
            )
            if sym is not None and sym > max_sym:
                max_sym = sym
                alpha = NormalizeMinuDir(ddir + temp_alpha)

        return alpha, score

    # ---------------------------------------
    def ComputeDeltaDirection(sp, DIR, MASK):
        """Delta direction
        Args:
            sp ([type]): [x,y,type] of singular point
            DIR ([type]): direction image
            MASK ([type]):  1 object, 0 background
        Returns:
            alphas: [description]
            score: [description]
        """
        alphas = np.zeros((3,))
        # The first angle is computed in the same way as core angle.
        r = 30
        step = 1
        [X, Y] = np.meshgrid(np.arange(-r, r + 1, step), np.arange(-r, r + 1, step))
        refCosv = np.true_divide(
            Y, np.sqrt(np.power(X, 2) + np.power(Y, 2)) + np.spacing(1)
        )
        refSinv = np.true_divide(
            X, np.sqrt(np.power(X, 2) + np.power(Y, 2)) + np.spacing(1)
        )
        x1 = sp[0] - r
        y1 = sp[1] - r
        x2 = sp[0] + r
        y2 = sp[1] + r
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(DIR.shape[1] - 1, x2))
        y2 = int(min(DIR.shape[0] - 1, y2))
        mask = MASK[y1 : (y2 + 1), x1 : (x2 + 1)]
        mask[sp[1] - y1, sp[0] - x1] = 0
        dirs = DIR[y1 : (y2 + 1), x1 : (x2 + 1)]
        cosv = np.cos(dirs[mask == 1] * np.pi / 90)
        sinv = np.sin(dirs[mask == 1] * np.pi / 90)
        Z = cosv + 1j * sinv

        if mask.shape[0] == X.shape[0] and mask.shape[1] == X.shape[1]:
            refZ = refCosv[mask == 1] + 1j * refSinv[mask == 1]
        else:
            refy, refx = np.nonzero(mask == 1)
            refy = refy + y1 - sp[1] + r
            refx = refx + x1 - sp[0] + r
            refZ = refCosv[refy, refx] + 1j * refSinv[refy, refx]

        s = np.multiply(Z, refZ)
        alphas[0] = np.arctan2(np.sum(np.imag(s)), np.sum(np.real(s))) * 60 / np.pi + 90
        score = np.abs(np.mean(s))

        # The other 2 angles are computed as mean direction of predicted region.
        for k in range(0, 2):
            pred_angle = NormalizeMinuDir(alphas[0] - (k + 1) * 120)
            alphas[k + 1] = pred_angle
            continue
            # THETA,RHO = np.meshgrid(np.arange(pred_angle-30,pred_angle+30+1),np.arange(10,31))
            # X,Y = Pol2Cart(-THETA*pi/180,RHO)
            # X = np.round(ROUND_EPS + X + sp[0])
            # Y = np.round(ROUND_EPS + Y + sp[1])

            # mask2 = (X>=0) & (X<DIR.shape[1]) & (Y>=0) & (Y<DIR.shape[0])
            # x_idx = X[mask2]
            # y_idx = Y[mask2]

            # y_idx2, x_idx2 = np.nonzero(MASK[y_idx,x_idx])

            # if x_idx2.shape[0] == 0:
            #     score = -1
            #     break

            # x_idx = x_idx[x_idx2]
            # y_idx = y_idx[y_idx2]

            # cosv = cos(DIR[y_idx,x_idx]*np.pi/90)
            # sinv = sin(DIR[y_idx,x_idx]*np.pi/90)
            # alphas[k+1] = np.arctan2(np.mean(sinv),np.mean(cosv))*90/np.pi
            # if np.abs(NormalizeMinuDir(alphas[k+1]-pred_angle))>90:
            #     alphas[k+1] = NormalizeMinuDir(alphas[k+1]+180)

        return alphas, score

    # ---------------------------------------
    def ClusterSP(SPS):
        """Clustering singular points
        Args:
            SPS ([type]): array of [[x, y, type direction]]
        Returns:
            sps: array of [[x, y, type direction]] after clustering
        """
        clusters = []
        properties = []  # [[x,y,type]]
        for i in range(0, SPS.shape[0]):
            min_dist = 100
            cid = 0
            for j in range(0, len(clusters)):
                if properties[j][2] != SPS[i, 2]:  # distihguish up core and down cores
                    continue
                dist = np.sqrt(
                    np.power(properties[j][0] - SPS[i, 0], 2)
                    + np.power(properties[j][1] - SPS[i, 1], 2)
                )
                if dist < min_dist:
                    cid = j
                    min_dist = dist

            if min_dist <= 2:
                clusters[cid].append(i)
                properties[cid][0] = np.mean(SPS[clusters[cid], 0])
                properties[cid][1] = np.mean(SPS[clusters[cid], 1])
            else:
                clusters.append([i])
                properties.append([SPS[i, 0], SPS[i, 1], SPS[i, 2]])

        sps = np.array(properties)
        sps[:, 0:2] = np.round(ROUND_EPS + sps[:, 0:2])

        return sps.astype(np.int)

    # ---------------------------------------
    def iDetectSP(type, DIR, x, y):
        """Calculate the possible singular points according to the angle change
        Note: the adjacent positions of the same singular point may be marked, so clustering is required
        Args:
            type ([type]): singular type
            DIR ([type]): orientation field
            x ([type]): candidate points' x coordinate
            y ([type]): candidate points' y coordinate
        Returns:
            sps: singular points
        """
        xx = np.array([1, 0, -1, -1, -1, 0, 1, 1], dtype=np.int16)
        yy = np.array([-1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int16)

        cdir = np.empty(9, dtype=np.double)
        dif = np.empty(9, dtype=np.double)
        sps = []

        for i in range(0, len(x)):
            cdir[0:8] = DIR[y[i] + yy, x[i] + xx]
            cdir[8] = cdir[0]

            tmp1 = -1
            tmp2 = 0
            for j in range(0, 8):
                dif[j] = np.fmod(cdir[j + 1] - cdir[j], 180)
                if dif[j] < 0:
                    dif[j] += 180
                if dif[j] > 90:
                    dif[j] -= 180
                if dif[j] == 90:
                    if tmp1 == -1:
                        tmp1 = j
                    elif tmp1 > 0:
                        tmp1 = -2
                if dif[j] < -10:
                    tmp2 += 1
            if tmp1 >= 0 and tmp2 >= 2:
                dif[tmp1] = -90
            index = np.sum(dif[0:8])
            if abs(index - 180) < 0.1:
                index = 1
            elif abs(index + 180) < 0.1:
                index = 2
            else:
                index = 0

            if (type != -1 and index == type) or (type == -1 and index > 0):
                sps.append([x[i], y[i], index])
        return np.array(sps)

    sps = []

    DIRS = np.empty((layer_num,), dtype=object)
    MASKS = np.empty((layer_num,), dtype=object)

    DIRS[0] = copy.deepcopy(DIR)
    MASKS[0] = copy.deepcopy(MASK)

    for k in range(1, layer_num):
        DIRS[k], MASKS[k] = DownSampleDirField(DIRS[k - 1], MASKS[k - 1])

    MASKS[layer_num - 1][[1, -1], :] = 0
    MASKS[layer_num - 1][:, [1, -1]] = 0

    MASKS[layer_num - 1] = cv2.filter2D(
        MASKS[layer_num - 1].astype(np.double), -1, np.ones((3, 3), dtype=np.double)
    )
    MASKS[layer_num - 1] = (MASKS[layer_num - 1] == 9).astype(np.int)

    # Detect SP
    SPS = np.empty((layer_num), dtype=object)
    y, x = np.nonzero(MASKS[layer_num - 1] > 0)
    SPS[layer_num - 1] = iDetectSP(-1, DIRS[layer_num - 1], x, y)
    if SPS[layer_num - 1].shape[0] == 0:
        return SPS[layer_num - 1]

    [X, Y] = np.meshgrid(np.arange(-2, 2), np.arange(-2, 2))

    for k in range(layer_num - 2, -1, -1):
        for i in range(0, SPS[k + 1].shape[0]):
            sps_new = iDetectSP(
                    SPS[k + 1][i, -1],
                    DIRS[k],
                    2 * SPS[k + 1][i, 0] + X.reshape(-1, order="F") + 1,
                    2 * SPS[k + 1][i, 1] + Y.reshape(-1, order="F") + 1,
                )
            if sps_new is not None and sps_new.shape[0] > 0:
                if SPS[k] is None or SPS[k].shape[0] == 0:
                    SPS[k] = sps_new
                else:
                    SPS[k] = np.vstack((SPS[k], sps_new))

    # Clustering SP
    if SPS[0] is None or SPS[0].shape[0] == 0:
        return np.empty((0, 6))
    sps = ClusterSP(SPS[0])

    sps_out = np.hstack((sps.astype(np.double), np.ones_like(sps)))

    if bComputeDir:
        for i in range(0, sps.shape[0]):
            if np.mod(sps[i, 2], 2) == 1:
                alpha, score = ComputeCoreDirection(sps[i, :], DIR, MASK)
                sps_out[i, 3:6] = alpha
            else:
                alphas, score = ComputeDeltaDirection(sps[i, :], DIR, MASK)
                sps_out[i, 3:6] = alphas

    return sps_out


def GetBranchCuts(DIR, sps):
    """Find branch cuts by tracing orientation field starting from singular points
    Stop tracing when crossing current curve
    Args:
        DIR ([type]): direction image
        sps ([type]): [x y type direction] of singular points
    Returns:
        xnn: [description]
        ynn: [description]
    """
    xn = []
    yn = []
    if sps.shape[0] == 0 or sps.shape[1] == 0:
        return xn, yn

    # constants
    radius = 6
    step = 4
    pts = np.empty((sps.shape[0],), dtype=object)
    SID = np.empty((sps.shape[0],), dtype=object)  # connected sp
    h, w = DIR.shape

    # deal with cores, connect two cores if they agree
    ind = np.nonzero(sps[:, 2] == 1)[0]
    rec = np.array([])

    for ii in range(0, len(ind) - 1):
        for jj in range(ii + 1, len(ind)):
            i = ind[ii]
            j = ind[jj]
            if np.any(i == rec) or np.any(j == rec):
                continue

            xnn, ynn = BresLine(
                int(sps[i, 0]), int(sps[i, 1]), int(sps[j, 0]), int(sps[j, 1])
            )
            flag = 0
            if len(xnn) < 30:
                if np.abs(np.abs(sps[i, 3] - sps[j, 3]) - 180) > 70:
                    continue
                flag = 1
            else:
                if np.abs(np.abs(sps[i, 3] - sps[j, 3]) - 180) > 50:
                    continue
                dirs = DIR[ynn, xnn]
                if (
                    np.sum(
                        np.abs(
                            NormalizeRidgeDir(
                                dirs
                                - np.arctan2(
                                    -(sps[i, 1] - sps[j, 1]), sps[i, 0] - sps[j, 0]
                                )
                                / np.pi
                                * 180
                            )
                        )
                        < 45
                    )
                    / len(dirs)
                    > 0.6
                ):
                    flag = 1
            if flag == 1:
                rec = np.hstack((rec, i, j))
                SID[i] = np.array([j])
                SID[j] = np.array([i])
                pts[i] = np.empty((1,), dtype=object)
                pts[j] = np.empty((1,), dtype=object)
                pts[i][0] = np.array([xnn, ynn])
                pts[j][0] = np.vstack(
                    (np.fliplr(xnn.reshape(1, -1)), np.fliplr(ynn.reshape(1, -1)))
                )

    def SetSpDist(sps, sid):
        SPD = 100 * np.ones_like(DIR)
        SIMAGE = -1 * np.ones_like(DIR, dtype=int)
        for a in range(0, sps.shape[0]):
            if a == sid:
                continue
            x1 = int(max(sps[a, 0] - radius, 0))
            x2 = int(min(sps[a, 0] + radius, w - 1))
            y1 = int(max(sps[a, 1] - radius, 0))
            y2 = int(min(sps[a, 1] + radius, h - 1))
            c1 = int(x1 - sps[a, 0])
            c2 = int(x2 - sps[a, 0])
            r1 = int(y1 - sps[a, 1])
            r2 = int(y2 - sps[a, 1])
            X, Y = np.meshgrid(np.arange(c1, c2 + 1), np.arange(r1, r2 + 1))
            _, RADIUS = Cart2Pol(X, Y)
            SPD[y1 : (y2 + 1), x1 : (x2 + 1)] = RADIUS
            SIMAGE[y1 : (y2 + 1), x1 : (x2 + 1)] = a
        return SPD, SIMAGE

    def TraceOrientationdField(x0, y0, dir0):
        x0 = int(x0)
        y0 = int(y0)
        T = np.zeros((h, w))  # record traced line
        nCount = 1
        xs = np.array([x0])
        ys = np.array([y0])
        sid = -1
        bStop = 0
        while True:
            x1 = np.round(ROUND_EPS + x0 + step * np.cos(dir0 * np.pi / 180))
            y1 = np.round(ROUND_EPS + y0 - step * np.sin(dir0 * np.pi / 180))
            xx, yy = BresLine(x0, y0, x1, y1)
            for k in range(1, len(xx)):
                xs = np.hstack((xs, xx[k]))
                ys = np.hstack((ys, yy[k]))

                # No need to check border, since border direction is set as background (91)

                # background?
                if DIR[yy[k], xx[k]] == 91:
                    xs = xs[0:-1]
                    ys = ys[0:-1]
                    bStop = 1
                    break

                # circle
                if T[yy[k], xx[k]] > 0 and T[yy[k], xx[k]] < nCount - 1:
                    bStop = 1
                    break

                # another sp?
                if SPD[yy[k], xx[k]] < radius:
                    # connect them
                    sid = SIMAGE[yy[k], xx[k]]
                    xx2, yy2 = BresLine(xx[k], yy[k], sps[sid, 0], sps[sid, 1])
                    xs = np.hstack((xs, xx2[1:]))
                    ys = np.hstack((ys, yy2[1:]))
                    bStop = 1
                    break

            if bStop == 1:
                break

            # Record line
            xn, yn = eight2four(xx, yy)
            T[yn, xn] = nCount
            nCount = nCount + 1

            x0 = xx[-1]
            y0 = yy[-1]
            if np.abs(NormalizeMinuDir(DIR[y0, x0] - dir0)) > 90:
                dir0 = NormalizeMinuDir(DIR[y0, x0] + 180)
            else:
                dir0 = DIR[y0, x0]

        return (
            xs.astype(np.int).reshape((1, -1)),
            ys.astype(np.int).reshape((1, -1)),
            sid,
        )
        """[summary]
        Returns:
            [type]: [description]
        """

    for i in range(0, sps.shape[0]):
        if np.any(rec == i):
            continue
        SPD, SIMAGE = SetSpDist(sps, i)

        if np.mod(sps[i, 2], 2) == 1:
            pts[i] = np.empty((1,), dtype=object)
            xs, ys, sid = TraceOrientationdField(sps[i, 0], sps[i, 1], sps[i, 3])
            pts[i][0] = np.vstack((xs, ys))
            SID[i] = np.array([sid])
        else:
            pts[i] = np.empty((3,), dtype=object)
            SID[i] = np.empty((3,), dtype=object)
            for j in range(0, 3):
                xs, ys, sid = TraceOrientationdField(
                    sps[i, 0], sps[i, 1], sps[i, 3 + j]
                )
                pts[i][j] = np.vstack((xs, ys))
                SID[i][j] = sid

    # select the shortest branch cuts
    sets = np.empty((len(pts[0]) + 1,), dtype=object)
    used = np.empty((len(pts[0]) + 1,), dtype=object)
    lengths = np.empty((len(pts[0]) + 1,), dtype=object)
    for j in range(0, len(pts[0])):
        sets[j] = np.array([j])
        used[j] = np.zeros((sps.shape[0],))
        used[j][0] = np.array([1])
        if SID[0][j] > 0:
            used[j][SID[0][j]] = 1
        lengths[j] = np.array([pts[0][j].shape[1]])
    # select nothing
    sets[-1] = np.array([-1])
    used[-1] = np.zeros((sps.shape[0]))
    lengths[-1] = np.array([0])

    for i in range(1, sps.shape[0]):
        newSet = np.empty((len(sets) * (len(pts[i]) + 1),), dtype=object)
        newUsed = np.empty((len(sets) * (len(pts[i]) + 1),), dtype=object)
        newLengths = np.empty((len(sets) * (len(pts[i]) + 1),), dtype=object)
        new_id = 0
        for m in range(0, len(sets)):
            # select nothing
            newSet[new_id] = np.hstack((sets[m], -1))
            newUsed[new_id] = copy.deepcopy(used[m])
            newLengths[new_id] = copy.deepcopy(lengths[m])
            new_id += 1
            # This sp is already connected
            if used[m][i] > 0:
                continue
            for j in range(0, len(pts[i])):
                # Connect to a used sp?
                k = SID[i][j]
                newSet[new_id] = np.hstack((sets[m], j))
                newUsed[new_id] = copy.deepcopy(used[m])
                newUsed[new_id][i] = 1
                if k > -1:
                    newUsed[new_id][k] = 1
                newLengths[new_id] = lengths[m] + pts[i][j].shape[1]
                new_id += 1
        sets = copy.deepcopy(newSet[0:new_id])
        used = copy.deepcopy(newUsed[0:new_id])
        lengths = copy.deepcopy(newLengths[0:new_id])

    # remove invalid
    newSet = np.empty((len(sets),), dtype=object)
    newUsed = np.empty((len(sets),), dtype=object)
    newLengths = np.empty((len(sets),), dtype=object)
    new_id = 0
    for i in range(0, len(sets)):
        if np.sum(used[i]) == sps.shape[0]:
            newSet[new_id] = copy.deepcopy(sets[i])
            newUsed[new_id] = copy.deepcopy(used[i])
            newLengths[new_id] = copy.deepcopy(lengths[i])
            new_id += 1
    newSet = newSet[0:new_id]
    newUsed = newUsed[0:new_id]
    newLengths = newLengths[0:new_id]

    idx = np.argmin(newLengths)
    xn = np.empty((sps.shape[0] + 1,), dtype=object)
    yn = np.empty((sps.shape[0] + 1,), dtype=object)
    cnt = 0
    for i in range(0, sps.shape[0]):
        if newSet[idx][i] != -1:
            xtmp, ytmp = eight2four(
                pts[i][newSet[idx][i]][0, :], pts[i][newSet[idx][i]][1, :]
            )
            if len(xtmp) == 1:
                continue
            xn[cnt] = xtmp
            yn[cnt] = ytmp
            cnt = cnt + 1
    xn = xn[0 : cnt + 1]
    yn = yn[0 : cnt + 1]

    l = len(xn) - 1
    xn[-1] = np.array([])
    yn[-1] = np.array([])
    for i in range(0, l):
        xn[-1] = np.hstack((xn[-1], xn[i]))
        yn[-1] = np.hstack((yn[-1], yn[i]))

    return xn, yn


def eight2four(xm, ym):
    """Convert a 8-connected line to a 4-connected line
    Args:
        xm ([type]): [description]
        ym ([type]): [description]
    Returns:
        [type]: [description]
    """
    xn = []
    yn = []
    xn.append(xm[0])
    yn.append(ym[0])
    for i in range(1, len(xm)):
        if xm[i] != xm[i - 1] and ym[i] != ym[i - 1]:
            xn.append(xm[i])
            yn.append(ym[i - 1])
        xn.append(xm[i])
        yn.append(ym[i])
    return np.array(xn), np.array(yn)


def BresLine(Ax, Ay, Bx, By):
    """Bresenham's line drawing algorithm.
    Args:
        Ax ([type]): [description]
        Ay ([type]): [description]
        Bx ([type]): [description]
        By ([type]): [description]
    Returns:
        [type]: [description]
    """
    Ax = np.int(Ax)
    Ay = np.int(Ay)
    Bx = np.int(Bx)
    By = np.int(By)
    dX = abs(Bx - Ax)
    dY = abs(By - Ay)
    if Ax > Bx:
        Xincr = -1
    else:
        Xincr = 1
    if Ay > By:
        Yincr = -1
    else:
        Yincr = 1

    xn = []
    yn = []
    if dX >= dY:
        dPr = dY * 2
        dPru = dPr - dX * 2
        P = dPr - dX
        for m in range(0, dX + 1):
            xn.append(Ax)
            yn.append(Ay)
            if P > 0:
                Ax = Ax + Xincr
                Ay = Ay + Yincr
                P = P + dPru
            else:
                Ax = Ax + Xincr
                P = P + dPr
    else:
        dPr = dX * 2
        dPru = dPr - dY * 2
        P = dPr - dY
        for m in range(0, dY + 1):
            xn.append(Ax)
            yn.append(Ay)
            if P > 0:
                Ax = Ax + Xincr
                Ay = Ay + Yincr
                P = P + dPru
            else:
                Ay = Ay + Yincr
                P = P + dPr

    return np.array(xn), np.array(yn)


def Cart2Pol(x, y, z=None):
    """Transform Cartesian to polar coordinates.
    Args:
        x ([type]): x
        y ([type]): y
        z ([type], optional): z. Defaults to None.
    Returns:
        [type]: polar coordinates th,r,[z]
    """
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    if z is None:
        return th, r
    else:
        return th, r, z


def Pol2Cart(th, r, z=None):
    """Transform polar to Cartesian coordinates.
    Args:
        th ([type]): Angle (rad)
        r ([type]): Radius
        z ([type], optional): Height. Defaults to None.
    Returns:
        [type]: Cartesian coordinates X,Y,[Z]
    """
    x = np.multiply(r, np.cos(th))
    y = np.multiply(r, np.sin(th))
    if z is None:
        return x, y
    else:
        return x, y, z


def DownSampleDirField(DIR1, MASK1):
    """Downsampling orientation field by a factor of 2
    Args:
        DIR1 ([type]): Direction image
        MASK1 ([type]): 0: background, 1: reliable block
    Returns:
        DIR2: Resized direction image
        MASK2: Resized mask
    """
    cosv = np.cos(DIR1 * np.pi / 90)
    sinv = np.sin(DIR1 * np.pi / 90)

    cosv = zoom(cosv, 0.5, order=1, grid_mode=False)
    sinv = zoom(sinv, 0.5, order=1, grid_mode=False)

    DIR2 = np.arctan2(sinv, cosv) * 90 / np.pi

    MASK2 = zoom(MASK1, 0.5, order=0, grid_mode=False)

    DIR2[MASK2 == 0] = 91

    return DIR2, MASK2


def ComputeSymmetry(DIR, x, y, theta):
    """Compute symmetry of orientation field around a point
    Args:
        DIR ([type]): direction image
        x ([type]): X axis
        y ([type]): Y axis
        theta ([type]): [description]
    Returns:
        sym: Symmetry of orientation field around a point
    """
    sym = 0

    r = 80
    step = 4
    n = int(r / step)

    # Sampling points
    [X1, Y1] = np.meshgrid(np.arange(-r, r + 1, step), np.arange(-r, r + 1, step))
    theta1 = np.deg2rad(theta)
    X = (x + np.round(ROUND_EPS + X1 * np.cos(theta1) + Y1 * np.sin(theta1))).astype(
        np.int
    )
    Y = (y + np.round(ROUND_EPS + -X1 * np.sin(theta1) + Y1 * np.cos(theta1))).astype(
        np.int
    )

    # Out of image
    if (
        np.max(X) >= DIR.shape[1]
        or np.min(X) < 0
        or np.max(Y) >= DIR.shape[0]
        or np.min(Y) < 0
    ):
        return

    error = 0
    count = 0
    for c in range(-n, 0):
        x1 = X[:, c + n]
        x2 = X[:, -c + n]
        y1 = Y[:, c + n]
        y2 = Y[:, -c + n]
        dir1 = DIR[y1, x1]
        dir2 = DIR[y2, x2]
        flag = (dir1 != 91) & (dir2 != 91)
        dir1 = dir1[flag]
        dir2 = dir2[flag]
        if len(dir1) < 1:
            continue
        dir1 = NormalizeRidgeDir(dir1 - theta - 90)
        dir2 = NormalizeRidgeDir(dir2 - theta - 90)
        error = error + np.sum(np.abs(NormalizeRidgeDir(dir1 + dir2)))
        count = count + len(dir1)
    sym = (90 - error / (count+ROUND_EPS)) / 90
    return sym


def NormalizeRidgeDir(X):
    """Convert an angle to the range of (-90,90]
    Args:
        X ([type]): angles
    Returns:
        [type]: angles in the range of (-90,90]
    """
    X = np.mod(X, 180)
    if type(X) is np.ndarray:
        X[X > 90] = X[X > 90] - 180
    elif X > 90:
        X = X - 180
    return X


def NormalizeMinuDir(X):
    """Convert an angle to the range of (-180,180]
    Args:
        X ([type]): angles
    Returns:
        X: angles in the range of (-180,180]
    """
    X = np.mod(X, 360)
    if type(X) is np.ndarray:
        r, c = np.nonzero(X > 180)
        X[r, c] = X[r, c] - 360
    elif X > 180:
        X = X - 360
    return X


def FindCurve2(I, startx, starty):
    I_bak = copy.deepcopy(I)
    I = np.ones((I_bak.shape[0] + 2, I_bak.shape[1] + 2))
    I[1:-1, 1:-1] = copy.deepcopy(I_bak)
    startx = startx + 1
    starty = starty + 1
    h, w = I.shape
    idx_y, idx_x = np.nonzero(I == 0)
    if idx_y.shape[0] == 0:
        raise Exception("empty ridge")

    for i in range(0, idx_y.shape[0]):
        n = idx_x[i]
        m = idx_y[i]
        if startx - n > 1 or starty - m > 1:
            continue
        if len(np.nonzero(I[m - 1 : m + 2, n - 1 : n + 2] == 0)[0]) == 2:
            break
        t_patch = (I[m - 1 : m + 2, n - 1 : n + 2] == 0).astype(np.int)
        t_patch[1, 1] = 0
        _, t_num = measure.label(t_patch, connectivity=1, return_num=True)
        if t_num == 1:
            break

    pts = [[n, m]]
    I[pts[0][1], pts[0][0]] = 1
    head = 0
    tail = 0
    while head <= tail:
        for j in range(head, tail + 1):
            cy, cx = np.nonzero(
                I[pts[j][1] - 1 : pts[j][1] + 2, pts[j][0] - 1 : pts[j][0] + 2] == 0
            )
            m_order = np.argsort(np.abs(cy - 1) + np.abs(cx - 1))
            for k in range(0, len(cy)):
                tmpx = pts[j][0] + cx[m_order[k]] - 1
                tmpy = pts[j][1] + cy[m_order[k]] - 1
                I[tmpy, tmpx] = 1
                pts.append([tmpx, tmpy])

        head = tail + 1
        tail = len(pts) - 1
    pts = (np.array(pts) - 1).T
    return pts


def RidgeFilterComplex(im, orient, freq, kx, ky):
    """Function to enhance fingerprint image via oriented filters
    Args:
        im ([type]): Image to be processed.
        orient ([type]): Ridge orientation image, obtained from RIDGEORIENT.
        freq ([type]): Ridge frequency image, obtained from RIDGEFREQ.
        kx, ky ([type]): Scale factors specifying the filter sigma relative
                        to the wavelength of the filter.  This is done so
                        that the shapes of the filters are invariant to the
                        scale.  kx controls the sigma in the x direction
                        which is along the filter, and hence controls the
                        bandwidth of the filter.  ky controls the sigma
                        across the filter and hence controls the
                        orientational selectivity of the filter. A value of
                        0.5 for both kx and ky is a good starting point.
    Returns:
        newim_real,newim_imag: The enhanced image in real and imagine
    """
    # Fixed angle increment between filter orientations in
    # degrees. This should divide evenly into 180
    angleInc = 6

    im = im.astype(np.double)
    rows, cols = im.shape
    newim_real = np.zeros((rows, cols))
    newim_imag = np.zeros((rows, cols))

    validr, validc = np.nonzero(freq > 0)  # find where there is valid frequency data.

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    freq[freq > 0] = np.round(ROUND_EPS + freq[freq > 0] * 100) / 100

    # Generate an array of the distinct frequencies present in the array
    # freq
    unfreq = np.unique(freq[freq > 0])

    # Generate a table, given the frequency value multiplied by 100 to obtain
    # an integer index, returns the index within the unfreq array that it
    # corresponds to
    freqindex = np.ones((100,), dtype=np.int16)
    for k in range(0, len(unfreq)):
        freqindex[int(round(unfreq[k] * 100))] = k

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.
    filter_real = np.empty((len(unfreq), int(360 / angleInc)), dtype=object)
    filter_imag = np.empty((len(unfreq), int(360 / angleInc)), dtype=object)
    sze = np.zeros((len(unfreq),), dtype=np.int)

    for k in range(0, len(unfreq)):
        sigmax = 1 / unfreq[k] * kx
        sigmay = 1 / unfreq[k] * ky

        sze[k] = np.round(ROUND_EPS + 3 * max(sigmax, sigmay))
        x, y = np.meshgrid(
            np.arange(-sze[k], sze[k] + 1), np.arange(-sze[k], sze[k] + 1)
        )
        reffilter_real = np.multiply(
            np.exp(
                -(
                    np.power(x, 2) / np.power(sigmax, 2)
                    + np.power(y, 2) / np.power(sigmay, 2)
                )
                / 2
            ),
            np.cos(2 * np.pi * unfreq[k] * x),
        )
        reffilter_imag = np.multiply(
            -np.exp(
                -(
                    np.power(x, 2) / np.power(sigmax, 2)
                    + np.power(y, 2) / np.power(sigmay, 2)
                )
                / 2
            ),
            np.sin(2 * np.pi * unfreq[k] * x),
        )

        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.
        for o in range(0, int(360 / angleInc)):
            rr_pad = np.max(reffilter_real.shape)
            reffilter_real_pad = np.pad(
                reffilter_real, ((rr_pad, rr_pad), (rr_pad, rr_pad))
            )
            filter_real[k, o] = rotate(
                reffilter_real_pad, -((o + 1) * angleInc + 90), reshape=False, order=1
            )[rr_pad:-rr_pad, rr_pad:-rr_pad]
            filter_real[k, o] = filter_real[k, o] - np.mean(filter_real[k, o])

            ri_pad = np.max(reffilter_imag.shape)
            reffilter_imag_pad = np.pad(
                reffilter_imag, ((ri_pad, ri_pad), (ri_pad, ri_pad))
            )
            filter_imag[k, o] = rotate(
                reffilter_imag_pad, -((o + 1) * angleInc + 90), reshape=False, order=1
            )[rr_pad:-rr_pad, rr_pad:-rr_pad]
            filter_imag[k, o] = filter_imag[k, o] - np.mean(filter_imag[k, o])

    # Find indices of matrix points greater than maxsze from the image
    # boundary
    maxsze = sze[0]
    borderSize = maxsze
    finalind = np.nonzero(
        (validr > maxsze)
        & (validr < rows - maxsze)
        & (validc > maxsze)
        & (validc < cols - maxsze)
    )[0]

    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)
    maxorientindex = np.round(ROUND_EPS + 360 / angleInc)
    orientindex1 = np.round(ROUND_EPS + orient / np.pi * 180 / angleInc)
    orientindex = np.mod(orientindex1, maxorientindex)
    orientindex[orientindex == 0] = orientindex[orientindex == 0] + maxorientindex
    orientindex = (orientindex - 1).astype(np.int16)
    # Finally do the filtering
    for k in range(0, len(finalind)):
        r = validr[finalind[k]]
        c = validc[finalind[k]]

        # find filter corresponding to freq(r,c)
        filterindex = freqindex[int(np.round(ROUND_EPS + freq[r, c] * 100))]

        s = sze[filterindex]
        newim_real[r, c] = np.sum(
            np.multiply(
                im[r - s : r + s + 1, c - s : c + s + 1],
                filter_real[filterindex, orientindex[r, c]],
            )
        )
        newim_imag[r, c] = np.sum(
            np.multiply(
                im[r - s : r + s + 1, c - s : c + s + 1],
                filter_imag[filterindex, orientindex[r, c]],
            )
        )

    return newim_real, newim_imag


def ShowRegisteredImg(img1, img2):
    """Show registered img
    fp common is green
    fp only 1 is gray
    fp only 2 is red
    Args:
        img1 ([type]): fp1
        img2 ([type]): fp2
    Returns:
        im: [description]
    """
    img1 = img1 < 255
    img2 = img2 < 255
    h, w = img1.shape

    im = 255 * np.ones((h, w, 3))
    red = 255 * np.ones((h, w))
    green = 255 * np.ones((h, w))
    blue = 255 * np.ones((h, w))

    common = img1 & img2
    only1 = img1 & (~img2)
    only2 = img2 & (~img1)

    red[common] = 0
    blue[common] = 0
    green[common] = 255

    red[only1] = 126
    blue[only1] = 126
    green[only1] = 126

    red[only2] = 255
    blue[only2] = 0
    green[only2] = 0

    im[:, :, 0] = red
    im[:, :, 1] = green
    im[:, :, 2] = blue

    return im.astype(np.uint8)