'''
Descripttion: 
version: 
Author: Xiongjun Guan
Date: 2021-07-23 11:15:30
LastEditors: Xiongjun Guan
LastEditTime: 2021-08-15 12:01:52
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from scipy import optimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import griddata
from skimage.measure import CircleModel
# from tqdm import tqdm
# from scipy.optimize import minimize
# from skimage.measure import EllipseModel
# from scipy import integrate
# from sympy import Symbol, solve
# import sympy
# from PIL import Image
'''
def GetNearestEllipsePoint(p, a, b, xc, yc, theta):
    """find the shortest distance from any point to the ellipse by combining the elliptic equation with the tangent equation.

    Args:
        p ((1, 2) array): Point.
        a (float): Major axis.
        b (float): Minor axis.
        xc (float): Horizontal axis center.
        yc (float): Vertical axis center.
        theta (float): Inclination angle(counterclockwise).

    Returns:
        [type]: The shortest distance from point to the ellipse.
    """
    tc = np.array([xc, yc])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    pc = np.dot(p - tc, R)

    x = Symbol('x')
    y = Symbol('y')
    p_ell = solve([(x / a)**2 + (y / b)**2 - 1, a**2 * y *
                   (x - pc[0]) - b**2 * x * (y - pc[1])], [x, y])

    p0 = np.array(p_ell[0]).astype(np.float)
    p1 = np.array(p_ell[1])

    if np.hypot(p0 - p) < np.hypot(p1 - p):
        return np.dot(p0, R.T) + tc
    else:
        return np.dot(p1, R.T) + tc


def EllipseFit(px,
               py,
               param_init=np.array([0.005, 0.005, 0.005, 0.005, 0.005])):
    """Fit 2D scatter points with ellipse
    The elliptic equation is set to : Ax^2 + Bxy + Cy^2 + Dx + Ey + F, where A is set to 1.

    Args:
        px ((N, ) array): The horizontal axis of points.
        py ((N, ) array): The vertical axis of points.
        param_init (float, optional): The initial value of the parameter [B,C,D,E,F]. Defaults to np.array([0.005, 0.005, 0.005, 0.005, 0.005]).

    Returns:
        [A, B, C, D, E, F]: Fitting parameters of the elliptic general equation. If the fitting fails, return [].
        flag: If it can be fitted into an ellipse, return True, otherwise return False.
        [xc, yc, a, b, q]: [horizontal axis center, vertical axis center,  major axis, minor axis, inclination angle(counterclockwise)]. If the fitting fails, return [].
    """
    def myfunc(x):
        B, C, D, E, F = x
        A = 1
        v = A * px * px + B * px * py + C * py * py + D * px + E * py + F
        return np.mean(np.abs(v))

    def consfun1(x):
        B, C, D, E, F = x
        A = 1
        xc = (B * E - 2 * C * D) / (4 * A * C - B * B)
        yc = (B * D - 2 * A * E) / (4 * A * C - B * B)

        p1 = A * (xc**2) + C * (yc**2) + B * xc * yc - F
        p2 = A + C + math.sqrt((A - C)**2 + B**2)
        p3 = A + C - math.sqrt((A - C)**2 + B**2)

        return p1 / p3 - p2 / p3

    # get params
    res = minimize(
        fun=myfunc,
        x0=param_init,
        method='SLSQP',
        bounds=((None, None), (None, None), (None, None), (None, None),
                (None, None)),
        constraints=(
            {
                'type': 'ineq',
                'fun': lambda x: x[1]
            },  # b > 0
            {
                'type': 'ineq',
                'fun': lambda x: x[4]
            },
            {
                'type': 'ineq',
                'fun': lambda x: consfun1
            }))
    B, C, D, E, F = res.x
    A = 1

    DELTA = np.linalg.det(np.array([[A, B, D], [B, C, E], [D, E, F]]))
    delta = np.linalg.det(np.array([[A, B], [B, C]]))

    # calculate the center of the ellipse
    xc = (B * E - 2 * C * D) / (4 * A * C - B * B)
    yc = (B * D - 2 * A * E) / (4 * A * C - B * B)

    try:
        # major / minor axis
        b = math.sqrt(2 * ((A * (xc**2) + C * (yc**2) + B * xc * yc - F) /
                      (A + C + math.sqrt((A - C)**2 + B**2))))
        a = math.sqrt(2 * ((A * (xc**2) + C * (yc**2) + B * xc * yc - F) /
                      (A + C - math.sqrt((A - C)**2 + B**2))))
    except:
        return [], False, []

    # calculate the inclination angle of the ellipse
    if abs(B) < 1e-10:
        if A < C:
            q = 0
        else:
            q = np.pi / 2
    elif abs(A) < abs(C):
        q = 0.5 * math.atan(B / (A - C))
    else:
        q = 0.5 * math.atan(B / (C - A))

    return [A, B, C, D, E, F], True, [xc, yc, a, b, q]
'''


def GetNearestEllipseAngle(p, params):
    """Get the angle from the negative direction of the vertical axis to the point on the ellipse closest to the specified point (anti-clockwise).

    Args:
        p ((N, 2) array): Points.
        params (list): [xc, yc, a, b, theta] for [horizontal axis center, vertical axis center,  major axis, minor axis, inclination angle(counterclockwise)].

    Returns:
        t ((N, ) array): Anti-clockwise angle.
        residuals ((N, ) array): Euclidean distance from the nearest point on the ellipse to the specified point.
    """
    xc, yc, a, b, theta = params

    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    x = p[:, 0]
    y = p[:, 1]

    def fun(t, xi, yi):
        ct = math.cos(t)
        st = math.sin(t)
        xt = xc + a * ctheta * st + b * stheta * ct
        yt = yc + a * stheta * st - b * ctheta * ct
        return (xi - xt)**2 + (yi - yt)**2

    N = p.shape[0]
    t = np.empty((N, ), dtype=np.double)
    residuals = np.empty((N, ), dtype=np.double)

    # initial guess for parameter t of closest point on ellipse
    t0 = np.arctan2(x - xc, yc - y) - theta

    # determine shortest distance to ellipse for each point
    for i in range(N):
        xi = x[i]
        yi = y[i]
        ti, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
        t[i] = np.mod(ti, 2 * math.pi)
        residuals[i] = np.sqrt(fun(t[i], xi, yi))

    return t, residuals


def Euler2Matrix(roll, pitch, yaw):
    """Convert Euler angle to rotation matrix R.
    If the points' shape is (N, 3), then the rotated vector is np.dot(points, R.T).

    Args:
        roll (float): X.
        pitch (float): Y.
        yaw (float): Z.

    Returns:
        R ((3, 3) array): Rotation matrix.
    """
    R = Rotation.from_euler("ZYX", [yaw, pitch, roll],
                            degrees=True).as_matrix()
    return R


def DrawPCA_2D(points, d1, d2, center=None):
    """project [the point cloud] and [its principal axis vector obtained by PCA] in 2D plane.

    Args:
        points ((N, 3) array): Points for [x ,y, z]
        d1 (int): Dimension (0, 1, 2 for x, y, z)
        d2 (int): The other dimension (0, 1, 2 for x, y, z)
        center ((2, ) array, optional): Points' center. Defaults to None (decentralize the point set and set the center to [0,0,0]).
    """
    dim = ['x', 'y', 'z']

    pca = PCA(n_components=1)
    pca.fit(points)
    main_vec = pca.components_

    if center is None:
        points -= np.mean(points, axis=0)
        center = np.zeros(3)
    plt.figure()
    plt.scatter(points[:, d1], points[:, d2], s=1, c='b')
    plt.axis("equal")
    plt.scatter(center[d1], center[d2], s=8, c='r')
    plt.plot(np.array([center[d1], center[d1] + 10 * main_vec[0, d1]]),
             np.array([center[d2], center[d2] + 10 * main_vec[0, d2]]),
             '-r',
             linewidth=2)
    plt.xlabel(dim[d1])
    plt.ylabel(dim[d2])


def PoseCorrection(points, normals=None):
    """Correct finger posture. Make the finger surface face the negative direction of Y axis.
    Use PCA method to correct x-axis and z-axis.
    Use the normal vector to correct y-axis.

    Args:
        points ((N,3) array): [x,y,z] points.
        normals ((N,3) array, optional): Normal vectors of points. Defaults to None (will not correct the y-axis posture).

    Returns:
        points ((N,3) array): [x,y,z] points.
        normals ((N,3) array): Normal vectors of points. 
    """
    points -= np.mean(points, axis=0)
    pca = PCA(n_components=1)
    pca.fit(points)

    main_vec = pca.components_

    standard_vec = np.array([0, -1, 0])
    if np.dot(main_vec, standard_vec) < 0:
        main_vec = -main_vec

    yaw = math.degrees(math.atan2(main_vec[0, 0], -main_vec[0, 1]))
    roll = -math.degrees(math.asin(main_vec[0, 2]))
    R = Euler2Matrix(roll=roll, pitch=0, yaw=yaw)

    points = np.dot(points, R)  # (R.T).T
    points -= np.mean(points, axis=0)

    if normals is not None:
        normals = np.dot(normals, R)  # (R.T).T
        mean_normal = np.mean(normals, axis=0)
        pitch = math.degrees(math.atan2(mean_normal[0], -mean_normal[2]))
        R = Euler2Matrix(roll=0, pitch=pitch, yaw=0)
        normals = np.dot(normals, R.T)
        points = np.dot(points, R.T)
        points -= np.mean(points, axis=0)

    return points, normals


def ClockwiseAngle(base_vec, rotate_vec):
    """Clockwise angle from base_vec to rotate_vec

    Args:
        base_vec ((2, ) array): Reference vector.
        rotate_vec ((2, ) array): Rotated vector.

    Returns:
        theta (float): Clockwise angle (rad).
    """
    x1, y1 = base_vec
    x2, y2 = rotate_vec
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    return theta


def DrawEllipse(points_x, points_y, xc, yc, a, b, theta):
    """Draw points and corresponding fitted ellipse

    Args:
        points_x ((N, ) array): Abscissa of points.
        points_y ((N, ) array)): Ordinate of points.
        xc (float): Horizontal axis center.
        yc (float): Vertical axis center.
        a (float): Major axis.
        b (float): Minor axis.
        theta (float): Inclination angle(counterclockwise).
    """
    plt.figure()
    plt.axes().set_aspect('equal', adjustable='box')
    plt.axes().scatter(points_x, points_y, c='b')
    ell_patch = Ellipse((xc, yc),
                        2 * a,
                        2 * b,
                        np.rad2deg(theta),
                        edgecolor='red',
                        facecolor='none')
    plt.axes().add_patch(ell_patch)


def arc_int(phi, a, b):
    """ Differential function of elliptic arc length (Start with the minor axis below).
    The parametric equation of the ellipse is:
        x = a sin(phi),
        y = -b cos(phi).
    Therefore, the differential formula of arc length is:
        [dL] = sqrt( (a cos(phi))**2 + (b sin(phi))**2 ) [dphi].

    Args:
        phi (float): Anti-clockwise angle from the lower minor axis to the corresponding point (rad).
        a (float): Major axis.
        b (float): Minor axis.

    Returns:
        float: [dL]/[dphi].
    """
    return math.sqrt(
        math.pow(a * math.cos(phi), 2) + math.pow(b * math.sin(phi), 2))


def GetEllPoints(t, params):
    """Get the coordinates of a point according to [the ellipse parameters] and [the corresponding angle of the point on the ellipse]

    Args:
        t (float): Anti-clockwise angle from the lower minor axis of the ellipse to this point (rad).
        params (list): [xc, zc, a, b, theta] for [horizontal axis center, vertical axis center,  major axis, minor axis, inclination angle(counterclockwise)].

    Returns:
        xt (float): Abscissa of point.
        zt (float): Ordinate of point.
    """
    xc, zc, a, b, theta = params
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    ct = math.cos(t)
    st = math.sin(t)
    xt = xc + a * ctheta * st + b * stheta * ct
    zt = zc + a * stheta * st - b * ctheta * ct
    return xt, zt


'''
def PointsFlattenByEll(points, dpi):
    """Calculate the corresponding coordinates in a 2D plane after expanding the 3D point cloud
    * The expanded abscissa is obtained by fitting the ellipse and integrating according to the elliptic curve. 
    The starting point of integration is the corresponding point closest to the longitudinal axis
    * The expanded ordinate is obtained by calculating the distance and accumulating the integral 
    according to the corresponding points in the negative direction of the longitudinal axis of each slice fitting ellipse

    Args:
        points ((N, 3) array): Points for [x, y, z]. The scale unit of point cloud is [mm] by default.
        dpi (flaot): Dots per inch. 1 inch = 25.4 mm.

    Returns:
        uv_points ((N, 2) array): 2D coordinates (vertical axis, horizontal axis) after expanding.
        jump_step (list): Indexes of sections where fitting failed.
    """
    mm2pixel = dpi / 25.4
    pixel2mm = 1 / mm2pixel

    # expanded coordinates
    uv_points = np.ones([points.shape[0], 2]) * np.nan

    # longitudinal slice according to dpi
    max_y = np.max(points[:, 1])
    min_y = np.min(points[:, 1])
    len_y = math.ceil((max_y - min_y) * mm2pixel)

    # indexes corresponding to each point
    id = np.arange(points.shape[0])

    # cumulative integral initialization of vertical axis
    v = 0
    init_step = 0
    while True:
        step_flag = (points[:, 1] >= min_y + init_step * pixel2mm) & (
            points[:, 1] < min_y + (init_step + 1) * pixel2mm)
        points_step = points[step_flag, :]
        try:
            ell = EllipseModel()
            ell.estimate(points_step[:, [0, 2]])
            _, z_pre = GetEllPoints(-ell.params[4], ell.params)
            break
        except:
            init_step += 1

    # failure of ellipse fitting
    jump_step = []

    for step in tqdm(range(init_step, len_y, 1)):
        # points and indexes corresponding to each longitudinal slice
        step_flag = (points[:, 1] >= min_y + step * pixel2mm) & (
            points[:, 1] < min_y + (step + 1) * pixel2mm)
        points_step = points[step_flag, :]
        id_step = id[step_flag]

        # # ---- fit x-y points with ellipse -------
        # _, ell_flag, params = EllipseFit(points_step[:, 0], points_step[:,
        #                                                                   2])
        # if ell_flag is False:
        #     # print("can not fit in %d" % step)
        #     jump_step.append(step)
        #     continue
        # xc, yc, a, b, theta = params
        # print(xc, yc, a, b, np.rad2deg(theta))
        # # ----------------------------------------

        # fit x-y points with ellipse
        try:
            ell = EllipseModel()
            ell.estimate(points_step[:, [0, 2]])
            xc, zc, a, b, theta = ell.params
        except:
            # When fitting is not possible, the last longitudinal gradient is used as the approximation
            v += math.sqrt(math.pow(z_post - z_pre, 2) + math.pow(pixel2mm, 2))
            jump_step.append(step)
            continue

        # flat z
        _, z_post = GetEllPoints(-theta, ell.params)
        v += math.sqrt(math.pow(z_post - z_pre, 2) + math.pow(pixel2mm, 2))
        z_pre = z_post

        # # ---- show the fitting of the section ----
        # DrawEllipse(points_step[:,0], points_step[:,2], xc, yc, a, b, theta)
        # print(xc, yc, a, b, np.rad2deg(theta))
        # plt.show()
        # # -----------------------------------------

        # # ---- flat x ----------------------------
        # phi_begin = -theta
        # base_vec = np.array([0, -1])
        # for ps_i in range(points_step.shape[0]):
        #     point = points_step[ps_i, [0,2]]
        #     rotate_vec = point - np.array([xc, yc])
        #     phi_end = ClockwiseAngle(base_vec, rotate_vec) - theta

        #     if phi_end > math.pi:
        #         phi_end = phi_end - 2 * math.pi

        #     u, err = integrate.quad(arc_int, phi_begin, phi_end, args=(a, b))
        #     uv_points[id_step[ps_i], 0] = v * mm2pixel
        #     uv_points[id_step[ps_i], 1] = u * mm2pixel
        # # ----------------------------------------

        # flat x
        phi_end, _ = GetNearestEllipseAngle(points_step[:, [0, 2]], ell.params)
        # phi_begin = -theta
        # !!!! There is a problem with the estimation of the starting point, which has not been solved !!!!
        phi_begin = -math.atan(xc/(b*math.sqrt(1-(xc/a)**2)))-theta
        for ps_i in range(points_step.shape[0]):
            if phi_end[ps_i] - phi_begin > math.pi:
                phi_end[ps_i] = phi_end[ps_i] - 2 * math.pi
            u, _ = integrate.quad(arc_int,
                                  phi_begin,
                                  phi_end[ps_i],
                                  args=(a, b))
            uv_points[id_step[ps_i], 0] = v * mm2pixel
            uv_points[id_step[ps_i], 1] = u * mm2pixel

    return uv_points, jump_step
'''


def PointsFlattenByCircle(points, dpi):
    """Calculate the corresponding coordinates in a 2D plane after expanding the 3D point cloud
    * The expanded abscissa is obtained by fitting the circle and integrating according to the circle curve. 
    The starting point of integration is the corresponding point closest to the longitudinal axis
    * The expanded ordinate is obtained by calculating the distance and accumulating the integral 
    according to the corresponding points in the negative direction of the longitudinal axis of each slice fitting ellipse

    Args:
        points ((N, 3) array): Points for [x, y, z]. The scale unit of point cloud is [mm] by default.
        dpi (flaot): Dots per inch. 1 inch = 25.4 mm.

    Returns:
        uv_points ((N, 2) array): 2D coordinates (vertical axis, horizontal axis) after expanding.
        jump_step (list): Indexes of sections where fitting failed.
    """
    mm2pixel = dpi / 25.4
    pixel2mm = 1 / mm2pixel

    # expanded coordinates
    uv_points = np.ones([points.shape[0], 2]) * np.nan

    # longitudinal slice according to dpi
    max_y = np.max(points[:, 1])
    min_y = np.min(points[:, 1])
    len_y = math.ceil((max_y - min_y) * mm2pixel)

    # indexes corresponding to each point
    id = np.arange(points.shape[0])

    # cumulative integral initialization of vertical axis
    v = 0
    init_step = 0
    while True:
        step_flag = (points[:, 1] >= min_y + init_step * pixel2mm) & (
            points[:, 1] < min_y + (init_step + 1) * pixel2mm)
        points_step = points[step_flag, :]
        try:
            circle = CircleModel()
            circle.estimate(points_step[:, [0, 2]])
            xc, zc, r = circle.params
            if points_step.shape[0] == 0 or abs(xc) > r or zc < np.mean(
                    points_step[:, 2]):
                init_step += 1
                continue
            z_pre = zc - r
            z_post = z_pre
            break
        except:
            init_step += 1

    # failure of circle fitting
    jump_step = []

    # for step in tqdm(range(init_step, len_y, 1)):
    for step in range(init_step, len_y, 1):
        # points and indexes corresponding to each longitudinal slice
        step_flag = (points[:, 1] >= min_y + step * pixel2mm) & (
            points[:, 1] < min_y + (step + 1) * pixel2mm)
        points_step = points[step_flag, :]
        id_step = id[step_flag]

        # fit x-y points with circle
        try:
            circle = CircleModel()
            circle.estimate(points_step[:, [0, 2]])
            xc, zc, r = circle.params
            # When fitting is incorrect, the last longitudinal gradient is used as the approximation
            if points_step.shape[0] == 0 or abs(xc) > r or zc < np.mean(
                    points_step[:, 2]):
                v += math.sqrt(
                    math.pow(z_post - z_pre, 2) + math.pow(pixel2mm, 2))
                jump_step.append(step)
                continue
        except:
            # When fitting is not possible, the last longitudinal gradient is used as the approximation
            v += math.sqrt(math.pow(z_post - z_pre, 2) + math.pow(pixel2mm, 2))
            jump_step.append(step)
            continue

        # flat z
        z_post = zc - r
        v += math.sqrt(math.pow(z_post - z_pre, 2) + math.pow(pixel2mm, 2))
        z_pre = z_post

        # # ---- show the fitting of the section ----
        # DrawEllipse(points_step[:,0], points_step[:,2], xc, zc, r, r, 0)
        # plt.show()
        # # -----------------------------------------

        # flat x
        base_vec = np.array([0, -1])
        phi_begin = math.asin(xc / r)
        for ps_i in range(points_step.shape[0]):
            point = points_step[ps_i, [0, 2]]
            rotate_vec = point - np.array([xc, zc])
            phi_end = ClockwiseAngle(base_vec, rotate_vec)
            phi = phi_end + phi_begin
            if phi > math.pi:
                phi = phi - 2 * math.pi
            u = phi * r
            uv_points[id_step[ps_i], 0] = v * mm2pixel
            uv_points[id_step[ps_i], 1] = u * mm2pixel

    return uv_points, jump_step


def HistEqualization(arr):
    """Normalize arr to [0, 255] and perform histogram equalization

    Args:
        arr ((N, ) array): Arr
    Returns:
        arr_equal ((N, ) array): Arr after histogram equalization
    """
    min_arr = arr.min()
    max_arr = arr.max()
    arr = 255 * (arr - min_arr) / (max_arr - min_arr)
    arr = np.squeeze(np.uint8(arr))
    arr_hist, arr_bins = np.histogram(arr, bins=256, range=(0, 256))

    cdf = np.cumsum(arr_hist)
    cdf = (cdf - cdf[0]) * 255 / (cdf[-1] - 1)
    cdf = np.uint8(cdf)
    arr_equal = cdf[arr]

    return arr_equal


def GenerateFlattenResult(points,
                          depth,
                          uv_points,
                          edge,
                          brightness=0.6,
                          fill_value=255):
    """Get the interpolated results from the expanded coordinates.

    Args:
        points ((N, 3) array): Points for [x, y, z].
        depth ((N, ) array): The surface depth of each point, with this value as the gray value.
        uv_points ((N, 2) array): Expanded coordinates [v, u].
        edge (int): Pixels of the blank boundary of the image.
        brightness (float, optional): Luminance coefficient of image. If it is 1, the image does not change. Defaults to 0.6.
        fill_value (int, optional): The fill value of the blank area of the image. Defaults to 255.

    Returns:
        grid_fp ((r, c) array): Generated image.
        grid_gt ((r, c, 3) array): Coordinates [x, y, z] of each pixel in the image on the point cloud.
    """

    # remove points where deployment failed
    depth = depth[~np.isnan(uv_points[:, 0])]
    points = points[~np.isnan(uv_points[:, 0]), :]
    uv_points = uv_points[~np.isnan(uv_points[:, 0]), :]

    # translate to image coordinates and sets boundaries.
    uv_points[:, 0] = uv_points[:, 0] - np.min(uv_points[:, 0]) + edge
    uv_points[:, 1] = uv_points[:, 1] - np.min(uv_points[:, 1]) + edge

    # image size
    max_v = np.round(np.max(uv_points[:, 0])) + edge
    max_u = np.round(np.max(uv_points[:, 1])) + edge

    depth = 255 - HistEqualization(depth)

    grid_v, grid_u = np.mgrid[1:max_v:1, 1:max_u:1]

    grid_fp = np.squeeze(
        griddata(uv_points,
                 depth, (grid_v, grid_u),
                 method='linear',
                 fill_value=fill_value))
    grid_gt = np.squeeze(
        griddata(uv_points,
                 points, (grid_v, grid_u),
                 method='linear',
                 fill_value=math.nan))

    grid_fp /= brightness

    grid_fp[grid_fp < 0] = 0
    grid_fp[grid_fp > 255] = 255
    grid_fp = np.uint8(grid_fp)

    return grid_fp, grid_gt
