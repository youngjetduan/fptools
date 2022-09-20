import scipy.io as scio
import numpy as np
import os
import math
# from stl import mesh  # numpy-stl
import open3d
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score


def mkdir(path):
    '''
    判断是否存在文件夹, 如果不存在则创建为文件夹
    :param path:文件夹路径
    :return:
    '''
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def PtSegmentation(path, type, save_dir, save_name=None):
    if type is 'mat':
        data = scio.loadmat(path)
        pts = data['points']
    else:
        print('wrong input \"type\"!')
        os._exit(0)
        return

    if save_name is None:
        save_name = os.path.basename(path)[:-4]

    z = pts[:, 2].reshape(-1, 1)
    gmm_2 = GaussianMixture(n_components=2).fit(z)
    labels_2 = gmm_2.predict(z)

    if davies_bouldin_score(z, labels_2) < 0.4:
        if gmm_2.means_[0] > gmm_2.means_[1]:
            pts = pts[labels_2 == 0]
        else:
            pts = pts[labels_2 == 1]

        mkdir(save_dir)
        file = open(save_dir + save_name + '.txt', 'w')
        file.close()

    mkdir(save_dir)
    scio.savemat(save_dir + save_name + '.mat', {'points': pts})


def ComputePt(centralized_verts, r):
    '''
    :param centralized_verts: 中心化的点云数据(Nx3)
    :param r: 以某点附近半径r的球形域内部点作为平面微元
    :return:
    '''
    # 初始化法向量、表面深度存储
    normal_vec = np.nan * np.ones((centralized_verts.shape[0], 3))
    surface_depth = np.nan * np.ones(centralized_verts.shape[0])

    verts_len = centralized_verts.shape[0]

    mean = np.mean
    svd = np.linalg.svd
    argmin = np.argmin

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(centralized_verts)
    pcd_tree = open3d.geometry.KDTreeFlann(pcd)

    for i in range(verts_len):

        # 将所有与选定点距离小于r的点构成平面
        point = centralized_verts[i, :]

        [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], r)

        Plane = centralized_verts[idx, :]

        # 微元平面中心化
        plane_center = np.array([mean(Plane[:, 0]), mean(Plane[:, 1]), mean(Plane[:, 2])])
        centerPlane = Plane - plane_center

        # SVD分解找到法向量
        [_, S, VT] = svd(centerPlane)
        V = VT.T

        # 如果周围点不足，此点作废
        if S.shape[0] < 3:
            continue

        # 法向量为投影强度最小对应的特征向量
        plane_nv = np.ascontiguousarray(V[:, argmin(np.array([S[0], S[1], S[2]]))])  # 局部平面法向量

        # 校正法向量方向，从点云中心指向局部平面
        if plane_center.dot(plane_nv) < 0:
            plane_nv = -plane_nv

        # 保存法向量
        normal_vec[i, :] = plane_nv

        # 表面深度
        surface_depth[i] = np.vdot(point - plane_center, plane_nv)

    # print('\n')
    return normal_vec, surface_depth


def ComputePtData(path, type='stl', r=0.15, save_dir=None, save_name=None, rotate=None):
    '''
    计算点云中每个点的法向量和表面深度
    :param path: 文件路径
    :param r: 以某点附近半径r的球形域内部点作为平面微元
    :param save_dir: 数据保存路径
    :param save_name: 数据保存名称，默认为读入的名称
    :return: 点云(Nx3), 法向量(Nx3), 归一化后的表面深度(Nx3)
    '''
    if type is 'mat':
        data = scio.loadmat(path)
        points = data['points']
    else:
        print('error input \"type\"!')
        os._exit(0)
        return

    if save_name is None:
        save_name = os.path.basename(path)[:-4]

    if rotate is not None:
        R = Ang2Matrix(rotate[0], rotate[1], rotate[2], type='ang')
        points = np.dot(points, R.T)

    verts_center = np.mean(points, 0)  # 点云中心

    # 中心化
    points -= verts_center

    # normals, depth = JitComputePt(points, r)
    normals, depth = ComputePt(points, r)

    # 排除孤立点
    points = points[~np.isnan(depth), :]
    normals = normals[~np.isnan(depth), :]
    # stl_normals = stl_normals[~np.isnan(depth), :]
    depth = depth[~np.isnan(depth)].reshape(-1, 1)

    if save_dir is not None:
        mkdir(save_dir)
        scio.savemat(save_dir + save_name + '.mat',
                     {'points': points, 'normals': normals, 'depth': depth})


def Ang2Matrix(pitch, yaw, roll, type):
    '''
    从角度转换为旋转矩阵，弧度制
    :param pitch:绕X轴旋转
    :param yaw:绕Y轴旋转
    :param roll:绕Z轴旋转
    :param type:'ang'角度制，'rad'弧度制
    :return:旋转矩阵R
    '''
    if type is 'ang':
        pitch, yaw, roll = math.radians(pitch), math.radians(yaw), math.radians(roll)
    elif type is not 'rad':
        print('wrong input in Ang2Matrix \"measure\"!')
        os._exit(0)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), - np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    R = np.dot(np.dot(Rx, Ry), Rz)
    return R
