'''
Description:  
Author: Guan Xiongjun
Date: 2022-02-07 20:40:38
LastEditTime: 2022-09-24 11:31:35
LastEditors: Please set LastEditors
'''
import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
import sys
import os
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))
from uni_tps import tps_module_numpy,tps_apply_transform


def apply_distortion_torch(img, coordinate,image_height,image_width,mode='bilinear'):
    """_summary_

    Args:
        img (_type_): _description_
        coordinate (_type_): dst coordinate. from -1 to 1 .
        image_height (_type_): _description_
        image_width (_type_): _description_
        mode (str, optional): _description_. Defaults to 'bilinear'.

    Returns:
        _type_: _description_
    """
    batch_size = img.size(0)
    coordinate = coordinate.view(batch_size, image_height, image_width, 2)
    img = F.grid_sample(img, coordinate,align_corners=False,mode=mode)
    return img

def apply_distortion(img, dx, dy, need_rect=False):
    """distorted image according to the displacement
    I'(x+dx, y+dy) = I(x,y)

    Args:
        img (_type_): M x N (gray style)
        dx (_type_): M x N 
        dy (_type_): M x N 
        need_rect (bool, optional): give the dual displacement.like DX'(x+dx, y+dy) = -DX(x,y)

    Returns:
        _type_: _description_
    """
    img_shape = img.shape
    x, y = np.meshgrid(np.arange(0, img_shape[0]), np.arange(0, img_shape[1]))
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    img_dis = griddata(
        np.hstack((x + dx.reshape((-1, 1)),
                  y + dy.reshape((-1, 1)))),
        img.reshape((-1, 1)),
        np.hstack((x, y)),
        method="linear",
        fill_value=255,
    ).reshape(img_shape)

    if need_rect is True:
        dx_rect = griddata(
            np.hstack((x + dx.reshape((-1, 1)),
                    y + dy.reshape((-1, 1)))),
            (-dx).reshape((-1, 1)),
            np.hstack((x, y)),
            method="linear",
        ).reshape(img_shape)

        dy_rect = griddata(
            np.hstack((x + dx.reshape((-1, 1)),
                    y + dy.reshape((-1, 1)))),
            (-dy).reshape((-1, 1)),
            np.hstack((x, y)),
            method="linear",
        ).reshape(img_shape)

        return img_dis,dx_rect,dy_rect

    else:
        return img_dis

def transform_rect_to_dis(img_shape, dx_rect, dy_rect):
    """give the dual displacement. Noted that distortion should be the true size (1,2,3,4,...)

    Args:
        img_shape (_type_): (M, N, ...)
        dx_rect (_type_): M x N
        dy_rect (_type_): M x N

    Returns:
        _type_: _description_
    """
    x, y = np.meshgrid(np.arange(0, img_shape[0]), np.arange(0, img_shape[1]))
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    dx = griddata(
        np.hstack((x + dx_rect.reshape((-1, 1)),
                y + dy_rect.reshape((-1, 1)))),
        (-dx_rect).reshape((-1, 1)),
        np.hstack((x, y)),
        method="nearest",
    ).reshape(img_shape)

    dy = griddata(
        np.hstack((x + dx_rect.reshape((-1, 1)),
                y + dy_rect.reshape((-1, 1)))),
        (-dy_rect).reshape((-1, 1)),
        np.hstack((x, y)),
        method="nearest",
    ).reshape(img_shape)

    return dx,dy

def generate_outside_distortion_by_common_area(dx,dy,common_mask):
    h,w = common_mask.shape
    x, y = np.meshgrid(np.arange(0, h), np.arange(0, w))

    zoom_param = 1/8
    dx_resize = zoom(dx,zoom=zoom_param, order=1)
    dy_resize = zoom(dy,zoom=zoom_param ,order=1)
    mask_resize = zoom(common_mask,zoom=zoom_param ,order=1)
    mask_resize = (mask_resize>0.5)
    x_resize = zoom(x,zoom=zoom_param ,order=1)
    y_resize = zoom(y,zoom=zoom_param ,order=1)

    xs = x_resize[mask_resize>0]
    ys = y_resize[mask_resize>0]
    dxs = dx_resize[mask_resize>0]
    dys = dy_resize[mask_resize>0]

    src_cpts = np.float32(np.vstack((xs,ys)).T)
    src_pts = np.float32(np.vstack((x.reshape((-1,)),y.reshape((-1,)))).T)
    tar_cpts = np.float32(np.vstack(((xs+dxs,ys+dys))).T)

    mapping_matrix = tps_module_numpy(src_cpts,tar_cpts,5)
    tar_pts = tps_apply_transform(src_pts, src_cpts, mapping_matrix)
    dx = tar_pts[:,0].reshape(x.shape)-x
    dy = tar_pts[:,1].reshape(y.shape)-y
    return dx,dy
    