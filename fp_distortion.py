'''
Description:  
Author: Guan Xiongjun
Date: 2022-02-07 20:40:38
LastEditTime: 2022-08-31 13:52:30
LastEditors: Please set LastEditors
'''
import numpy as np
from scipy.interpolate import griddata



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
        method="nearest",
    ).reshape(img_shape)

    if need_rect is True:
        dx_rect = griddata(
            np.hstack((x + dx.reshape((-1, 1)),
                    y + dy.reshape((-1, 1)))),
            (-dx).reshape((-1, 1)),
            np.hstack((x, y)),
            method="nearest",
        ).reshape(img_shape)

        dy_rect = griddata(
            np.hstack((x + dx.reshape((-1, 1)),
                    y + dy.reshape((-1, 1)))),
            (-dy).reshape((-1, 1)),
            np.hstack((x, y)),
            method="nearest",
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