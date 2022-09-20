'''
Description: 
Author: Guan Xiongjun
Date: 2022-09-12 10:37:58
LastEditTime: 2022-09-20 17:31:09
'''
import cv2
import numpy as np
import sys
import os
import os.path as osp
import scipy.io as scio
from scipy.spatial.distance import pdist
from scipy.ndimage import distance_transform_edt
import networkx as nx
from copy import deepcopy

sys.path.append(osp.dirname(osp.abspath(__file__)))
from fp_orientation import calc_orientation_graident
from fp_segmtation import find_largest_connected_region


eps = 1e-5

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') // array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols

def sub2ind_cols(array_shape, rows, cols):
    """ind2sub (python index)

    Args:
        array_shape (_type_): _description_
        rows (_type_): _description_
        cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = rows + cols*array_shape[0]
    return ind

def sub2ind_rows(array_shape, rows, cols):
    """ind2sub (MATLAB index)

    Returns:
        _type_: _description_
    """
    ind = rows*array_shape[1] + cols
    return ind


def getMaskCenter(mask):
    """get center of mask

    Args:
        mask (_type_): 0 / 1 binary mask

    Returns:
        _type_: _description_
    """
    h,w = mask.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    xc = np.sum(x*mask)/(np.sum(mask)+eps)
    yc = np.sum(y*mask)/(np.sum(mask)+eps)

    return np.array([xc,yc])


def mask_to_graph(mask,weights):
    """An undirected graph is constructed.
    in which the loss of each pixel is used as the weight of the edge where the adjacent point reaches the point.

    Args:
        mask (_type_): 0 / 1 binary mask
        weights (_type_): loss function

    Returns:
        _type_: _description_
    """
    h,w = mask.shape
    mask = np.float64(mask>0)
    indices=np.where(mask>0)
    num_of_nodes=len(indices[0])
    indices=np.arange(0,num_of_nodes)[:,np.newaxis]

    # horizontal edge
    mask1=np.hstack((mask[:,1:],np.zeros((h,1))))
    mask1=mask1*mask

    mask1=mask1[mask>0]
    s1=indices[mask1>0]
    mask1=np.hstack((np.ones((h,1)),mask[:,0:-1]))
    mask1=mask1*mask
    w1=weights[mask1>0][:,np.newaxis]
    mask1=mask1[mask>0]
    t1=indices[mask1>0]

    # vertical edge
    mask1=np.vstack((mask[1:,:],np.zeros((1,w))))
    mask1=mask1*mask
    mask1=mask1[mask>0]
    s2=indices[mask1>0]
    mask1=np.vstack((np.zeros((1,w)),mask[0:-1,:]))
    mask1=mask1*mask
    w2=weights[mask1>0][:,np.newaxis]
    mask1=mask1[mask>0]
    t2=indices[mask1>0]

    s = np.vstack((s1,s2))
    t = np.vstack((t1,t2))
    w = np.vstack((w1,w2))
    edges_data = np.hstack((s,t,w))

    G = nx.Graph()
    G.add_weighted_edges_from(edges_data)

    return G

def find_path(G,mask,p_start,p_end):
    """Generate the shortest path according to the undirected graph and the given starting and ending points.
    Binary mask is returned, where the shortest path is 1 and the rest is 0.

    Args:
        G (_type_): _description_
        mask (_type_): _description_
        p_start (_type_): _description_
        p_end (_type_): _description_

    Returns:
        _type_: _description_
    """
    h,w = mask.shape
    ind1=sub2ind_rows([h,w],p_start[1],p_start[0])
    ind2=sub2ind_rows([h,w],p_end[1],p_end[0])

    indices=np.arange(0,mask.shape[0]*mask.shape[1]).reshape(mask.shape)[mask>0]
   
    s = np.where(indices==ind1)
    t = np.where(indices==ind2)

    P = np.int32(nx.shortest_path(G,s[0][0],t[0][0],'weight'))
    path_mask=np.zeros((h,w))
    P=indices[P]
    path_mask = path_mask.reshape((-1,))
    path_mask[P]=1
    path_mask = path_mask.reshape(mask.shape)

    return path_mask


def mosaic_line(img1,img2,bimg1,bimg2,MASK1,MASK2):
    """Generate Divided Region.
    The farthest intersection of the two mask boundaries is chosen as the starting point/end point. 
    The overlap of the binary graph and the distance from the boundary are used as weights to generate the shortest path

    Args:
        img1 (_type_): _description_
        img2 (_type_): _description_
        bimg1 (_type_): _description_
        bimg2 (_type_): _description_
        MASK1 (_type_): _description_
        MASK2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    MASK1 = np.float32(MASK1)
    MASK2 = np.float32(MASK2)
    MASK = MASK1*MASK2
    MASK = find_largest_connected_region(MASK)
    DIR1,_ = calc_orientation_graident(bimg1, win_size=16, stride=1)
    DIR2,_ = calc_orientation_graident(bimg2, win_size=16, stride=1)

    dir_diff=np.abs(DIR1-DIR2)*MASK
    dir_diff=np.minimum(dir_diff,180-dir_diff)
    img_diff=np.abs(np.float32(bimg1)/255-np.float32(bimg2)/255)

    
    # start and end point
    contours,_ = cv2.findContours(np.uint8(MASK1),mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    edge1 = np.zeros_like(MASK1)
    cv2.drawContours(edge1,contours,-1,1,1) 

    contours,_ = cv2.findContours(np.uint8(MASK2),mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    edge2 = np.zeros_like(MASK2)
    cv2.drawContours(edge2,contours,-1,1,1)


    h,w = MASK1.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    edge = (edge1>0) & (edge2>0) & (MASK>0)
    xi = x[edge]
    yi = y[edge]

    if len(xi)==0:
        if np.sum(MASK2)<np.sum(MASK1):
            xi = x[edge2>0]
            yi = y[edge2>0]
        else:
            xi = x[edge1>0]
            yi = y[edge1>0]

    xi_rep = np.repeat(xi[:,np.newaxis],len(xi),1)
    yi_rep = np.repeat(yi[:,np.newaxis],len(xi),1)
    dx = xi_rep - xi_rep.T
    dy = yi_rep - yi_rep.T

    d = np.sqrt(dx*dx+dy*dy)

    id = np.argmax(d)
    id1,id2 = ind2sub([len(xi),len(xi)], id)
    p_start = [xi[id1],yi[id1]]
    p_end = [xi[id2],yi[id2]]

    distance = distance_transform_edt(np.int32(1-MASK))
    weights=dir_diff+img_diff+np.exp(-distance/100)
    weights=weights*MASK
    weights[weights<0]=0

    G = mask_to_graph(MASK,weights)
    P = find_path(G,MASK,p_start,p_end)


    MASK_2=deepcopy(MASK)
    MASK_2[P>0]=0
    region1 = find_largest_connected_region(MASK_2)
    region2 = MASK*(1-region1)
    mask11 = MASK1 * (1-MASK)
    mask22 = MASK2 * (1-MASK)

    MASK1_cen = getMaskCenter(MASK1).reshape((2,1))
    region1_cen = getMaskCenter(region1).reshape((2,1))
    region2_cen = getMaskCenter(region2).reshape((2,1))
    
    if pdist(MASK1_cen-region1_cen)[0]<pdist(MASK1_cen-region2_cen)[0]:
        mask1_crop = (mask11>0) | (region1>0)
        mask2_crop = (mask22>0) | (region2>0)
    else:
        mask1_crop = (mask11>0) | (region2>0)
        mask2_crop = (mask22>0) | (region1>0)
    
    img = img1 * mask1_crop + img2 * mask2_crop
    blk_mask = 1-((mask1_crop>0) | (mask2_crop>0))
    img[blk_mask>0] = 255

    return np.uint8(img),mask1_crop,mask2_crop



if __name__ == "__main__":
    img1 = cv2.imread('/disk1/guanxiongjun/tmp/111/1.png',0)
    img2 = cv2.imread('/disk1/guanxiongjun/tmp/111/2.png',0)
    bimg1 = cv2.imread('/disk1/guanxiongjun/tmp/111/b1.png',0)
    bimg2 = cv2.imread('/disk1/guanxiongjun/tmp/111/b2.png',0)
    mask1 = cv2.imread('/disk1/guanxiongjun/tmp/111/m1.png',0)/255
    mask2 = cv2.imread('/disk1/guanxiongjun/tmp/111/m2.png',0)/255

    img,_,_ = mosaic_line(img1,img2,bimg1,bimg2,mask1,mask2)

    cv2.imwrite('/home/guanxiongjun/code/hisign_registration/tmp/img.png',img)


