'''
Description:  
Author: Guan Xiongjun
Date: 2022-05-08 14:53:34
LastEditTime: 2022-05-10 11:28:24
LastEditors: Guan Xiongjun
'''
import os
import os.path as osp
import numpy as np
from glob import glob
import cv2
import time


def match_SIFT(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create(500)
    start = time.time()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    # H, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC, 5.0)
    H,mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    sx = np.sign(H[0,0])*np.sqrt(H[0,0]**2 + H[0,1]**2)
    sy = np.sign(H[1,1])*np.sqrt(H[1,0]**2 + H[1,1]**2)
    s = (sx+sy)/2
    if abs(1 - s) > 0.1:
        raise ValueError('Scaling size changes too much during rigid alignment !')
    H[0:2,0:2]/=s
    H = np.vstack((H,np.array([[0,0,1]])))
    return H,src_pts,dst_pts

def regist_sift(image1, image2):
    H,src_pts,dst_pts = match_SIFT(image1, image2)
    imgOut = cv2.warpPerspective(image2, H, (image1.shape[1],image1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    return imgOut,src_pts,dst_pts
    
def mosaic(image1, image2, mask1):
    H = match_SIFT(image1, image2)
    maskOut = cv2.warpPerspective(mask0.copy(), H, (image1.shape[1],image1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    maskOut[mask1==1] = 0
    imgOut = cv2.warpPerspective(image2, H, (image1.shape[1],image1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    imgOut = imgOut * (maskOut==1)
    image1[mask1==0] = 0
    maskOut[mask1==1] = 1
    imgOut = imgOut + image1
    imgOut[maskOut==0] = 255
    return imgOut, maskOut

if __name__ == "__main__":
    img_path1 = 'D:/hisign/partial_fingerprint/160160/1_11/1_11.bmp'
    img_path2 = 'D:/hisign/partial_fingerprint/160160/1_11/1_11_1.bmp'
    img_path3 = 'D:/hisign/partial_fingerprint/160160/1_11/1_11_2.bmp'
    img_path4 = 'D:/hisign/partial_fingerprint/160160/1_11/1_11_3.bmp'
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(img_path3, cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread(img_path4, cv2.IMREAD_GRAYSCALE)
    img1 = np.pad(img1, pad_width=((160, 160), (160, 160)), constant_values=255)
    img2 = np.pad(img2, pad_width=((160, 160), (160, 160)), constant_values=255)
    img3 = np.pad(img3, pad_width=((160, 160), (160, 160)), constant_values=255)
    img4 = np.pad(img4, pad_width=((160, 160), (160, 160)), constant_values=255)
    mask0 = np.zeros((480, 480))
    mask0[160:320,160:320] = 1
    img12, mask12 = mosaic(img1, img2, mask0)
    img123, mask123 = mosaic(img12, img3, mask12)
    img1234, mask1234 = mosaic(img123, img4, mask123)
    cv2.imshow('123', img1234)
    cv2.waitKey(0)
