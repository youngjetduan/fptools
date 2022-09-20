'''
Description: 
Author: Guan Xiongjun
Date: 2022-08-31 17:04:04
LastEditTime: 2022-09-20 17:31:42
'''
import os
import os.path as osp
import shutil
import scipy.io as scio
import argparse
import sys
import cv2
import time
import numpy as np
from tqdm import tqdm
from glob import glob
from skimage import morphology

sys.path.append(osp.dirname(osp.abspath(__file__)))
from BasePhase import PhaseRegistration
from BaseFeature import PadImage, ExtractFeature
from Base import ShowRegisteredImg
from fp_segmtation import segmentation_coherence
from fp_mosaic import mosaic_line
from fp_enhancement import localEqualHist
from fp_verifinger import Verifinger,load_minutiae

def phase_match_mosaic_single(ftitle1,ftitle2,img_dir,tmp_init_dir,tmp_register_dir,tool,need_enh=False,ext="png"):
    if not osp.exists(tmp_init_dir):
        os.makedirs(tmp_init_dir)
    if not osp.exists(tmp_register_dir):
        os.makedirs(tmp_register_dir)


    init_prefix = "init_"
    phase_prefix = "phase_"

    tmp_init_init_dir = osp.join(tmp_init_dir,'init')
    tmp_init_feature_dir = osp.join(tmp_init_dir,'feature')

    if not osp.exists(tmp_init_init_dir):
        os.makedirs(tmp_init_init_dir)
    if not osp.exists(tmp_init_feature_dir):
        os.makedirs(tmp_init_feature_dir)
        
    shutil.copy(osp.join(img_dir,ftitle1+'.'+ext),osp.join(tmp_init_feature_dir,ftitle1+'.'+ext))
    shutil.copy(osp.join(img_dir,ftitle2+'.'+ext),osp.join(tmp_init_feature_dir,ftitle2+'.'+ext))
 
    # --------------- feature extraction : begin --------------- #
    ExtractFeature(tmp_init_feature_dir, tmp_init_feature_dir, ftitle1, tool, ext,vf=True,phase=True)
    ExtractFeature(tmp_init_feature_dir, tmp_init_feature_dir, ftitle2, tool, ext,vf=True,phase=False)
    # --------------- feature extraction : end --------------- #

    # --------------- phase match : begin --------------- #
    img_phase, dx, dy = PhaseRegistration(
        tmp_init_feature_dir,
        tmp_init_init_dir,
        ftitle1,
        ftitle2,
        tool,
        ext,
        init_prefix,
        phase_prefix,
        initFunc="VeriFinger",
        threshNum=7,
        RANSAC_th=15,
    )  
    
        
    img_phase2_title = ftitle2
    cv2.imwrite(osp.join(tmp_register_dir, img_phase2_title + "." + ext), img_phase)
    register_title = ftitle2

    # --------------- phase match : end--------------- #


    # --------------- mosaic : begin--------------- #
    img1 = cv2.imread(osp.join(tmp_init_feature_dir,ftitle1+'.'+ext),0)
    bimg1 = cv2.imread(osp.join(tmp_init_feature_dir,'b'+ftitle1+'.'+ext),0)
    mask1 = segmentation_coherence(bimg1,stride=8)
    selem = np.ones((15, 15))
    mask1 = morphology.binary_erosion(mask1.astype(bool), selem)
    mask1 = morphology.binary_erosion(mask1.astype(bool), selem)

    img2 = cv2.imread(osp.join(tmp_register_dir,ftitle2+'.'+ext),0)
    tool.binary_extraction(tmp_register_dir, ftitle2, tmp_register_dir, "b" + register_title, ext)
    bimg2 = cv2.imread(osp.join(tmp_register_dir,'b'+ftitle2+'.'+ext),0)
    mask2 = segmentation_coherence(bimg2,stride=8)
    mask2 = morphology.binary_erosion(mask2.astype(bool), selem)
    mask2 = morphology.binary_erosion(mask2.astype(bool), selem)

    img,mask1_crop,mask2_crop = mosaic_line(img1,img2,bimg1,bimg2,mask1,mask2)
    # --------------- mosaic : end--------------- #

    # enh
    if need_enh is True:
        img1 = localEqualHist(img1)
        img2 = localEqualHist(img2)
        img_enh = img1 * mask1_crop + img2 * mask2_crop
        blk_mask = 1-((mask1_crop>0) | (mask2_crop>0))
        img_enh[blk_mask>0] = 255
        return img,img_enh,dx,dy
    else:
        return img,dx,dy

def affine_match_mosaic_single(ftitle1,ftitle2,img_dir,tmp_init_dir,tmp_register_dir,tool,need_enh=False,ext="png"):
    if not osp.exists(tmp_init_dir):
        os.makedirs(tmp_init_dir)
    if not osp.exists(tmp_register_dir):
        os.makedirs(tmp_register_dir)

    # --------------- feature extraction : begin --------------- #
    shutil.copy(osp.join(img_dir,ftitle1+'.'+ext),osp.join(tmp_init_dir,ftitle1+'.'+ext))
    shutil.copy(osp.join(img_dir,ftitle2+'.'+ext),osp.join(tmp_init_dir,ftitle2+'.'+ext))

    tool.minutia_extraction(tmp_init_dir, ftitle1, tmp_init_dir, "mf" + ftitle1, ext)
    tool.binary_extraction(tmp_init_dir, ftitle1, tmp_init_dir, "b" + ftitle1, ext)
    tool.minutia_extraction(tmp_init_dir, ftitle2, tmp_init_dir, "mf" + ftitle2, ext)
    
    score, init_minu_pairs = tool.fingerprint_matching_single(
        tmp_init_dir, "mf" + ftitle1, tmp_init_dir, "mf" + ftitle2
    )
    MINU1 = load_minutiae(osp.join(tmp_init_dir, "mf" + ftitle1 + ".mnt"))
    MINU2 = load_minutiae(osp.join(tmp_init_dir, "mf" + ftitle2 + ".mnt"))
    # --------------- feature extraction : end --------------- #

    # --------------- affine match : begin--------------- #
    if init_minu_pairs.shape[0] < 5:
        raise ValueError("Too few match minutiae pairs")

    H, mask = cv2.estimateAffinePartial2D(MINU2[init_minu_pairs[:, 1], 0:2], MINU1[init_minu_pairs[:, 0], 0:2],method=cv2.RANSAC, ransacReprojThreshold=15.0)
    sx = np.sign(H[0,0])*np.sqrt(H[0,0]**2 + H[0,1]**2)
    sy = np.sign(H[1,1])*np.sqrt(H[1,0]**2 + H[1,1]**2)
    s = (sx+sy)/2
    if abs(1 - s) > 0.15:
        raise ValueError('Scaling size changes too much during rigid alignment !')
    mask = mask.reshape((-1,))
    init_minu_pairs = init_minu_pairs.take(np.where(mask==1)[0],0)

    img2 = cv2.imread(osp.join(tmp_init_dir,ftitle2+'.'+ext),0)
    img2_affine = cv2.warpAffine(img2,H,(img2.shape[1],img2.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    
    cv2.imwrite(osp.join(tmp_register_dir, ftitle2 + "." + ext), img2_affine)

    # --------------- affine match : end--------------- #


    # --------------- mosaic : begin--------------- #
    img1 = cv2.imread(osp.join(tmp_init_dir,ftitle1+'.'+ext),0)
    bimg1 = cv2.imread(osp.join(tmp_init_dir,'b'+ftitle1+'.'+ext),0)
    mask1 = segmentation_coherence(img1,stride=8)
    selem = np.ones((15, 15))
    mask1 = morphology.binary_erosion(mask1.astype(bool), selem)
    mask1 = morphology.binary_erosion(mask1.astype(bool), selem)

    img2 = cv2.imread(osp.join(tmp_register_dir,ftitle2+'.'+ext),0)
    tool.binary_extraction(tmp_register_dir, ftitle2, tmp_register_dir, "b" + ftitle2, ext)
    bimg2 = cv2.imread(osp.join(tmp_register_dir,'b'+ftitle2+'.'+ext),0)
    mask2 = segmentation_coherence(img2,stride=8)
    mask2 = morphology.binary_erosion(mask2.astype(bool), selem)
    mask2 = morphology.binary_erosion(mask2.astype(bool), selem)

    img,mask1_crop,mask2_crop = mosaic_line(img1,img2,bimg1,bimg2,mask1,mask2)
    # --------------- mosaic : end--------------- #

    # enh
    if need_enh is True:
        img1 = localEqualHist(img1)
        img2 = localEqualHist(img2)
        img_enh = img1 * mask1_crop + img2 * mask2_crop
        blk_mask = 1-((mask1_crop>0) | (mask2_crop>0))
        img_enh[blk_mask>0] = 255
        
        return img,img_enh,H
    else:
        return img,H
    
if __name__ == "__main__":
    img_dir = "/disk1/guanxiongjun/Hisign_data/big/lst_all/img/"
    tmp_init_dir = "/disk1/guanxiongjun/tmp/tmp/"
    tmp_register_dir = "/disk1/guanxiongjun/tmp/register/"

    result_dir = "/disk1/guanxiongjun/tmp/result/"
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    ext = "png"

    ftitle1 = '1m_X0_L1_L'
    ftitle2 = '1m_X0_L1_M'
    save_title = ftitle1 + "_with_" + ftitle2 

    tool = Verifinger()
    img,img_enh,dx,dy = phase_match_mosaic_single(ftitle1,ftitle2,img_dir,tmp_init_dir,tmp_register_dir,tool,need_enh=True,ext="png")
    cv2.imwrite(osp.join(result_dir,save_title+'.png'),img)
    cv2.imwrite(osp.join(result_dir,save_title+'_enh.png'),img_enh)