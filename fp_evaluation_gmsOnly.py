'''
Date: 2021-11-13 15:55:21
LastEditTime: 2022-09-20 17:30:22
Description:  
Autor: Guan Xiongjun
LastEditors: Please set LastEditors
'''
import numpy as np
import matplotlib.pyplot as plt
import math

def DrawROC_gmsOnly(ax,gms,linewidth,color,linestyle='-',label=None):
    FAR = np.arange(-8,0.5,0.2)
    threshold = -FAR*12
    TAR = np.zeros((len(FAR),1))
    for i in range(len(FAR)):
        TAR[i] = sum(gms>threshold[i])/len(gms)
    
    ax.plot(np.power(10,FAR),TAR,linewidth=linewidth,color=color,linestyle=linestyle,label=label)
 
    
    ax.set_xscale("log")
    ax.set_xlabel("FNR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve using Verifinger")

def DrawDET_gmsOnly(ax,gms,linewidth,color,linestyle='-',label=None):
    FAR = np.arange(-8,0.5,0.2)
    
    threshold = -FAR*12

    TAR = np.zeros((len(FAR),1))
    for i in range(len(FAR)):
        TAR[i] = (sum(gms>threshold[i]))/len(gms)
    
    FMR = np.power(10,FAR)
    FNMR = (1-TAR).reshape((-1,))
    

    ax.plot(FMR,FNMR,linewidth=linewidth,color=color,linestyle=linestyle,label=label)

    # # EER line
    # points = np.array([-4,1])
    # ax.plot(points,points,linewidth=linewidth,linestyle='-',color = 'k')

    # # draw wide grid lines
    # for i in range(-3,1):
    #     y = np.array([math.pow(10,i),math.pow(10,i)])
    #     x = np.array([math.pow(10,-5),math.pow(10,0)])
    #     ax.plot(x,y,linewidth=linewidth,linestyle='-',color = 'k')
    # for j in range(-4,1):
    #     y = np.array([math.pow(10,-4),math.pow(10,0)])
    #     x = np.array([math.pow(10,j),math.pow(10,j)])
    #     ax.plot(x,y,linewidth=linewidth,linestyle='-',color = 'k')
    

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("FMR",fontsize=12)
    ax.set_ylabel("FNMR",fontsize=12)
    ax.minorticks_on()
    
    ax.grid(which='minor', axis='both', linestyle='--',linewidth=0.5)

    # ax.text(np.power(10,-3.95),np.power(10,-1.75),'FMR10000',color = "darkred",rotation=90)
    # ax.text(np.power(10,-2.95),np.power(10,-1.75),'FMR1000',color = "darkred",rotation=90)
    # ax.text(np.power(10,-1.95),np.power(10,-1.75),'FMR100',color = "darkred",rotation=90)
    # ax.text(np.power(10,-1.65),np.power(10,-1.45),'EER',color = "darkred",rotation=45)
    
    ax.set_title("DET curve using Verifinger")
