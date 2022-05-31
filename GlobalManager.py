"""
This file (GlobalManager.py) is designed for:
    
Copyright (c) 2022, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob


def _init():  # 初始化
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    """定义一个全局变量"""
    _global_dict[key] = value


def get_value(key, defValue=None):
    """获得一个全局变量,不存在则返回默认值"""
    try:
        return _global_dict[key]
    except KeyError:
        return defValue
