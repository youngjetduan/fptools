"""
This file (fp_indexing.py) is designed for:
    fingerprint indexing algorithm, refer to Su's PR paper (Fingerprint indexing with pose constraint)
Copyright (c) 2022, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
from ctypes import cdll
import subprocess
from xml.etree.ElementTree import ElementTree, Element, tostring

# server 27
# neu_dir = "/mnt/data5/fptools/Verifinger"
# server 33
# neu_dir = "/mnt/data1/dyj"
neu_dir = np.loadtxt(osp.join(osp.dirname(osp.abspath(__file__)), "neu_dir.txt"), str).tolist()


def create_xml_element(key, val=None):
    child = Element(key)
    if val is not None:
        if isinstance(val, Element):
            child.append(val)
        else:
            child.text = str(val)
    return child


def pretty_xml(element, indent="\t", newline="\n", level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def __indent(elem, level=0):
    ii = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = ii
        if not elem.tail or not elem.tail.strip():
            elem.tail = ii
        for e in elem:
            __indent(e, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = ii
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = ii


def mcc_indexing(gallery_db, query_db, mnt_dir, mcc_dir, pose_dir, out_name, delta_xy=200, delta_a=40):
    parameter = Element("IndexingParameter")
    parameter.append(create_xml_element("Rank", f"{out_name}_xy{delta_xy}_a{delta_a}.txt"))
    parameter.append(create_xml_element("DBTit", gallery_db))
    parameter.append(create_xml_element("QueryTit", query_db))
    parameter.append(create_xml_element("MinuPath", mnt_dir))
    parameter.append(create_xml_element("FeatPath", mcc_dir))
    parameter.append(create_xml_element("Pose", create_xml_element("Path", pose_dir)))
    parameter.append(create_xml_element("HashFunc", osp.join(neu_dir, "MCCIndexing", "hashfuncs.txt")))
    parameter.append(create_xml_element("DeltaXY", delta_xy))
    parameter.append(create_xml_element("DeltaTheta", delta_a))
    parameter.append(create_xml_element("IsICF", 0))
    parameter.append(create_xml_element("TimeAccuracy", 0))
    parameter.append(create_xml_element("IsWait", 1))
    parameter.append(create_xml_element("PM_NBit", 312))
    parameter.append(create_xml_element("PM_NByte", 39))
    parameter.append(create_xml_element("PM_P", 30))
    parameter.append(create_xml_element("PM_MINPC", 2))
    parameter.append(create_xml_element("PM_H", 24))
    parameter.append(create_xml_element("PM_L", 32))
    xml_tree = ElementTree(parameter)
    # __indent(parameter)
    pretty_xml(parameter)
    xml_tree.write(f"{out_name}_xy{delta_xy}_a{delta_a}.xml", encoding="utf-8", xml_declaration=True)

    # ret = subprocess.run([osp.join(neu_dir, "MCCIndexing", "mcc_indexing"), f"{out_name}_xy{delta_xy}_a{delta_a}.xml"])
    # return ret
