'''
Descripttion: 
version: 
Author: Xiongjun Guan
Date: 2021-08-11 11:41:33
LastEditors: Xiongjun Guan
LastEditTime: 2021-08-11 11:43:10
'''
import os
from PtFunctions import PtSegmentation, ComputePtData
import multiprocessing
import tqdm
import time
import argparse


def ComputePtData_mp(input):
    PtSegmentation(path=input[1] + input[0], type='mat', save_dir=input[2], save_name=None)
    ComputePtData(path=input[2] + input[0], type='mat', r=0.15, save_dir=input[3], save_name=None, rotate=input[4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="data dir")
    parser.add_argument("tmp_dir", type=str, help="tmp dir")
    parser.add_argument("save_dir", type=str, help="save dir")
    parser.add_argument("--pitch", '-p', type=int, default="0", help="rotate angle: pitch")
    parser.add_argument("--yaw", '-y', type=int, default="180", help="rotate angle: yaw")
    parser.add_argument("--roll", '-r', type=int, default="-90", help="rotate angle: roll")
    args = parser.parse_args()

    # data_dir = '/mnt/data1/hefei_data/3d_processed/pt_stl2mat/GOOD/'
    # tmp_dir = '/mnt/data1/hefei_data/3d_processed/pt_seg/GOOD/'
    # save_dir = '/mnt/data1/hefei_data/3d_processed/pt_processed/GOOD/'

    data_dir = args.data_dir
    tmp_dir = args.tmp_dir
    save_dir = args.save_dir
    rotate = [args.pitch, args.yaw, args.roll]

    f_list = os.listdir(data_dir)
    paths = list([])
    for fi in f_list:
        if os.path.splitext(fi)[1] == '.mat':
            paths.append((fi, data_dir, tmp_dir, save_dir, rotate))

    # ComputePtData_mp(paths=[data_dir + '90_1_1.mat', save_dir, [0, 180, -90]])

    print('data_dir: ' + data_dir)
    print('tmp_dir: ' + tmp_dir)
    print('save_dir: ' + save_dir)
    print('rotate: [%d, %d, %d]' % (rotate[0], rotate[1], rotate[2]))
    print('data_num: ' + str(len(paths)))

    num_processes = round(multiprocessing.cpu_count() * 0.5)
    with multiprocessing.Pool(num_processes) as pool:
        r = list(
            tqdm.tqdm(pool.imap(ComputePtData_mp, paths), mininterval=5, maxinterval=30, total=len(paths)))
#     # pool.close()
#     # pool.join()
