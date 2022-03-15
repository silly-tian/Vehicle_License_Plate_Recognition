# coding=gbk

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def write_to_path(file_pathname_all, end_path):
    for file_pathname in os.listdir(file_pathname_all):
        for file_name in os.listdir(file_pathname_all+'/'+file_pathname):
            print(file_pathname_all+'/'+file_pathname+'/'+file_name)
            image = cv2.imread(file_pathname_all+'/'+file_pathname+'/'+file_name)
            # cv2.imshow('image', image)
            # cv2.waitKey()
            image = card_rotate(image)
            # 写入前必须将文件夹已经创建好
            cv2.imwrite(end_path + '/' + file_pathname + '/' + file_name, image)


def creat_path(file_pathname_all, end_path):
    os.chdir(end_path)  #  打开目标文件夹的路径
    for file_pathname in os.listdir(file_pathname_all):
        path = file_pathname
        if not os.path.exists(path):
            os.makedirs(path)


def card_rotate(image):
    s = 123