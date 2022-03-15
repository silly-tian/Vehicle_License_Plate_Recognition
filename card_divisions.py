# coding=gbk

import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt


def write_to_path(file_pathname_all, end_path):
    for file_pathname in os.listdir(file_pathname_all):
        for file_name in os.listdir(file_pathname_all+'/'+file_pathname):
            # print(file_pathname_all+'/'+file_pathname+'/'+file_name)
            filenew_name = re.findall("(.*)\.jpg", file_name)
            image = cv2.imread(file_pathname_all+'/'+file_pathname+'/'+file_name)
            # cv2.imshow('image', image)
            # cv2.waitKey()
            image = card_division(image)
            for i in range(len(image)):
                # 写入前必须将文件夹已经创建好
                imagei = image[i]
                cv2.imwrite(end_path + '/' + file_pathname + '/' + filenew_name[0] + '/' + filenew_name[0] + '_'+str(i)+'.jpg', imagei)


def creat_path(file_pathname_all, end_path):
    os.chdir(end_path)  #  打开目标文件夹的路径
    for file_pathname in os.listdir(file_pathname_all):
        path = file_pathname
        if not os.path.exists(path):
            os.makedirs(path)
        for file_name in os.listdir(file_pathname_all + '/' + file_pathname):
            filenew_name = re.findall("(.*)\.jpg", file_name)
            if not os.path.exists(file_pathname + "/" + filenew_name[0]):
                os.makedirs(file_pathname + "/" + filenew_name[0])



def color_position(image):
    colors = [
        # ([26, 43, 46], [34, 255, 255]),  # 黄色
        # ([100, 43, 46], [124, 255, 255]),  # 蓝色
        ([100, 100, 50], [124, 255, 255])
        # ([35, 43, 46], [77, 255, 255])   # 绿色
    ]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for (lower, upper) in colors:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到相应的颜色
        mask = cv2.inRange(hsv, lowerb=lower, upperb=upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("image", img)
        # cv2.imshow("image_color", output)
        # cv2.waitKey(0)
    return output


def card_division(src):
    src2 = src.copy()
    # sp[0]是宽方向；sp[1]长方向
    sp = src.shape
    # 颜色通道取出全图蓝色的部分，方便之后的操作
    src2 = color_position(src2)
    # 高斯模糊，取出芝麻粒类噪点
    Gauss = cv2.GaussianBlur(src2, (11, 11), 0)
    # 转换为灰度图
    gray = cv2.cvtColor(Gauss, cv2.COLOR_BGR2GRAY)
    # sobel算子边缘检测
    Sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)
    image = absX.copy()
    # 去除小的白点来创建的
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    # 膨胀，腐蚀
    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)
    # 腐蚀，膨胀
    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)
    # 中值滤波去除噪点
    # image = cv2.medianBlur(image, 15)
    # 转化为二值图象，以用于findcontours
    ret, binary = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    # 轮廓检测
    # cv2.RETR_EXTERNAL表示只检测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素， 只保留该方向的终点坐标，
    # 例如一个矩形轮廓只需要四个点来保存轮廓信息
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    imagel = src.copy()
    cv2.drawContours(imagel, contours, -1, (0, 0, 255), 2)
    #筛选出车牌位置的轮廓
    # s = []
    # for item in contours:
    #     # cv2.boundingRect用一个最小的矩形，把新形状包起来
    #     rect = cv2.boundingRect(item)
    #     x = rect[0]
    #     y = rect[1]
    #     weight = rect[2]
    #     height = rect[3]
    #     if (weight > (height * 1.2)) and (weight < (height * 6)):
    #         s.append(weight * height)
    # for item in contours:
    #     # cv2.boundingRect用一个最小的矩形，把新形状包起来
    #     rect = cv2.boundingRect(item)
    #     x = rect[0]
    #     y = rect[1]
    #     weight = rect[2]
    #     height = rect[3]
    #     if (weight > (height * 1.2)) and (weight < (height * 6)) and weight*height==max(s):
    #         image = src[y:y + height, x:x + weight]

    image_all = []
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if (weight > (height * 1.2)) and (weight < (height * 6)):
            if x - 30 < 0 or x + weight + 30 > imagel.shape[1]:
                image = src[y:y + height, x:x + weight]
            else:
                image = src[y:y + height, x - 30:x + weight + 30]
            image_all.append(image)
    return image_all


filepath = "D:/Pycharm_Project/Vehicle_license_plate_recognition/cardsneed"
endpath = "D:/Pycharm_Project/Vehicle_license_plate_recognition/division_end"
creat_path(filepath, endpath)
write_to_path(filepath, endpath)







