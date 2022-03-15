# coding=gbk

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class retreat(object):
    def __init__(self, image):
        self.image = image
    def color_position(image):
        colors = [
            # ([26, 43, 46], [34, 255, 255]),  # 黄色
            ([100, 100, 50], [124, 255, 255]),  # 蓝色
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


    def re_treat_card(src):
        # 首先颜色过滤
        src2 = src.copy()
        # src2 = src2[int(sp[0]/10):int(sp[0]*4/5):, :int(sp[1]*2/5)]
        # src2 = color_position(src2)
        Gauss = cv2.GaussianBlur(src2, (5, 5), 0)
        # 转换为灰度图
        gray = cv2.cvtColor(Gauss, cv2.COLOR_BGR2GRAY)
        # sobel算子边缘检测
        Sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        absX = cv2.convertScaleAbs(Sobel_x)
        image = absX.copy()
        # 转化为二值图象，以用于findcontours
        ret, binary = cv2.threshold(image, 55, 255, cv2.THRESH_BINARY)
        kerne2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kerne2)
        dst = cv2.medianBlur(closing, 7)
        # 去除小的白点来创建的
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        # kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        # 膨胀，腐蚀
        image = cv2.dilate(dst, kernelX)
        image = cv2.erode(image, kernelX)
        # 轮廓检测
        # cv2.RETR_EXTERNAL表示只检测外轮廓
        # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素， 只保留该方向的终点坐标，
        # 例如一个矩形轮廓只需要四个点来保存轮廓信息
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        imagel = src.copy()
        cv2.drawContours(imagel, contours, -1, (0, 0, 255), 2)
        s = []
        for item in contours:
            # cv2.boundingRect用一个最小的矩形，把新形状包起来
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            if (weight > (height * 1.2)) and (weight < (height * 6)):
                s.append(weight * height)
        for item in contours:
            # cv2.boundingRect用一个最小的矩形，把新形状包起来
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            if (weight > (height * 1.2)) and (weight < (height * 6)) and weight*height==max(s):
                image = src[y:y + height, x:x + weight]
                break
        return image

