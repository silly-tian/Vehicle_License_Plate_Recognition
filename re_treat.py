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
            # ([26, 43, 46], [34, 255, 255]),  # ��ɫ
            ([100, 100, 50], [124, 255, 255]),  # ��ɫ
            # ([35, 43, 46], [77, 255, 255])   # ��ɫ
        ]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for (lower, upper) in colors:
            lower = np.array(lower, dtype="uint8")  # ��ɫ����
            upper = np.array(upper, dtype="uint8")  # ��ɫ����

            # ������ֵ�ҵ���Ӧ����ɫ
            mask = cv2.inRange(hsv, lowerb=lower, upperb=upper)
            output = cv2.bitwise_and(image, image, mask=mask)
            # cv2.imshow("image", img)
            # cv2.imshow("image_color", output)
            # cv2.waitKey(0)
        return output


    def re_treat_card(src):
        # ������ɫ����
        src2 = src.copy()
        # src2 = src2[int(sp[0]/10):int(sp[0]*4/5):, :int(sp[1]*2/5)]
        # src2 = color_position(src2)
        Gauss = cv2.GaussianBlur(src2, (5, 5), 0)
        # ת��Ϊ�Ҷ�ͼ
        gray = cv2.cvtColor(Gauss, cv2.COLOR_BGR2GRAY)
        # sobel���ӱ�Ե���
        Sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        absX = cv2.convertScaleAbs(Sobel_x)
        image = absX.copy()
        # ת��Ϊ��ֵͼ��������findcontours
        ret, binary = cv2.threshold(image, 55, 255, cv2.THRESH_BINARY)
        kerne2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kerne2)
        dst = cv2.medianBlur(closing, 7)
        # ȥ��С�İ׵���������
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        # kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        # ���ͣ���ʴ
        image = cv2.dilate(dst, kernelX)
        image = cv2.erode(image, kernelX)
        # �������
        # cv2.RETR_EXTERNAL��ʾֻ���������
        # cv2.CHAIN_APPROX_SIMPLEѹ��ˮƽ���򣬴�ֱ���򣬶Խ��߷����Ԫ�أ� ֻ�����÷�����յ����꣬
        # ����һ����������ֻ��Ҫ�ĸ���������������Ϣ
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # ��������
        imagel = src.copy()
        cv2.drawContours(imagel, contours, -1, (0, 0, 255), 2)
        s = []
        for item in contours:
            # cv2.boundingRect��һ����С�ľ��Σ�������״������
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            if (weight > (height * 1.2)) and (weight < (height * 6)):
                s.append(weight * height)
        for item in contours:
            # cv2.boundingRect��һ����С�ľ��Σ�������״������
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            if (weight > (height * 1.2)) and (weight < (height * 6)) and weight*height==max(s):
                image = src[y:y + height, x:x + weight]
                break
        return image

