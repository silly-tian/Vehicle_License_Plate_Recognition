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
                # д��ǰ���뽫�ļ����Ѿ�������
                imagei = image[i]
                cv2.imwrite(end_path + '/' + file_pathname + '/' + filenew_name[0] + '/' + filenew_name[0] + '_'+str(i)+'.jpg', imagei)


def creat_path(file_pathname_all, end_path):
    os.chdir(end_path)  #  ��Ŀ���ļ��е�·��
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
        # ([26, 43, 46], [34, 255, 255]),  # ��ɫ
        # ([100, 43, 46], [124, 255, 255]),  # ��ɫ
        ([100, 100, 50], [124, 255, 255])
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


def card_division(src):
    src2 = src.copy()
    # sp[0]�ǿ���sp[1]������
    sp = src.shape
    # ��ɫͨ��ȡ��ȫͼ��ɫ�Ĳ��֣�����֮��Ĳ���
    src2 = color_position(src2)
    # ��˹ģ����ȡ��֥���������
    Gauss = cv2.GaussianBlur(src2, (11, 11), 0)
    # ת��Ϊ�Ҷ�ͼ
    gray = cv2.cvtColor(Gauss, cv2.COLOR_BGR2GRAY)
    # sobel���ӱ�Ե���
    Sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)
    image = absX.copy()
    # ȥ��С�İ׵���������
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    # ���ͣ���ʴ
    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)
    # ��ʴ������
    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)
    # ��ֵ�˲�ȥ�����
    # image = cv2.medianBlur(image, 15)
    # ת��Ϊ��ֵͼ��������findcontours
    ret, binary = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    # �������
    # cv2.RETR_EXTERNAL��ʾֻ���������
    # cv2.CHAIN_APPROX_SIMPLEѹ��ˮƽ���򣬴�ֱ���򣬶Խ��߷����Ԫ�أ� ֻ�����÷�����յ����꣬
    # ����һ����������ֻ��Ҫ�ĸ���������������Ϣ
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ��������
    imagel = src.copy()
    cv2.drawContours(imagel, contours, -1, (0, 0, 255), 2)
    #ɸѡ������λ�õ�����
    # s = []
    # for item in contours:
    #     # cv2.boundingRect��һ����С�ľ��Σ�������״������
    #     rect = cv2.boundingRect(item)
    #     x = rect[0]
    #     y = rect[1]
    #     weight = rect[2]
    #     height = rect[3]
    #     if (weight > (height * 1.2)) and (weight < (height * 6)):
    #         s.append(weight * height)
    # for item in contours:
    #     # cv2.boundingRect��һ����С�ľ��Σ�������״������
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







