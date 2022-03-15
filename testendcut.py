# coding=gbk

import os
import cv2
import numpy as np
from re_treat import retreat
from Affine_transformation import Affine
import re
from skimage import io as iio


def cut_char(src):
    # ��˹ȥ��
    image = cv2.GaussianBlur(src, (3, 3), 0)
    # �Ҷȴ���
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    rows = image.shape[0]
    cols = image.shape[1]
    # ��ֵͳ��,ͳ��ûÿһ�еĺ�ֵ��0���ĸ���
    hd = []
    for row in range(rows):
        res = 0
        for col in range(cols):
            if image[row][col] != 0:
                res = res + 1
        hd.append(res)
    # ����һ���㷨,�ҵ�����,��λ�����ַ�����������
    # �ҵ�˼·;����һ������,�м�λ�ÿ϶����о��ȵĺ�ɫ���,�����ҽ�ͼƬ��ֱ��Ϊ������,�Ҳ���
    mean = sum(hd[0:int(rows / 2)]) / (int(rows / 2) + 1)
    region = []
    for i in range(int(rows / 2), 0, -1):  # 0,1�п϶��Ǳ߿�,ֱ�Ӳ�����,ֱ�Ӵӵڶ��п�ʼ
        if hd[i] < mean / 2:
            region.append(i)
            break
        else:
            region.append(i)
    for i in range(int(rows / 2), rows):  # 0,1�п϶��Ǳ߿�,ֱ�Ӳ�����,ֱ�Ӵӵڶ��п�ʼ
        if hd[i] < mean / 2:
            region.append(i)
            break
        else:
            region.append(i)
    image1 = image[min(region):max(region), :]  # ʹ��������
    image1[:, :1] = 0
    image1[:, -1:] = 0
    image11 = image1.copy()
    image11.shape  # 47�У�170��
    rows = image11.shape[0]
    cols = image11.shape[1]
    # ��ֵͳ��,ͳ��ûÿһ�еĺ�ֵ��0���ĸ���
    hd1 = []
    for col in range(cols):
        res = 0
        for row in range(rows):
            if image11[row][col] == 255:
                res = res + 1
        hd1.append(res)
    # �����в�Ϊ0������(����)
    region1 = []
    reg = []
    for i in range(cols - 1):
        if hd1[i] == 0 and hd1[i + 1] != 0:
            reg.append(i)
        if hd1[i] != 0 and hd1[i + 1] == 0:
            reg.append(i + 2)
        if len(reg) == 2:

            if (reg[1] - reg[0]) > 7:  # �޶����䳤��Ҫ����5(���Ը���),���˵�����Ҫ�ĵ�
                region1.append(reg)
                reg = []
            else:
                reg = []
    # ȷ���ַ�
    endlist = []
    endlen = int(src.shape[1]/7*1.3)
    if len(region1) != 7:
        j = 0
        for i in range(len(region1)):
            if region1[i][1] - region1[i][0] > endlen:
                mean = abs(7 - len(region1)) + 1
                lis_1 = [region1[i][0], int(region1[i][0] + (region1[i][1] - region1[i][0]) / mean)]
                lis_2 = [int(region1[i][0] + 2 + (region1[i][1] - region1[i][0]) / mean), region1[i][1]]
                endlist.append(lis_1)
                endlist.append(lis_2)
            else:
                endlist.append(region1[i])
    else:
        endlist = region1.copy()
    if len(endlist) != 0:
        if endlist[-1][1] - endlist[-1][0] > int(region1[-1][-1] / 7):
            endlist[-1] = [endlist[-1][0], endlist[-1][0] + int((endlist[-1][1] - endlist[-1][0]) * 2 / 3)]
    # �洢�ַ����ļ�
    image_all = []
    for i in range(len(endlist)):
        image2 = image1[:, endlist[i][0]:endlist[i][1]]
        image_all.append(image2)
    return image_all


def creat_path(file_pathname_all, end_path):
    os.chdir(end_path)
    for file_pathname in os.listdir(file_pathname_all):
        path = file_pathname
        if not os.path.exists(path):
            os.makedirs(path)



# def write_to_path(file_pathname_all, end_path):
#     for file_pathname in os.listdir(file_pathname_all):
#         for file_name in os.listdir(file_pathname_all+'/'+file_pathname):
#             # print(file_pathname_all+'/'+file_pathname+'/'+file_name)
#             filenew_name = re.findall("(.*)\.jpg", file_name)
#             image = cv2.imread(file_pathname_all+'/'+file_pathname+'/'+file_name)
#             # cv2.imshow('image', image)
#             # cv2.waitKey()


def read_path(file_pathname_all, end_path):
    for file in os.listdir(file_pathname_all):
        print(file)
        image = iio.imread(file_pathname_all + '/' + file)
        if image is None:
            continue
        image = retreat.re_treat_card(image)
        # cv2.imshow("789789", image)
        # cv2.waitKey()
        cv2.imwrite(end_path + '/' + file, image)
        image = Affine.affine_transform(image)
        if len(image.shape)==2:
            continue

        image = cut_char(image)
        if len(image)<7:
            continue
        else:
            print(file)
            for i in range(len(image)):
                cv2.imwrite(end_path + '/' + file + '/' + str(i) + '.jpg', image[i])


filepath = "D:/Pycharm_Project/Vehicle_license_plate_recognition/end1"
endpath = "D:/Pycharm_Project/Vehicle_license_plate_recognition/affine_trans_end1"

#
creat_path(filepath, endpath)
read_path(filepath, endpath)

# image = r'C:\Users\15098\Desktop\15311020468_0.jpg'
# image = retreat.re_treat_card(image)
# # cv2.imshow("123456", image)
# # cv2.waitKey()
# image = Affine.affine_transform(image)
# # cv2.imshow("789789", image)
# # cv2.waitKey()
# if len(image.shape) == 2:
#     print("error")
# image = cut_char(image)
# if len(image) < 7:
#     print("error2")
# else:
#     print(image)
#     for i in range(len(image)):
#         cv2.imshow("1111", i)
#         cv2.waitKey()


