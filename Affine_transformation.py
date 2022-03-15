# coding=gbk

import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt





#
#             #   #   #   #   #   #
#             image = card_division(image)
#             for i in range(len(image)):
#                 # д��ǰ���뽫�ļ����Ѿ�������
#                 imagei = image[i]
#                 cv2.imwrite(end_path + '/' + file_pathname + '/' + filenew_name[0] + '/' + filenew_name[0] + '_'+str(i)+'.jpg', imagei)
# ???


# def read_path(file_pathname_all, end_path):
#     for file_pathname in os.listdir(file_pathname_all):
#         for file_name in os.listdir(file_pathname_all + '/' + file_pathname):
#             for file in os.listdir(file_pathname_all + '/' + file_pathname + '/' +file_name):
#                 image = cv2.imread(file_pathname_all + '/' + file_pathname + '/' +file_name + '/' + file)
#                 if type(image)=='NoneType':
#                     continue
#                 image = affine_transform(image)
#                 cv2.imwrite(end_path + '/' + file, image)
#                 print(image.shape)



class Affine(object):
    def __init__(self, lane):
        self.lane = lane
    # ���б�Ե��⣬����ͼ��ռ�����Ҫ���ĵ�����
    def affine_transform(lane):
        lanecopy = lane.copy()
        lane1 = cv2.resize(lane, (lanecopy.shape[1], lanecopy.shape[1]))
        lane1 = cv2.GaussianBlur(lane1, (5, 5), 0)
        # ��˹ģ����Canny��Ե�����Ҫ��
        lane = cv2.Canny(lane1, 50, 200)
        # cv2.imshow("lane", lane)
        # cv2.waitKey(0)

        # lines = cv2.HoughLines(edges, 4, np.pi/180,200)
        # lines = np.reshape(lines,(lines.shape[0],-1))

        rho = 1  # ����ֱ���
        theta = np.pi / 180  # �Ƕȷֱ���
        threshold = 10  # ����ռ��ж��ٸ������ཻ��������ʽ����
        min_line_len = 10  # ���ٶ��ٸ����ص�Ź���һ��ֱ��
        max_line_gap = 50  # �߶�֮������������
        lines = cv2.HoughLinesP(lane, rho, theta, threshold, maxLineGap=max_line_gap)
        line_img = np.zeros_like(lane)

        point_x = []
        point_y = []
        point_distence = []
        if lines is None:
            return lanecopy
        for line in lines:
            for x1, y1, x2, y2 in line:
                point_x.append(x1)
                point_x.append(x2)
                point_y.append(y1)
                point_y.append(y2)
                point_distence.append(x1*x1+(y1+10)**2)
                point_distence.append(x2*x2+(y2+10)**2)
        #         cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)
        max_index = point_distence.index(max(point_distence))
        min_index = point_distence.index(min(point_distence))
        cv2.line(line_img, (point_x[max_index], point_y[max_index]), (point_x[min_index], point_y[min_index]), 255, 1)
        # cv2.imshow("line_img", line_img)
        # cv2.waitKey()

        x1y1 = [point_x[min_index], point_y[min_index]]
        x4y4 = [point_x[max_index], point_y[max_index]]

        for i in range(len(point_distence)):
            point_y[i] = lane.shape[0] - point_y[i]
            point_distence[i] = point_x[i]*point_x[i]+point_y[i]*point_y[i]
        max_index = point_distence.index(max(point_distence))
        min_index = point_distence.index(min(point_distence))
        cv2.line(line_img, (point_x[max_index], lane.shape[0] - point_y[max_index]), (point_x[min_index], lane.shape[0] - point_y[min_index]), 255, 1)
        # cv2.imshow("line_img", line_img)
        # cv2.waitKey()

        x3y3 = [point_x[min_index], lane.shape[0] - point_y[min_index]]
        x2y2 = [point_x[max_index], lane.shape[0] - point_y[max_index]]


        # ʹ�÷���ǰ��ĵ�
        pts1 = np.float32([x1y1, x2y2, x3y3, x4y4])
        pts2 = np.float32([[0, 0], [lane.shape[1], 0], [0, lane.shape[0] * 0.7], [lane.shape[1], lane.shape[0] * 0.7]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(lane1, M, (lane.shape[1], lane.shape[0]))
        endd = dst[0:int(lane.shape[0] * 0.65), :]
        # cv2.imshow("end", endd)
        # cv2.waitKey()


        return endd

#
# filepath = "D:/Pycharm_Project/Vehicle_license_plate_recognition/division_end"
# endpath = "D:/Pycharm_Project/Vehicle_license_plate_recognition/affine_trans_end"
# # creat_path(filepath, endpath)
# # write_to_path(filepath, endpath)
#
# # creat_path(filepath, endpath)
# read_path(filepath, endpath)
