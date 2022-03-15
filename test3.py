# coding=gbk

import cv2
import numpy as np


# src = cv2.imread('C:\\Users\\15098\\Desktop\\t9.jpg')
src = cv2.imread('C:\\Users\\15098\\Desktop\\cardsneed\\20180709\\15311073037.jpg')
cv2.imshow("testimage", src)


def color_position(img):
    colors = [
        # ([26, 43, 46], [34, 255, 255]),  # ��ɫ
        ([100, 43, 46], [124, 255, 255]),  # ��ɫ
        # ([35, 43, 46], [77, 255, 255])   # ��ɫ
    ]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for (lower, upper) in colors:
        lower = np.array(lower, dtype="uint8")  # ��ɫ����
        upper = np.array(upper, dtype="uint8")  # ��ɫ����

        # ������ֵ�ҵ���Ӧ����ɫ
        mask = cv2.inRange(hsv, lowerb=lower, upperb=upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow("image", img)
        # cv2.imshow("image_color", output)
        # cv2.waitKey(0)
    return output


mark_img = color_position(src)
# cv2.imshow("mark_img", mark_img)


# ��ԭͼ����ǣ���ͼ��
gray_mark = cv2.cvtColor(mark_img, cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


# ͼ���ֵ��
ret, binary = cv2.threshold(gray_mark, 0, 255, cv2.THRESH_BINARY)
# cv2.imshow("binary", binary)


# �����㣬����ɫ������������
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
# print(kernelX)
image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernelX, iterations=4)
# cv2.imshow("close", image)


# �������
contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
drawing = src.copy()
cv2.drawContours(drawing, contours, -1, (0, 0, 255), 3)  # ���������ɫ
cv2.imshow('drawing', drawing)


temp_contours = []  # �洢���������
car_plates = []
if len(contours) > 0:
    for contour in contours:
        if cv2.contourArea(contour) > 1:
            temp_contours.append(contour)
        car_plates = []
        for temp_contour in temp_contours:
            rect_tupple = cv2.minAreaRect(temp_contour)
            rect_width, rect_height = rect_tupple[1]
            if rect_width < rect_height:
                rect_width, rect_height = rect_height, rect_width
            aspect_ratio = rect_width / rect_height
            # ������������ֵ����߱ȣ���2-5.5֮��
            if aspect_ratio > 2 and aspect_ratio < 5.5:
                car_plates.append(temp_contour)
                rect_vertices = cv2.boxPoints(rect_tupple)
                rect_vertices = np.int0(rect_vertices)
        if len(car_plates)==1:
            oldimg = cv2.drawContours(src, [rect_vertices], -1, (0, 0, 255), 2)
            # cv2.imshow("dingwei", oldimg)
            break

if len(car_plates)==1:
    for car_plate in car_plates:
        row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
        row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
        cv2.rectangle(src, (row_min, col_min), (row_max, col_max), (0, 255, 0), 2)
        card_img = src[col_min:col_max, row_min:row_max, :]
        cv2.imshow("src", src)
    cv2.imshow("card_img.", card_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.waitKey(0)









