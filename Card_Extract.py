# coding=gbk

import cv2
import numpy as np


src = cv2.imread("C:\\Users\\15098\\Desktop\\123.jpg")
cv2.imshow("testimage", src)


# ʹ����ɫͨ���޶�
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


color_position(src)
cv2.imshow("end",color_position(src))
cv2.waitKey()



