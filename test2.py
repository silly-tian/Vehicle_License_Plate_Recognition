# coding=gbk

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ������ȡ


def cv_show(name, image):
    # ��ʾͼƬ
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def plt_show(image):
    # plt��ʾ��ɫͼƬ
    b, g, r = cv2.split(image)
    img = cv2.merge([r, g, b])
    img = cv2.imshow(image)
    plt.show()


def plt_show(image):
    # plt��ʾ�Ҷ�ͼƬ
    plt.imshow(image, cmap='gray')
    plt.show()



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

src = cv2.imread('C:\\Users\\15098\\Desktop\\123.jpg')
cv2.imshow("testimage", src)

src2 = src.copy()
src2 = color_position(src2)
# ��˹ģ��
Gauss = cv2.GaussianBlur(src2, (7, 7), 0)
# cv2.imshow("Gauss", Gauss)

# ת�Ҷ�ͼ
gray = cv2.cvtColor(Gauss, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gary", gray)

# sobel���ӱ�Ե���
Sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
absX = cv2.convertScaleAbs(Sobel_x)
image = absX.copy()
# cv2.imshow("sobel", image)

# # ����Ӧ��ֵ����(����ʾ������û���ⲿ�������)
# ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
# # cv2.imshow("binary", binary)

# �����㣬����ɫ������������
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
# print(kernelX)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=4)
# cv2.imshow("close", image)

# ȥ��С�İ׵�
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))

# ���ͣ���ʴ
image = cv2.dilate(image, kernelX)
image = cv2.erode(image, kernelX)
# ��ʴ������
image = cv2.erode(image, kernelY)
image = cv2.dilate(image, kernelY)
# cv2.imshow("clear_small_point", image)

# ��ֵ�˲�ȥ�����
# image = cv2.medianBlur(image, 15)
cv2.imshow("medianBlur", image)


# ת��Ϊ��ֵͼ��������findcontours
ret, binary = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)

# �������
# cv2.RETR_EXTERNAL��ʾֻ���������
# cv2.CHAIN_APPROX_SIMPLEѹ��ˮƽ���򣬴�ֱ���򣬶Խ��߷����Ԫ�أ� ֻ�����÷�����յ����꣬
# ����һ����������ֻ��Ҫ�ĸ���������������Ϣ
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# ��������
imagel = src.copy()
cv2.drawContours(imagel, contours, -1, (0, 0, 255), 3)
cv2.imshow("imagel", imagel)
#
#
cv2.waitKey(0)


# ɸѡ������λ�õ�����
for item in contours:
    # cv2.boundingRect��һ����С�ľ��Σ�������״������
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if (weight > (height*2)) and (weight < (height*5)):
        image = src[y:y+height, x:x+weight]
        cv2.imshow("end", image)
        cv2.waitKey(0)


