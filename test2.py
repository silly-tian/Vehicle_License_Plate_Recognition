# coding=gbk

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 车牌提取


def cv_show(name, image):
    # 显示图片
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def plt_show(image):
    # plt显示彩色图片
    b, g, r = cv2.split(image)
    img = cv2.merge([r, g, b])
    img = cv2.imshow(image)
    plt.show()


def plt_show(image):
    # plt显示灰度图片
    plt.imshow(image, cmap='gray')
    plt.show()



def color_position(img):
    colors = [
        # ([26, 43, 46], [34, 255, 255]),  # 黄色
        ([100, 43, 46], [124, 255, 255]),  # 蓝色
        # ([35, 43, 46], [77, 255, 255])   # 绿色
    ]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for (lower, upper) in colors:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到相应的颜色
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
# 高斯模糊
Gauss = cv2.GaussianBlur(src2, (7, 7), 0)
# cv2.imshow("Gauss", Gauss)

# 转灰度图
gray = cv2.cvtColor(Gauss, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gary", gray)

# sobel算子边缘检测
Sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
absX = cv2.convertScaleAbs(Sobel_x)
image = absX.copy()
# cv2.imshow("sobel", image)

# # 自适应阈值处理(对于示例可能没有这部处理更好)
# ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
# # cv2.imshow("binary", binary)

# 闭运算，讲白色部分连成整体
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
# print(kernelX)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=4)
# cv2.imshow("close", image)

# 去除小的白点
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))

# 膨胀，腐蚀
image = cv2.dilate(image, kernelX)
image = cv2.erode(image, kernelX)
# 腐蚀，膨胀
image = cv2.erode(image, kernelY)
image = cv2.dilate(image, kernelY)
# cv2.imshow("clear_small_point", image)

# 中值滤波去除噪点
# image = cv2.medianBlur(image, 15)
cv2.imshow("medianBlur", image)


# 转化为二值图象，以用于findcontours
ret, binary = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)

# 轮廓检测
# cv2.RETR_EXTERNAL表示只检测外轮廓
# cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素， 只保留该方向的终点坐标，
# 例如一个矩形轮廓只需要四个点来保存轮廓信息
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
imagel = src.copy()
cv2.drawContours(imagel, contours, -1, (0, 0, 255), 3)
cv2.imshow("imagel", imagel)
#
#
cv2.waitKey(0)


# 筛选出车牌位置的轮廓
for item in contours:
    # cv2.boundingRect用一个最小的矩形，把新形状包起来
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if (weight > (height*2)) and (weight < (height*5)):
        image = src[y:y+height, x:x+weight]
        cv2.imshow("end", image)
        cv2.waitKey(0)


