# coding=gbk

import cv2
import numpy as np


def carimg_make(img):
    # Ԥ����ͼ��
    rect, afterimg = preprocessing(img)  # ��ʵ�����˶Գ��ƶ�λ
    print("rect:", rect)

    # �������
    cv2.rectangle(afterimg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv2.imshow('afterimg1', afterimg)

    # �ָ���뱳��
    cutimg = cut_license(afterimg, rect)
    cv2.imshow('cutimg', cutimg)

    # ��ֵ�����ɺڰ�ͼ
    thresh = lice_binarization(cutimg)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

    # �ָ��ַ�
    '''
    �жϵ�ɫ����ɫ
    '''
    # ��¼�ڰ������ܺ�
    white = []  # ��¼ÿһ�еİ�ɫ�����ܺ�
    black = []  # ��¼ÿһ�еĺ�ɫ�����ܺ�
    height = thresh.shape[0]  # 263
    width = thresh.shape[1]  # 400
    white_max = 0  # ������ÿ�У�ȡ���а�ɫ������������
    black_max = 0  # ������ÿ�У�ȡ���к�ɫ������������
    # ����ÿһ�еĺڰ������ܺ�
    for i in range(width):
        line_white = 0  # ��һ�а�ɫ����
        line_black = 0  # ��һ�к�ɫ����
        for j in range(height):
            if thresh[j][i] == 255:
                line_white += 1
            if thresh[j][i] == 0:
                line_black += 1
        white_max = max(white_max, line_white)
        black_max = max(black_max, line_black)
        white.append(line_white)
        black.append(line_black)
        print('white_max', white_max)
        print('black_max', black_max)
    # argΪtrue��ʾ�ڵװ��֣�FalseΪ�׵׺���
    arg = True
    if black_max < white_max:
        arg = False
    # �ָ�Ƶ�����
    n = 1
    start = 1
    end = 2
    s_width = 28
    s_height = 28
    temp = 1
    while n < width - 2:
        n += 1
        # �ж��ǰ׵׺��ֻ��Ǻڵװ���  0.05������Ӧ�����0.95 ��������
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):  #  ���û�����͸��
            start = n
            end = find_end(start, arg, black, white, width, black_max, white_max)
            n = end
            print("start" + str(start))
            print("end" + str(end))
            # ˼·���Ǵ���ʼ���ƥ���ַ�������ȣ�end - start��С��20����Ϊ�������� pass��  ��������ʶ�𣬷���˵����
            # ʡ�ݼ�ƣ����У�ѹ�� ���棬����һ��������λ������ 1 ʱ�����Ŀ��Ҳ�Ǻ�խ�ģ����Ծ�ֱ����Ϊ������ 1 ����Ҫ��
            # ��Ԥ���ˣ���Ȼ��խ�� 1 ����  ѹ�������Ǳ�����ģ���shutil.copy()�����ǵ����
            # �������ν�� 1 ʱ�����������п���һ�� 1 ��ͼƬ����ǰtemp�±��µ��ַ�
            # if end - start > 5:
            #     print("end - start" + str(end - start))
            if end - start > 5:
                cj = thresh[1:height, start:end]
                print("result/%s.jpg" % (n))
                cv2.imwrite('img/{0}.bmp'.format(n), cj)
                #�Էָ�������֡���ĸ���вü�
                b_img = cv2.resize(cj, None, fx=5, fy=3)
                contours, hierarchy = cv2.findContours(b_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                block = []
                for c in contours:
                    # �ҳ����������ϵ�����µ㣬�ɴ˼�����������ͳ��ȱ�
                    r = find_rectangle(c)  # ���������������ϵ�����µ�
                    a = (r[2] - r[0]) * (r[3] - r[1])  # ���
                    s = (r[2] - r[0]) / (r[3] - r[1])  # ���ȱ�
                    block.append([c, r, a, s])
                block1 = sorted(block, key=lambda block: block[2])[-1:]
                # rect = cv2.minAreaRect(block2)
                # box1 = np.int0(cv2.boxPoints(rect))
                box = block1[0][1]
                y_mia = box[0]  # y_mia
                x_min = box[1]  # x_min
                y_max = box[2]  # y_max
                x_max = box[3]  # x_max
                cropImg = b_img[x_min:x_max, y_mia:y_max]  # crop the image
                cv2.imwrite('img_test/{0}.bmp'.format(n), cropImg)
                cv2.imshow('cutlicense', cj)
                cv2.imshow("charecter",cropImg)
                cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocessing(img):
    '''
    Ԥ������
    '''
    m = 400 * img.shape[0] / img.shape[1]

    #ѹ��ͼ��
    img=cv2.resize(img,(400,int(m)),interpolation=cv2.INTER_CUBIC)

    #BGRת��Ϊ�Ҷ�ͼ��
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print('gray_img.shape',gray_img.shape)

    #�Ҷ�����
    #���һ��ͼ��ĻҶȼ����ڽϰ������������ͼ��ƫ���������ûҶ����칦��������(б��>1)����Ҷ������Ը���ͼ��
    # ͬ�����ͼ��Ҷȼ����ڽ��������������ͼ��ƫ����Ҳ�����ûҶ����칦����ѹ��(б��<1)����Ҷ������Ը���ͼ������
    stretchedimg=stretching(gray_img)#���лҶ����죬����Ϊ���Ը���ͼ�������
    print('stretchedimg.shape',stretchedimg.shape)

    '''���п����㣬����ȥ������'''
    r=15
    h=w=r*2+1
    kernel=np.zeros((h,w),np.uint8)
    cv2.circle(kernel,(r,r),r,1,-1)
    #������
    openingimg=cv2.morphologyEx(stretchedimg,cv2.MORPH_OPEN,kernel)
    #��ȡ���ͼ������ͼ������  cv2.absdiff('ͼ��1','ͼ��2')
    strtimg=cv2.absdiff(stretchedimg,openingimg)
    cv2.imshow("stretchedimg",stretchedimg)
    cv2.imshow("openingimg1",openingimg)
    cv2.imshow("strtimg",strtimg)
    cv2.waitKey(0)

    #ͼ���ֵ��
    binaryimg=allbinaryzation(strtimg)
    cv2.imshow("binaryimg",binaryimg)
    cv2.waitKey(0)

    #canny��Ե���
    canny=cv2.Canny(binaryimg,binaryimg.shape[0],binaryimg.shape[1])
    cv2.imshow("canny",canny)
    cv2.waitKey(0)

    '''�����������������������򣬴Ӷ���λ����'''
    #���б�����
    kernel=np.ones((5,23),np.uint8)
    closingimg=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("closingimg",closingimg)

    #���п�����
    openingimg=cv2.morphologyEx(closingimg,cv2.MORPH_OPEN,kernel)
    cv2.imshow("openingimg2",openingimg)

    #�ٴν��п�����
    kernel=np.ones((11,6),np.uint8)
    openingimg=cv2.morphologyEx(openingimg,cv2.MORPH_OPEN,kernel)
    cv2.imshow("openingimg3",openingimg)
    cv2.waitKey(0)

    #  ����С���򣬶�λ����λ��
    rect = locate_license(openingimg, img)   #  rect�������������ϵ�����µ㣬������Լ����

    return rect, img


def locate_license(img,afterimg):
    '''
    ��λ���ƺ�
    '''
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_copy = afterimg.copy()
    img_cont = cv2.drawContours(img_copy,contours,-1,(255,0,0),6)
    cv2.imshow("img_cont",img_cont)
    cv2.waitKey(0)
    #�ҳ�������������
    block=[]
    for c in contours:
        #  �ҳ����������ϵ�����µ㣬�ɴ˼�����������ͳ��ȱ�
        r=find_rectangle(c)         #   ���������������ϵ�����µ�
        a=(r[2]-r[0])*(r[3]-r[1])   #   ���
        s=(r[2]-r[0])/(r[3]-r[1])   #   ���ȱ�

        block.append([r, a, s])
    #ѡ���������3������
    block=sorted(block,key=lambda b: b[1])[-3:]

    #ʹ����ɫʶ���ж��ҳ������Ƶ�����
    maxweight,maxindex=0,-1
    for i in range(len(block)):#len(block)=3
        b=afterimg[block[i][0][1]:block[i][0][3],block[i][0][0]:block[i][0][2]]
        #BGRתHSV
        hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
        lower=np.array([100,50,50])
        upper=np.array([140,255,255])
        #������ֵ������Ĥ
        mask=cv2.inRange(hsv,lower,upper)
        #ͳ��Ȩֵ
        w1=0
        for m in mask:
            w1+=m/255

        w2=0
        for n in w1:
            w2+=n

        #ѡ�����Ȩֵ������
        if w2>maxweight:
            maxindex=i
            maxweight=w2

    return block[maxindex][0]


def stretching(img):
    '''
    ͼ�����캯��
    '''
    maxi=float(img.max())
    mini=float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]=(255/(maxi-mini)*img[i,j]-(255*mini)/(maxi-mini))

    return img

def allbinaryzation(img):
    '''
    ��ֵ��������
    '''
    maxi=float(img.max())
    mini=float(img.min())

    x=maxi-((maxi-mini)/2)
    #��ֵ��,������ֵret  ��  ��ֵ���������ͼ��thresh
    ret,thresh=cv2.threshold(img,x,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    #���ض�ֵ����ĺڰ�ͼ��
    return thresh

def cut_license(afterimg,rect):
    '''
    ͼ��ָ��
    '''
    #ת��Ϊ��Ⱥ͸߶�
    rect[2]=rect[2]-rect[0]
    rect[3]=rect[3]-rect[1]
    rect_copy=tuple(rect.copy())#tuple��һ��Ԫ��
    print("rect_copy",rect_copy)
    rect=[0,0,0,0]
    #������Ĥ
    mask=np.zeros(afterimg.shape[:2],np.uint8)
    #��������ģ��  ��Сֻ��Ϊ13*5������ֻ��Ϊ1����ͨ��������
    bgdModel=np.zeros((1,65),np.float64)
    #����ǰ��ģ��
    fgdModel=np.zeros((1,65),np.float64)
    #�ָ�ͼ��
    cv2.grabCut(afterimg,mask,rect_copy,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_show=afterimg*mask2[:,:,np.newaxis]
    return img_show


def lice_binarization(licenseimg):
    '''
    ����ͼƬ��ֵ��
    '''
    #���Ʊ�Ϊ�Ҷ�ͼ��
    gray_img=cv2.cvtColor(licenseimg,cv2.COLOR_BGR2GRAY)

    #��ֵ�˲�  ȥ������
    kernel=np.ones((3,3),np.float32)/9
    gray_img=cv2.filter2D(gray_img,-1,kernel)

    #��ֵ������
    ret,thresh=cv2.threshold(gray_img,120,255,cv2.THRESH_BINARY)

    return thresh

def find_end(start,arg,black,white,width,black_max,white_max):
    end=start+1
    for m in range(start+1,width-1):
        if (black[m] if arg else white[m])>(0.95*black_max if arg else 0.95*white_max):
            end=m
            break
    return end

def find_rectangle(contour):
    '''
    Ѱ�Ҿ�������
    '''
    y,x=[],[]

    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]





image = cv2.imread("cardsneed/20180710/15311525352.jpg")
carimg_make(image)

