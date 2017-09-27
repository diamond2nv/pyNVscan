# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 17:50:00 2016

pyNVscan Advanced Tools by OpenCV for hot spot
GUI by OpenCV , for Hough Test

@author: listen
"""

import matplotlib
matplotlib.use('Agg')
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import sys


from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16) #默认宋体
#font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=16)  #微软雅黑


png_io = BytesIO()
PZT_LIMIT_X = 100.0
PZT_LIMIT_Y = 100.0

def nothing(x):
    pass


def read_CSV_Head(csv_head):
    x0 = csv_head[0]
    y0 = csv_head[1]
    x1 = csv_head[2]
    y1 = csv_head[3]
    step_move = csv_head[4]    
    return x0, y0, x1, y1, step_move
    

#由区域扫描结果Data.csv，创建head
def creat_CSV_Head_File(my_matrix):
    x0 = np.min(my_matrix[:,0])
    x1 = np.max(my_matrix[:,0])
    y0 = np.min(my_matrix[:,1])
    y1 = np.max(my_matrix[:,1])
    step_move = my_matrix[1,0] - my_matrix[0,0]
    dx = x1 - x0
    dy = y1 - y0
    lx = np.size(my_matrix[:,0])
    ldata = np.size(my_matrix[:,2])
    if abs(lx-(dx/step_move+1.0)*(dy/step_move+1.0)) >= 0.01:
        print "Error: creat large_scan_head.csv"
        sys.exit(255)
    elif x0<0 or y0<0 or x1>PZT_LIMIT_X or y1>PZT_LIMIT_Y: 
        print "Error: PZT XY LIMITs"
        sys.exit(255)
    elif x0>=x1 or y0>=y1:
        print "Error: creat Large Scan"
        sys.exit(255)
    head = [x0, y0, x1, y1, step_move, lx, ldata]
    
    return head

        
        
#最大公约数
def gcd(a, b):
    if a < b:
        a, b = b, a

    while b != 0:
        temp = a % b
        a = b
        b = temp

    return a
    
    
#由csv绘制纯图像,返回比例尺值
def csv_to_PNG(my_matrix, csv_head, mode, contour_i, png_file_path):
#   print csv_head
    x0,y0,x1,y1,step_move = read_CSV_Head(csv_head)
    # Becouse of numpy arange()
    x2 = x1 + step_move * 0.5
    y2 = y1 + step_move * 0.5
    size_xy = int(csv_head[5])
    progress_len = int(csv_head[6])
    wave_data = my_matrix[:,2]

    w_data = np.zeros(size_xy) # ready for one-photon_count
    #progress_len
    w_data[0:progress_len] = wave_data[0:progress_len] # Warning: NOT progress_len-1
    X = np.arange(x0, x2, step_move)
    Y = np.arange(y0, y2, step_move)
    len_X = np.size(X)
    len_Y = np.size(Y)

    Z0 = w_data.reshape(-1,len_X)
    Z0[1::2,:] = Z0[1::2,::-1]

    extent = np.array([x0, x1, y0, y1]) + np.array([-step_move,step_move,-step_move,step_move])*0.5
    #自适应去除matplotlib figure tight_layout PAD,放大到6*100像素的宽
    fd = gcd(len_X,len_Y)
    fx = len_X / fd
    fy = len_Y / fd
    if fx>(fy*0.5) and fx<=3:
        fd = 6/fx
        fx = 6
        fy = fd*fy
        
    plt.figure(figsize=(fx,fy))
    axes = plt.subplot(111)
#   axes.imshow(Z0, extent=extent, origin="lower",interpolation='nearest')
    #   imshow, contour mode, 
    #0    +        -
    #1    +        +
    #2    -        +
    if mode < 2 :
        axes.imshow(Z0, extent=extent, cmap='gray',origin="lower")
    if mode > 0 :
        axes.contour(X, Y, Z0, contour_i, cmap='cool')
    #for OpenCV ,NO AXEX,NO PAD
    axes.set_xticks([])
    axes.set_yticks([])
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.spines['bottom'].set_color('none')
    axes.spines['left'].set_color('none')
    plt.tight_layout(pad=0)
    plt.savefig(png_file_path) 
    
    png_ruler = (x1 - x0 + step_move) / (fx * 100.0)    # 保存图像的比例尺：1个像素代表的微米    
    return png_ruler
    

#图片像素坐标[左上角(0,0)],变换到PZT坐标(微米)[左下角(0,0)] ，(dx,dy)偏移比例 
def pngXY_to_pztXY(csv_head, gray, png_x, png_y):
    x0,y0,x1,y1,step_move = read_CSV_Head(csv_head)
    
    Lx,Ly = gray.shape
    
#   pzt_x = png_x / Lx * (x1 - x0 + step_move) + x0 - step_move*0.5
#   pzt_y = (Ly - png_y) / Ly * (y1 - y0 + step_move) + y0 - step_move*0.5
    
    pzt_x = png_x / (Lx - 1.0) * (x1 - x0 + step_move) + x0 - step_move*0.5
    pzt_y = (Ly - png_y) / (Ly - 1.0) * (y1 - y0 + step_move) + y0 - step_move*0.5
    
    if pzt_x<0:
        pzt_x = 0;
    elif pzt_x>PZT_LIMIT_X:
        pzt_x =PZT_LIMIT_X
    elif pzt_y<0:
        pzt_y = 0;
    elif pzt_y>PZT_LIMIT_Y:
        pzt_y = PZT_LIMIT_Y
        
    return pzt_x, pzt_y
    
    
#PZT坐标(微米)变换到large_scan.png图片像素坐标
def pztXY_to_pngXY(csv_head, gray, pzt_x, pzt_y):
    x0,y0,x1,y1,step_move = read_CSV_Head(csv_head)
    
    Lx,Ly = gray.shape
    
#   png_x = (pzt_x - x0 + step_move*0.5) / (x1 - x0 + step_move) * Lx
#   png_y = Ly - (pzt_y - y0 + step_move*0.5) / (y1 - y0 + step_move) * Ly
    
    png_x = (pzt_x - x0 + step_move*0.5) / (x1 - x0 + step_move) * (Lx - 1.0)
    png_y = Ly - (pzt_y - y0 + step_move*0.5) / (y1 - y0 + step_move) * (Ly - 1.0)
    
#   png_x = np.uint16(round(png_x))    
#   png_y = np.uint16(round(png_y))
    return png_x, png_y


#筛选: 识别圆心坐标(浮点小数像素坐标系)，返回最靠近坐标的浮点像素坐标值
def select_Hot_Spot(circles,png_x,png_y):
    num_circles = np.size(circles[:,0])
    r_d = np.zeros(num_circles)
    r_d = r_d.reshape(-1,1)
    r_d[:,0] = (circles[:,0] - png_x)**2 + (circles[:,1] - png_y)**2
    
    amin_r = np.argmin(r_d[:]) 
    
    png_x = circles[amin_r,0]
    png_y = circles[amin_r,1] 
    png_r = circles[amin_r,2]          
    return png_x, png_y, png_r
    
    
#png高斯模糊后，霍夫圆变换，centers的XY像素坐标列表,返回最靠近坐标的浮点像素坐标值
def mpl_Hough(csv_head, img, gray, png_x, png_y, hot_r, png_ruler, lowThreshold, higThreshold): 
    
    min_Dist, sml_Radius, big_Radius = hot_R_Hough(hot_r, png_ruler)
    
    circles1 = cv2.HoughCircles(gray,cv.CV_HOUGH_GRADIENT ,1,
                                minDist=min_Dist,param1=lowThreshold,param2=higThreshold,
                                minRadius=sml_Radius,maxRadius=big_Radius)
    img0 = np.array(img)
    w, h = gray.shape
    line_x = np.uint16(round(png_x))
    line_y = np.uint16(round(png_y))
    cv2.line(img0,(line_x,0),(line_x,h),(255,0,0),2)
    cv2.line(img0,(0,line_y),(w,line_y),(255,0,0),2)
                            
    if circles1 is None:
        cv2.imshow('mpl_Hough',img0)
        return 0,0,0,0
    else:
        circles = circles1[0,:,:]#三维提取为二维,非常易出Bug，返回值为NoneType或数组
        png_x, png_y, png_r = select_Hot_Spot(circles,png_x,png_y)
        
        circles = np.uint16(np.around(circles))#四舍五入，取整
        for i in circles[:]: 
            cv2.circle(img0,(i[0],i[1]),i[2],(0,0,255),2)#画圆
            cv2.circle(img0,(i[0],i[1]),2,(0,0,255),2)#画圆心  
        box_x0, box_y0, box_x1, box_y1 = lock_Box_Draw(png_x, png_y, png_r)
        cv2.rectangle(img0,(box_x0,box_y0),(box_x1,box_y1),(0,255,0),2) 
        
        pzt_x, pzt_y = pngXY_to_pztXY(csv_head, gray, png_x, png_y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img0,"({:.2f}, {:.2f})".format(pzt_x, pzt_y),(10,30), font, 0.8,(255,255,255),2)
        cv2.imshow('mpl_Hough',img0) 
        return 1, png_x, png_y, png_r     #浮点像素坐标值
    
 
#由亮点的PZT坐标系半径值，和比例尺，求像素坐标系的霍夫圆变换部分参数
def hot_R_Hough(hot_r, png_ruler):
    hot_r_png = hot_r / png_ruler * 1.0
    min_Dist = np.uint16(round(hot_r_png * 2.5))
    sml_Radius = np.uint16(round(hot_r_png * 0.3))
    big_Radius = np.uint16(round(hot_r_png * 2.0))
    return min_Dist, sml_Radius, big_Radius   


def lock_Box_Draw(png_x, png_y, png_r):
    box_x0 = np.uint16(round(png_x - png_r - 1.0))
    box_x1 = np.uint16(round(png_x + png_r + 1.0))
    box_y0 = np.uint16(round(png_y - png_r - 1.0))
    box_y1 = np.uint16(round(png_y + png_r + 1.0))
    return box_x0, box_y0, box_x1, box_y1
    
    
def gui_CV_Number(hot_x_100, hot_y_100, hot_r_100):
    return hot_x_100/100.0, hot_y_100/100.0, hot_r_100/100.0


##########################################
########    main()    ####################
##########################################
def main():
    lowThreshold = 100
    higThreshold = 12  
    max_lowThreshold = 300 
    max_higThreshold = 200
    mode = 1
    contour_i = 10
    hot_x_100 = 5000
    hot_y_100 = 5000
    hot_r_100 = 20
    csv_filepath_large_data = 'pyNVscan_AT_CV_V3.1.csv'
    
    cv2.namedWindow('mpl_Hough',cv2.WINDOW_NORMAL) 
    cv2.namedWindow('param',cv2.WINDOW_NORMAL) 
  
    cv2.createTrackbar('Low threshold','param',100, max_lowThreshold,nothing)  
    cv2.createTrackbar('Hig threshold','param',12, max_higThreshold,nothing) 
#   cv2.createTrackbar('Matplotlib Mode','param',0, 2, nothing)
#   cv2.createTrackbar('Contour_i','param',10, 100, nothing)
    cv2.createTrackbar('Hot_X_100','param',5000, 10000, nothing)
    cv2.createTrackbar('Hot_Y_100','param',5000, 10000, nothing)
    cv2.createTrackbar('Hot_R_100','param',40, 10000, nothing)
    
    with open(csv_filepath_large_data,"rb") as f: 
        my_matrix = np.loadtxt(f, delimiter=",", skiprows=0)
        
    csv_head_large = creat_CSV_Head_File(my_matrix)

    png_ruler_large = csv_to_PNG(my_matrix, csv_head_large, mode, contour_i, png_io)
    
    img = cv2.imdecode(np.fromstring(png_io.getvalue(), dtype=np.uint8), 1)    #读内存中的二进制图像数据
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussian_gray = cv2.GaussianBlur(gray,(3,3),0)
    
    pzt_x, pzt_y, pzt_r = gui_CV_Number(hot_x_100, hot_y_100, hot_r_100)
    
    png_x, png_y = pztXY_to_pngXY(csv_head_large, gray, pzt_x, pzt_y)

    flag_hough, png_x1, png_y1, png_r1 = mpl_Hough(csv_head_large, img, gaussian_gray, png_x, png_y, pzt_r, png_ruler_large, lowThreshold, higThreshold)  # initialization  
    
    while(1):
        k=cv2.waitKey(20)&0xFF
        if k==27:
            break

        lowThreshold = cv2.getTrackbarPos('Low threshold','param')
        higThreshold = cv2.getTrackbarPos('Hig threshold','param')
#       mode = cv2.getTrackbarPos('Matplotlib Mode','param')
#       contour_i = cv2.getTrackbarPos('Contour_i','param')
        hot_x_100 = cv2.getTrackbarPos('Hot_X_100','param')
        hot_y_100 = cv2.getTrackbarPos('Hot_Y_100','param')
        hot_r_100 = cv2.getTrackbarPos('Hot_R_100','param')

        pzt_x, pzt_y, pzt_r = gui_CV_Number(hot_x_100, hot_y_100, hot_r_100)
    
        png_x, png_y = pztXY_to_pngXY(csv_head_large, gray, pzt_x, pzt_y)

        flag_hough, png_x1, png_y1, png_r1 = mpl_Hough(csv_head_large,img, gaussian_gray, png_x, png_y, pzt_r, png_ruler_large, lowThreshold, higThreshold)
        
    cv2.destroyAllWindows()  
    
main()