### Calibrate
### Created by: Lindsay Vasilak
### Last Modified: 04-Jan-2019
##Main
import numpy as np
import os
import cv2
import sys
import imutils
import pickle
import argparse
from pyimagesearch.shapedetector import ShapeDetector


############ TAKING THE SNAPSHOT OF BLANK BREADBOARD ###########
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_AUTOFOCUS,1)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()      
else:
    rval = False
while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            cv2.destroyAllWindows()
            vc.release()
            break
        if key == 99:
            rval, frame = vc.read()
            break
cv2.destroyAllWindows()
vc.release()
        

################ AUTO CALIBRATE #####################
img_rgb=frame
img = img_rgb.copy()

## make bgr image gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel_size = 5
## blur the gray image
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
cv2.imshow("preview",blur_gray)
cv2.waitKey(0)
## create binary image
ret, thresh = cv2.threshold(blur_gray, 80,255, cv2.THRESH_BINARY)
cv2.imshow("preview",thresh)
cv2.waitKey(0)
## create canny edge image
low_threshold = 90  
high_threshold =250 
edges = cv2.Canny(thresh, low_threshold, high_threshold,None,3)
cv2.imshow("preview",edges)
edgesCopy=edges.copy()
cv2.waitKey(0)

##h,w=img_rgb.shape[:2]
##mask=np.zeros((h+2,w+2),np.uint8)

## morph closing of the edges
kernel=np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(edges,cv2.MORPH_CLOSE, kernel)
cv2.imshow("preview",closing)
cv2.waitKey(0)

## get contours of morph closed edges
h,w=edges.shape[:2]
mask=np.zeros((h+2,w+2),np.uint8)
im,contours,hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

## draw all the contours
contour_img=img_rgb.copy()
cv2.drawContours(contour_img,contours,-1,(255,0,0),2)
cv2.imshow("preview",contour_img)
cv2.waitKey(0) 
    
## find the contour with the biggest area
c=max(contours,key=cv2.contourArea)
print cv2.contourArea(c)
epsilon=0.1*cv2.arcLength(c,True)
## approx the contour
approx_img=img_rgb.copy()
approx=cv2.approxPolyDP(c,epsilon,True)
cv2.drawContours(approx_img,[approx],-1,(255,255,0),3)
cv2.imshow("preview",approx_img)
cv2.waitKey(0)

## convex hull the contour
hull = cv2.convexHull(c)
hull_img=img_rgb.copy()
cv2.drawContours(hull_img,[hull],-1,(255,255,0),3)
cv2.imshow("preview",hull_img)
cv2.waitKey(0)
## get rotated rectangle version of the hull contour                             
rect=cv2.minAreaRect(hull)
box=cv2.boxPoints(rect)
box=np.int0(box)
print box
rectCont=img_rgb.copy()
cv2.drawContours(rectCont,[box],-1,(255,255,0),3)
cv2.imshow("preview",rectCont)
cv2.waitKey(0)
## close windows and save rectangle points
cv2.destroyAllWindows()
filehandler=open("box.obj","wb")
pickle.dump(box,filehandler)
filehandler.close()

