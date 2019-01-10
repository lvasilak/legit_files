### Calibrate
### Created by: Lindsay Vasilak
### Last Modified: 05-Jan-2019
##Main
import numpy as np
import os
import cv2
import sys
import imutils
import pickle
import argparse
from pyimagesearch.shapedetector import ShapeDetector
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils

###########Define functions for rectangle points ###############
def order_points_old(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect
    
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point	
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")
    

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
## erode and dilate the threshed image to remove noise
thresh=cv2.erode(thresh, None, iterations=2)
cv2.imshow("preview",thresh)
cv2.waitKey(0)
thresh=cv2.dilate(thresh, None, iterations=2)
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
##im,contours,hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

## sort the contours from left-to-right and initiaize the bounding bax
## point colors
cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

## draw all the contours
##contour_img=img_rgb.copy()
##cv2.drawContours(contour_img,contours,-1,(255,0,0),2)
##cv2.imshow("preview",contour_img)
##cv2.waitKey(0) 
    
## find the contour with the biggest area
c=max(cnts,key=cv2.contourArea)
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
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.array(box, dtype="int")

# compute the rotated bounding box of the contour, then
# draw the contours
order_img=img_rgb.copy()
cv2.drawContours(order_img, [box], -1, (0, 255, 0), 2)

# order the points in the contour such that they appear
# in top-left, top-right, bottom-right, and bottom-left
# order, then draw the outline of the rotated bounding
# box
rect = order_points_old(box)

# check to see if the new method should be used for
# ordering the coordinates
##if args["new"] > 0:
##	rect = perspective.order_points(box)

# loop over the original points and draw them 
for ((x, y), color) in zip(rect, colors):
        cv2.circle(order_img, (int(x), int(y)), 5, color, -1)

topLeft=rect[0]
topRight=rect[1]
botRight=rect[2]
botLeft=rect[3]
print box

# show the image
cv2.imshow("preview", order_img)
cv2.waitKey(0)

## determine the most extreme points along the contour
##extLeft = tuple(c[c[:, :, 0].argmin()][0])
##extRight = tuple(c[c[:, :, 0].argmax()][0])
##extTop = tuple(c[c[:, :, 1].argmin()][0])
##extBot = tuple(c[c[:, :, 1].argmax()][0])
##ext_img=img_rgb.copy()
##cv2.drawContours(ext_img,[c],-1,(0,255,255),2)
##cv2.circle(ext_img,extLeft,8, (0, 0, 255), -1)
##cv2.circle(ext_img,extRight,8, (0, 255,0), -1)
##cv2.circle(ext_img,extTop,8, (255, 0, 0), -1)
##cv2.circle(ext_img,extBot,8, (255, 255,0 ), -1)
##cv2.imshow("preview",ext_img)
##cv2.waitKey(0)

##  save rectangle points
filehandler=open("topLeft.obj","wb")
pickle.dump(topLeft,filehandler)
filehandler.close()
filehandler=open("topRight.obj","wb")
pickle.dump(topRight,filehandler)
filehandler.close()
filehandler=open("botLeft.obj","wb")
pickle.dump(botLeft,filehandler)
filehandler.close()
filehandler=open("botRight.obj","wb")
pickle.dump(botRight,filehandler)
filehandler.close()



### Get the regions of the breadboard
### region 1
h = dist.cdist([topLeft],[botLeft], "euclidean")
r1_w=0.16
r2_w=0.31
r3_w=r2_w
r4_w=r1_w
vtl=topLeft-botLeft
vtl_norm=np.linalg.norm(vtl)
utl=np.divide(vtl,vtl_norm)
vtr=topRight-botRight
vtr_norm=np.linalg.norm(vtr)
utr=np.divide(vtr,vtr_norm)
r1_br=topRight+r1_w*h*utr
r1_bl=topLeft + r1_w*h*utl

print r1_br
print np.array(r1_bl).flatten()
#testing=np.array(topLeft).flatten()
print np.array(topLeft).flatten()
print topRight
region1=np.array([np.array(topLeft).flatten(),np.array(topRight).flatten(),np.array(r1_bl).flatten(),np.array(r1_br).flatten()],dtype="int32")

#bar_w=int((box[3,1]-box[2,1])*0.04)
#region1=np.array([[box[1,0],box[1,1]],[box[2,0],box[2,1]],[box[2,0],box[2,1]+r1_w],[box[1,0],box[1,1]+r1_w]])

print region1
img_thing=img_rgb.copy()
cv2.drawContours(img_thing,region1,-1,(0,255,0),3)
cv2.imshow("preview",img_thing)
cv2.waitKey(0)


cv2.destroyAllWindows()
