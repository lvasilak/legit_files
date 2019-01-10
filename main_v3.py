### User places one component, then the component is detected,
### then finds location
### Created 2018-11-18
### Created by: Lindsay Vasilak

import numpy as np
import os
import cv2
import sys
import imutils
import pickle
import datetime
from snapshot import snapshot
import math
import time
import argparse
from pyimagesearch.shapedetector import ShapeDetector
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours


############## Defining Functions ##############################

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


def calibrate(frame):
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







def componentDetect(frame, resTemp, capTemp,icTemp,ledTemp):
        ### Detect Components in frame
                cv2.destroyAllWindows()
                resMat=[]
                capMat=[]
                ledMat=[]
                icMat=[]
                resCount=0
                capCount=0
                ledCount=0
                icCount=0
                src_img = np.copy(frame)
                img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
                ## Check if resistors
                for templateFile in os.listdir(resTemp):
                    tempName= os.path.join(resTemp, templateFile)
                    tempName=os.path.join(tempName)
                    template = cv2.imread(tempName)
                    c,w, h  = template.shape[::-1]
                    res = cv2.matchTemplate(src_img,template,cv2.TM_CCOEFF_NORMED)           
                    threshold = 0.63
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (255,0,255), 5)
                        resMat.append([pt, (pt[0] + w, pt[1] + h)])
                        resCount=resCount+1
                print 'resistor matches=', resCount
                for resistor in resMat:
                    cv2.rectangle(src_img,resistor[0],resistor[1],(255,255,0),5)
                cv2.imshow("preview", src_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ## Check if caps
                src_img = np.copy(src_img)
                for templateFile in os.listdir(capTemp):        
                    tempName= os.path.join(capTemp, templateFile)
                    template = cv2.imread(tempName)
                    c,w, h = template.shape[::-1]
                    res = cv2.matchTemplate(src_img,template,cv2.TM_CCOEFF_NORMED)
                    threshold = 0.63
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (250,200,0), 1)
                        capMat.append([pt, (pt[0] + w, pt[1] + h)])
                        capCount=capCount+1
                print 'cap matches=', capCount
                for cap in capMat:
                    cv2.rectangle(src_img,cap[0],cap[1],(0,255,0),5)
                cv2.imshow("preview", src_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()                
                ## Check if LEDs
                for templateFile in os.listdir(ledTemp):
                    tempName= os.path.join(ledTemp, templateFile)
                    tempName=os.path.join(tempName)
                    template = cv2.imread(tempName)
                    c,w, h  = template.shape[::-1]
                    res = cv2.matchTemplate(src_img,template,cv2.TM_CCOEFF_NORMED)           
                    threshold = 0.65
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (200,100,100), 5)
                        ledMat.append([pt, (pt[0] + w, pt[1] + h)])
                        ledCount=ledCount+1
                print 'led matches=', ledCount
                for led in ledMat:
                    cv2.rectangle(src_img,led[0],led[1],(0,255,255),5)
                cv2.imshow("preview", src_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ## Check if ICs
                for templateFile in os.listdir(icTemp):
                    tempName= os.path.join(icTemp, templateFile)
                    tempName=os.path.join(tempName)
                    template = cv2.imread(tempName)
                    c,w, h  = template.shape[::-1]
                    res = cv2.matchTemplate(src_img,template,cv2.TM_CCOEFF_NORMED)           
                    threshold = 0.9
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (0,100,250), 5)
                        icMat.append([pt, (pt[0] + w, pt[1] + h)])
                        icCount=icCount+1
                print 'IC matches=', icCount
                for ic in icMat:
                    cv2.rectangle(src_img,ic[0],ic[1],(0,0,0),-1)
                cv2.imshow("preview", src_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return resCount,capCount,ledCount,icCount

def componentLocate(fgmask,frame, resCount,capCount,ledCount,icCount,numRails):
        ############# LOCATION FIND ######################
                if capCount+resCount+ledCount+icCount>0:
                    if icCount>0:
                        xrailMat=[]
                        yrailMat=[]
                        ic=icMat[2]
                        xrailMat.append(ic[0][0])
                        xrailMat.append(ic[1][0])
                        yrailMat.append(ic[0][1])
                        yrailMat.append(ic[1][1])
                    else:
                        xrailMat=[]
                        yrailMat=[]
                        img=np.copy(fgmask)
                        ### get rid of gray shadow
            ##            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ##            lower_red = np.array([0,0,126]) 
            ##            upper_red = np.array([0,0,127])
            ##            mask = cv2.inRange(hsv, lower_red, upper_red)
            ##            mask_inv = cv2.bitwise_not(mask)
            ##            img = cv2.bitwise_and(img,img,mask=mask_inv)
                        cv2.imshow("preview",img)
                        cv2.waitKey(0)
                        kernel_size = 5
                        blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
                        cv2.imshow("preview",blur_gray)
                        cv2.waitKey(0)
                        low_threshold = 50  ##50
                        high_threshold = 150 ##240
                        edges = cv2.Canny(blur_gray, low_threshold, high_threshold,None,3)
                        cv2.imshow("preview",edges)
                        cv2.waitKey(0)
                        # Copy edges to the images that will display the results in BGR         
                        rho = cv2.HOUGH_PROBABILISTIC  # distance resolution in pixels of the Hough grid
                        theta = np.pi / 180  # angular resolution in radians of the Hough grid
                        threshold = 25                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     # minimum number of votes (intersections in Hough grid cell)
                        min_line_length = 1  # minimum number of pixels making up a line
                        max_line_gap = 30  # maximum gap in pixels between connectable line segments
                        line_image = np.copy(frame) * 0  # creating a blank to draw lines on
                        ##lines = cv2.HoughLines(edges, 1, np.pi / 180, 30, None, 0, 0) ##standard
                        img_rgb=frame.copy()
                        lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, theta, threshold, np.array([]),min_line_length, max_line_gap)
                        if lines is not None:
                            for line in lines:
                                for x1,y1,x2,y2 in line:
                                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                                    ##cv2.circle(line_image,(x1,y1),10,(0,255,0),3)
                                    ##cv2.circle(line_image,(x2,y2),10,(0,0,255),3)
                                    xrailMat.append(x1)
                                    xrailMat.append(x2)
                                    yrailMat.append(y1)
                                    yrailMat.append(y2)
                    ##            # Draw the lines on the  image    
                            ##img_rgb=cv2.imread(rgbPath)
                            lines_edges = cv2.addWeighted(img_rgb, 0.8, line_image, 1, 0)
                            cv2.imshow("preview",lines_edges)
                            cv2.waitKey(0)            
                    yMin=np.min(yrailMat)
                    yMax=np.max(yrailMat)
                    xMin=np.min(xrailMat)
                    xMax=np.max(xrailMat)
                    ydist=yMax-yMin
                    xdist=xMax-xMin
                    if xdist > 1.5*ydist:  ## horizontal
                            yavg=0.5*ydist+yMin
                            rail1=railLocate(numRails,xMin,int(np.ceil(yavg)),img_rgb)
                            rail2=railLocate(numRails,xMax,int(np.ceil(yavg)),img_rgb)
                    elif ydist > 1.5*xdist: ## vertical
                            xavg=0.5*xdist+xMin
                            rail1=railLocate(numRails,int(np.ceil(xavg)),yMin,img_rgb)
                            rail2=railLocate(numRails,int(np.ceil(xavg)),yMax,img_rgb)
                    else:  ##angled
                            yavg=0.5*ydist+yMin
                            rail1=railLocate(numRails,xMin,int(np.ceil(yavg)),img_rgb)
                            rail2=railLocate(numRails,xMax,int(np.ceil(yavg)),img_rgb)                    
                    print rail1,rail2
                    return rail1,rail2


def snapshot(str,rval,frame):
    import cv2    
    cv2.imwrite(str,frame)    
    return
                
def railLocate(numOfRails,x,y,img_rgb):
        x=float(x)
        y=float(y)
        rect=pickle.load(open("box.obj","rb"))        
        u2=float(rect[1,0])
        v2=float(rect[1,1])
        u1=float(rect[2,0])
        v1=float(rect[2,1])
        distTot=np.sqrt((v2-v1)**2+(u2-u1)**2)
        ##### LINE EQUATION ########
        ## Line perpendicular to y=(m1)*x+(b1) and passes through (x,y)
        m1=(v1-v2)/(u1-u2)
        b1=v1-m1*u1
        mp=-(1/m1)
        bp=y-mp*x
        xint=(b1-bp)/(mp-m1)
        yint=mp*xint+bp
        cv2.line(img_rgb,(int(u1),int(v1)),(int(u2),int(v2)),(255,0,255),5)
        cv2.line(img_rgb,(int(xint),int(yint)),(int(x),int(y)),(250,100,0),5)
        cv2.circle(img_rgb,(int(xint),int(yint)),10,(200,200,0),10)
        cv2.imshow("preview",img_rgb)
        cv2.waitKey(0)
        now=datetime.datetime.now()
        fileName = 'rail' + now.strftime("%Y-%m-%d %H%M%S")
        completeName = os.path.join(figSavePath, fileName + '.png')
        snapshot(completeName,True,img_rgb)
        ## Distance between xint yint and a1 b1
        distint=np.sqrt((u1-xint)**2+(v1-yint)**2)
        railReturn=np.ceil(numOfRails*(distint/distTot))
        return railReturn

##################################################################    
############# End of Function Defining Section ###################
##################################################################
##################################################################



figPathR = 'C:\Python27\shape-detection\shape-detection\snapshots\R'
figPathB = 'C:\Python27\shape-detection\shape-detection\snapshots\B'
figSavePath = 'C:\Python27\shape-detection\shape-detection\Match'

capTemp=r'C:\Python27\shape-detection\shape-detection\templates\caps'
icTemp=r'C:\Python27\shape-detection\shape-detection\templates\ic'
ledTemp=r'C:\Python27\shape-detection\shape-detection\templates\led'
resTemp=r'C:\Python27\shape-detection\shape-detection\templates\resistors'

numRails=30

#calibrate: camera looks at the blank breadboard 
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
calibrate(frame)
cv2.destroyAllWindows()
vc.release()


#initiate video feed after the calibration is finished
cv2.namedWindow("backgroundSubtract")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
fgbg=cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=75, detectShadows=False)
subFlag=1
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()   
    fgmask = fgbg.apply(frame)    
else:
    rval = False
while rval:
            cv2.imshow("preview", frame)
            cv2.imshow("backgroundSubtract",fgmask)
            rval, frame = vc.read()
            if subFlag==1:
                fgmask = fgbg.apply(frame)
            key = cv2.waitKey(20)    
            if key == 27: # exit on ESC
                cv2.destroyAllWindows()
                vc.release()
                break
            if key == 98: #pause the videoCapture for user to place component
                subFlag=0
            if key == 99: #component has been placed. Now time to detect and locate
                #vc.release()
                subFlag=1
                #wait 3 seconds for the camera to focus
                time.sleep(3)
                rval, frame = vc.read() #read the frame
                fgmask = fgbg.apply(frame) 
                now=datetime.datetime.now()
                # save the real frame and background subtracted frame
                fileName = 'fig' + now.strftime("%Y-%m-%d %H%M%S") 
                completeNameB = os.path.join(figPathB, fileName + 'B.png')
                completeNameR = os.path.join(figPathR, fileName + 'R.png')          
                snapshot(completeNameB,rval,fgmask)
                snapshot(completeNameR,rval,frame)
                #component detection
                resCount,capCount,ledCount,icCount=componentDetect(frame, resTemp, capTemp,icTemp,ledTemp)
                #component location
                rail1,rail2=componentLocate(fgmask,frame, resCount,capCount,ledCount,icCount,numRails)
                # continue while loop for user to add another component.
                
cv2.destroyAllWindows()           
