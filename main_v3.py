### User places one component, then the component is detected,
### then finds location
### Created 2018-11-18

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


############## Defining Functions ##############################

numRails=30


def railLocate(numOfRails,x,y,img_rgb):
        x=float(x)
        y=float(y)
        rect=pickle.load(open("approx.p","rb"))
        u2=float(rect[0,0,0])-25
        v2=float(rect[0,0,1])
        u1=float(rect[1,0,0])+15
        v1=float(rect[1,0,1])
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

    
############# End of Function Defining Section ################
figPathR = 'C:\Python27\shape-detection\shape-detection\snapshots\R'
figPathB = 'C:\Python27\shape-detection\shape-detection\snapshots\B'
figSavePath = 'C:\Python27\shape-detection\shape-detection\Match'
count=0
#Step 1: initiate video feed, background subtraction, take snapshot
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
    if key == 98: #pause the video
        #key2=cv2.waitKey(0)
        subFlag=0
    if key == 99:
        subFlag=1
        time.sleep(3)
        rval, frame = vc.read()
        fgmask = fgbg.apply(frame)
        from snapshot import snapshot
        now=datetime.datetime.now()
        fileName = 'fig' + now.strftime("%Y-%m-%d %H%M%S")
        completeNameB = os.path.join(figPathB, fileName + 'B.png')
        completeNameR = os.path.join(figPathR, fileName + 'R.png')          
        snapshot(completeNameB,rval,fgmask)
        snapshot(completeNameR,rval,frame)
        count=count+1

        ### Detect Components in frame
        cv2.destroyAllWindows()
        resTemp=r'C:\Python27\shape-detection\shape-detection\templates\resistors'
        resMat=[]
        capMat=[]
        ledMat=[]
        icMat=[]
        resCount=0
        capCount=0
        ledCount=0
        icCount=0
        capTemp=r'C:\Python27\shape-detection\shape-detection\templates\caps'
        icTemp=r'C:\Python27\shape-detection\shape-detection\templates\ic'
        ledTemp=r'C:\Python27\shape-detection\shape-detection\templates\led'
        img_rgb=frame
        img_back=fgmask
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        src_img = np.copy(img_rgb)        
        ## Check if resistors
        for templateFile in os.listdir(resTemp):
            tempName= os.path.join(resTemp, templateFile)
            tempName=os.path.join(tempName)
            template = cv2.imread(tempName)
            c,w, h  = template.shape[::-1]
            res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)           
            threshold = 0.63
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (200,0,100), 1)
                resMat.append([pt, (pt[0] + w, pt[1] + h)])
                resCount=resCount+1
        print 'resistor matches=', resCount
        for resistor in resMat:
            cv2.rectangle(img_back,resistor[0],resistor[1],(0,0,0),-1)
        cv2.imshow("preview", src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ## Check if caps
        src_img = np.copy(img_rgb)
        for templateFile in os.listdir(capTemp):        
            tempName= os.path.join(capTemp, templateFile)
            template = cv2.imread(tempName)
            c,w, h = template.shape[::-1]
            res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.63
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (250,200,0), 1)
                capMat.append([pt, (pt[0] + w, pt[1] + h)])
                capCount=capCount+1
        print 'cap matches=', capCount
        for cap in capMat:
            cv2.rectangle(img_back,cap[0],cap[1],(0,0,0),-1)
        cv2.imshow("preview", src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        ## Check if LEDs
        for templateFile in os.listdir(ledTemp):
            tempName= os.path.join(ledTemp, templateFile)
            tempName=os.path.join(tempName)
            template = cv2.imread(tempName)
            c,w, h  = template.shape[::-1]
            res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)           
            threshold = 0.65
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (200,0,100), 1)
                ledMat.append([pt, (pt[0] + w, pt[1] + h)])
                ledCount=ledCount+1
        print 'led matches=', ledCount
        for led in ledMat:
            cv2.rectangle(img_back,led[0],led[1],(0,0,0),-1)
        cv2.imshow("preview", src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ## Check if ICs
        for templateFile in os.listdir(icTemp):
            tempName= os.path.join(icTemp, templateFile)
            tempName=os.path.join(tempName)
            template = cv2.imread(tempName)
            c,w, h  = template.shape[::-1]
            res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)           
            threshold = 0.9
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (0,100,100), 1)
                icMat.append([pt, (pt[0] + w, pt[1] + h)])
                icCount=icCount+1
        print 'IC matches=', icCount
        for ic in icMat:
            cv2.rectangle(img_back,ic[0],ic[1],(0,0,0),-1)
        cv2.imshow("preview", src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
                img=np.copy(img_back)
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
                line_image = np.copy(img_rgb) * 0  # creating a blank to draw lines on
                ##lines = cv2.HoughLines(edges, 1, np.pi / 180, 30, None, 0, 0) ##standard
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
                


        
                
cv2.destroyAllWindows()

            
