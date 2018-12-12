# -*- coding: utf-8 -*-
"""
Created on Wed May 16 07:49:49 2018

@author: Alexander
"""

import cv2
import math
import numpy
import time
print cv2.__version__

cap = cv2.VideoCapture('vid2.mp4')

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break
    # Our operations on the frame come here
        
    frame = frame[2:531, 2:639,:]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    maskimg = numpy.ones(frame.shape)
    maskimg = maskimg[:,:,0] * mask
    maskimg=maskimg.astype(numpy.float32)
    maskimg = cv2.cvtColor(maskimg,cv2.COLOR_GRAY2BGR)

    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        sortedcont = sorted(contours, key= lambda cont: cv2.arcLength(cont,False),reverse = True)

        newcont = list()
        for i in range(len(sortedcont)):
            if cv2.arcLength(sortedcont[i],False) < 70:
                break
            newcont.append(sortedcont[i])
        
        for contour in newcont:
            hull = cv2.convexHull(contour,returnPoints = False)
            defects = cv2.convexityDefects(contour,hull)
            fingers = list()
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                #cv2.line(maskimg,start,end,[0,255,0],2)
                dist = math.hypot(end[0] - start[0], end[1] - start[1])
                angle = 2*numpy.arctan(dist/(2*(d/256.0)))
                if angle < 90*(math.pi/180):
                    #cv2.circle(maskimg,far,5,[0,0,255],-1)
                    cv2.circle(maskimg,start,5,[255,0,0],-1)
                    cv2.circle(maskimg,end,5,[255,0,0],-1)
                    fingers.append(start)
                    fingers.append(end)
                    
            if len(fingers) == 2:
                cv2.line(maskimg,fingers[0],fingers[1],[0,255,0],2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(maskimg,'Two finger Zoom',fingers[0], font, 0.4,(0,0,255),1,cv2.LINE_AA)
                if fingers[0][0] < 320:
                    cv2.putText(maskimg,'L',fingers[1], font, 1,(0,255,255),3,cv2.LINE_AA)
                else:
                    cv2.putText(maskimg,'R',fingers[1], font, 1,(255,0,255),3,cv2.LINE_AA)
        #maskimg = cv2.drawContours(maskimg, newcont, -1, (0,255,0), 3)




#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',maskimg)
    time.sleep(0.025)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()