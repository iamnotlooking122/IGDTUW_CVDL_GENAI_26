#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:53:03 2026

@author: akshitmudgal
"""

import cv2
import numpy as np


cap = cv2.VideoCapture("videos/input/parrot.mp4")

ret, prev_frame = cap.read()

ret, curr_frame = cap.read()


THRESHOLD = 25

MIN_AREA = 500

motion_count = 0

while True:
    
    ret, next_frame = cap.read()
    if not ret:
        break
    
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    diff1 = cv2.absdiff(prev_gray, curr_gray)
    diff2 = cv2.absdiff(curr_gray, next_gray)
    
    motion_mask = cv2.bitwise_and(diff1, diff2)
    
    _, thresh = cv2.threshold(motion_mask, THRESHOLD, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations = 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = curr_frame.copy()
    
    motion_now =False
    
    for contour in contours:
        
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        
        motion_now = True
        motion_count+=1
        
        x, y, w, h = cv2.boundingRect(contour)
        
        cv2.rectangle(output, (x,y), (x+w, y+h), (0,0,255), 2)
        
        cv2.putText(output, "MOTION DETECTED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        
        
    status = "MOTION DETECTED" if motion_now else "NO MOTION"
    
    satus_color=(0,0,255)  if motion_now else (0,255,0)
    
    cv2.putText(output, f"Events: {motion_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    
    
    #DULA DISPLAY
    
    thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([output, thresh_colored])
    combined = cv2.resize(combined, (1200,400))
    
    
    cv2.imshow("MOTION DETECTION LEFT FEED RIGHT FEED-MOTION MAKS", combined )
    
    prev_frame = curr_frame
    curr_frame = next_frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release() 
cv2.destroyAllWindows()
    