#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:09:27 2026

@author: akshitmudgal
"""

#HOG+SVM SE HUMAN DETECTION

#HOG DOESNT CARE ABOUT COLOR OR LIGHTING - ONLY SHAPE
#A HUMAN BODY HAS A VERY RCEOGNISABLE HOG FINGERPRINT

#HOG + SVM
#FEATURE EXTRCAXTOR
#CLASSIFIER

#CONFIDENCE THRESHOLD

import cv2

VIDEO_SOURCE = "videos/input/couple_walking_snow.mp4"

CONFIDENCE_THRESHOLD = 0.5

#LOA D THE HOG DETECTOR

hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Could nt find teh source {VIDEO_SOURCE}")
    exit()
    
    
print("")
print("")

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        print("Cannot read fraem | video endeded")
        break
    
    
    
    small = cv2.resize(frame, (640, 480))
    ratio = frame.shape[0]/480 #1080/480 => 2.xx
    
    boxes, weights = hog.detectMultiScale(small, winStride=(8,8), padding=(4,4), scale = 1.05)
    
    accepted_count = 0
    rejected_count = 0
    
    for i, (x,y,w,h) in enumerate(boxes):
        
        confidence = float(weights[i])
        
        x  = int(x * ratio)
        y  = int(y * ratio)
        w =  int(w * ratio)
        h = int(h * ratio)
        
        if confidence < CONFIDENCE_THRESHOLD:
            rejected_count+=1
            
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255),1)
            
            cv2.putText(frame, f" Rejected {confidence: .2f}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            
            continue
        
        accepted_count +=1
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
        cv2.putText(frame, f" Person {confidence: .2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
     #cv2.putText(frame, f" People Detected {accepted_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
     #cv2.putText(frame, f" Rejected {rejected_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)


    cv2.imshow("HOG -PERSON DETECTOR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release() 
cv2.destroyAllWindows()
    
        
    
    