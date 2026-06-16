#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:31:14 2026

@author: akshitmudgal
"""

#face detection - HAAR CASCADE Classifier

"""
pre tarined classifier
trained on thousand of faces
sliding window, is this is face?
haar like features
fast

"""

import cv2

#Viola-jonas
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#eye detectoe laod
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")



# WEBCAM OPEN

cap = cv2.VideoCapture("videos/input/people_walking.mp4")
print("Fcae detector running....Press Q key to quit")

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    
    #grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    #draw rectangle around it
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "FACE", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        #REGION OF INTEREST
        
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 4)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(face_roi_color, (ex+ew//2, ey+eh//2), ew//2, (255,0,0), 2)
            
    cv2.putText(frame, f"Faces:{len(faces)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    cv2.imshow("HAAR CASCADE - FACE DETECTORE", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release() 
cv2.destroyAllWindows()
    