#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:04:57 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt

#read image
img =cv2.imread("images/input/cat.jpg")

#convert BGR TO RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#INCREASE BRIGHTNESS

bright = cv2.convertScaleAbs(rgb, alpha = 2, beta = 100)

print("orginal 00 pixel ", rgb[0,0])

print("bright 00 pixel ", bright[0,0])

#display original image

plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
plt.imshow(rgb)
plt.title("Oriinal image")

plt.subplot(1,2,2)
plt.imshow(bright)
plt.title("Brightness +50")

plt.show()
