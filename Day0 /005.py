#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:07:32 2026

@author: akshitmudgal
"""

import cv2

import matplotlib.pyplot as plt


img = cv2.imread("images/input/cat.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("ORIGINAL RGB SHAPE", rgb.shape)
#convert rgb to grauscale


#rgb to gray

gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
print("GRAYSCALE", gray.shape)

#gray to rgb

gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# figsize parameter is used to control the width and height of a plot

plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.imshow(rgb)
plt.title("ORIGINAL RGB")

plt.subplot(1,3,2)
plt.imshow(gray, cmap="gray")

plt.subplot(1,3,3)
plt.imshow(gray_rgb)

plt.show()



