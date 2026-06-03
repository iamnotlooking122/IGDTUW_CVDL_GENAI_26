#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:25:54 2026

@author: akshitmudgal
"""


import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/input/lowlight.jpg")


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

enhanced = cv2.convertScaleAbs(img, alpha = 1.5, beta = 50)

plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(enhanced)
plt.title("image enhnaced")


plt.show()