#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:12:09 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt


img = cv2.imread("images/input/cat.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


rotated90 = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)

rotated180 = cv2.rotate(rotated90, cv2.ROTATE_180)

plt.figure(figsize=(10,5))

plt.subplot(1,4,1)
plt.imshow(img)
plt.title("OG Image")

plt.subplot(1,4,2)
plt.imshow(rotated)
plt.title("by 90")

plt.subplot(1,4,3)
plt.imshow(rotated90)
plt.title("by 90 again")

plt.subplot(1,4,4)
plt.imshow(rotated180)
plt.title("180 degrees")

plt.show()
