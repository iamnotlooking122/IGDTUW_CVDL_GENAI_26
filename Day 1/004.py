#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:06:03 2026

@author: akshitmudgal
"""

#cropping

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/input/cat.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

crop = img[250:400, 200:450]

plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original image")

plt.subplot(1,2,2)
plt.imshow(crop)
plt.title("Cat Cropped")

plt.show()