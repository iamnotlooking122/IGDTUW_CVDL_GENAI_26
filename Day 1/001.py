#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:41:19 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/input/cat.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#ADDITION

addition = cv2.convertScaleAbs(rgb, alpha = 1, beta = 50)


subtraction = cv2.convertScaleAbs(rgb, alpha = 1, beta = -50)

multiplication = cv2.convertScaleAbs(rgb, alpha = 1.5, beta= 0)

division = (rgb // 2)

plt.figure(figsize = (15,10))

plt.subplot(2,3, 1)
plt.imshow(rgb)
plt.title("Original iMage")
plt.imshow(rgb)

#addition show
plt.subplot(2,3,2)
plt.imshow(addition)
plt.title("Brightness + 50")

#subtrcation show

plt.subplot(2,3,3)
plt.imshow(subtraction)
plt.title("Subtraction - 50")

plt.subplot(2,3,4)
plt.imshow(multiplication)
plt.title("contrast * 1.5")

plt.subplot(2,3,5)
plt.imshow(division)
plt.title("DIVISION // 2")

plt.show()