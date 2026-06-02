#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:41:34 2026

@author: akshitmudgal
"""

import numpy as np

import matplotlib.pyplot as plt

#create an image of size 20 by 20

img = np.zeros((20,20), dtype = np.uint8)

img[2, 5:15] = 255

img[10, 5:15] = 255

img[18, 5:15] = 255

img[2:10, 15] = 255

img[10:19, 15] = 255

print(img.shape)

print(img)

plt.imshow(img, cmap="gray")


plt.title("DIGIT 3")
plt.show()