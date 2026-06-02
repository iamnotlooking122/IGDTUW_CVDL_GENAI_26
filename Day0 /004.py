#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:57:51 2026

@author: akshitmudgal
"""

import numpy as np

import matplotlib.pyplot as plt

img = np.zeros((3,3), dtype = np.uint8)

img[0,0]= 50 
img[1,1] = 150 
img[2,2] = 200 

print("GS MATRIX ")

print(img)
plt.title("GRAYUSCALE IMAGE 3 BY 3")


plt.imshow(img, cmap="gray")

plt.show()

