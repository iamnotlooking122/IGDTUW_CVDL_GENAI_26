#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:24:59 2026

@author: akshitmudgal
"""

import cv2

import matplotlib.pyplot as plt

#Read Image

img = cv2.imread("images/input/cb1.jpg")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

small = cv2.resize(img_rgb, (40,40))

print(small)

print("the fisrt pixel", small[0,0])

small[0,0] = [255,0,0]
small[0,1] = [0,255,0]
small[0,2] = [0,0,255]
small[0,3]= [255,0,0]
small[0,4] = [0,255,0]
small[1,0] =  [0,255,0]
print("shape of chess board ", small.shape)
plt.imshow(small)
plt.title("modified checkboard")
plt.axis("off")
plt.show()

