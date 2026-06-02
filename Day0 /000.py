#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:52:29 2026

@author: akshitmudgal
"""
import cv2 #image processing lib
import matplotlib.pyplot as plt

#how to read an image

img = cv2.imread("images/input/cat.jpg")


#opencv reads images as BGR, Matplotlib expects RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(rgb_img.shape)

plt.imshow(rgb_img) #load pixel values from memory 

plt.show() #displays



