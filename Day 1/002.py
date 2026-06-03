#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:49:06 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/input/cat.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

bright = cv2.add(rgb, 50)

plt.imshow(bright)
plt.show()