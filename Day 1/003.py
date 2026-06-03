#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:52:19 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("images/input/brain.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


gray_map = np.zeros((img.shape[0], img.shape[1]),dtype = np.uint8)

cv2.circle(gray_map, center=(320, 180), radius= 60, color=255, thickness = -1)

heatmap = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)

heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

result = cv2.addWeighted(img, 0.7, heatmap, 0.3, 50)

print(gray_map.shape)

plt.imshow(result)

plt.show()
