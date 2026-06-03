#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:39:58 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("images/input/gs.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(img.shape)

print(hsv.shape)

#detectiing lower bound, upper bound(range of green area)

lower_green= np.array([35,40,40])

upper_green = np.array([85, 255, 255])

#creating a mask

mask = cv2.inRange(hsv, lower_green, upper_green)

#inverted mask for object selection
mask_inv = cv2.bitwise_not(mask)


#extracting the person, applying inverted mask
person = cv2.bitwise_and(img, img, mask=mask_inv)

#reading the car background
car = cv2.imread("images/input/cbg.jpg")


car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)

#resize for adding

car = cv2.resize(car, (img.shape[1], img.shape[0]))


print(car.shape)
#applying mask tocar image
car_bg = cv2.bitwise_and(car, car, mask = mask)

#final image addition
final = cv2.add(person, car_bg)

plt.imshow(final)



final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
#for saving the image
cv2.imwrite("output.jpg", final)



plt.title("final image composition")

plt.show()
         

