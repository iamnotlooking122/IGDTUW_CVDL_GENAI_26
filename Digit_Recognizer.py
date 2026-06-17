#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:52:18 2026

@author: akshitmudgal
"""

#HANDWRIITEN DIGIT RECOGNIZER
#ANN, OPENCV, MATPLOTLIB, SEABORN


import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras


print("=" * 65)
print("    HANDWRIITEN DIGIT RECOGNIZER - ANN MINI PROJECT")
print("=" * 65)

#STEP1: LOAD AND EXPLORE THE DATA

print("\n STEP1 : LAODING MNIST DATASET...")
print("=" * 65)

(X_train, y_train), (X_test, y_test)= keras.datasets.mnist.load_data()

print(f"Training images: {X_train.shape} --> {X_train.shape[0]} images, each {X_train.shape[1]} * {X_train.shape[2]} pixels")

print(f"Trainng Labels: {y_train.shape} --> {y_train.shape[0]} numbers (0-9)")

print(f"Test images : {X_test.shape}")

print(f"Pixel Range : [{X_train.min()}, [{X_train.max()}]")


fig, axes = plt.subplots(1, 10, figsize= (15,2))
fig.suptitle("MNIST SAMPLE HANDWRITTEN DATASET", fontsize = 12, fontweight = 'bold')

for digit in range(10):
    
    idx = np.where(y_train == digit)[0][0]
    axes[digit].imshow(X_train[idx], cmap ='gray')
    axes[digit].set_title(f"Label: {digit}", fontsize = 9, fontweight= 'bold')
    axes[digit].axis('off')
    
plt.tight_layout()

plt.savefig('project_step1_dataset.png', dpi=150, bbox_inches = 'tight')

plt.show()
print("DATSET VISULASIATION SAVED")


#STEP2: PREPROCESSING
print("\n STEP2 : preprocessing images...")
print("=" * 40)

sample_img = X_train[0]
print(f"Original SHAPE: {sample_img.shape}  <= 28 by 28 ")
sample_flat = sample_img.flatten()
print(f"After flatten {sample_flat.shape} <= 784 numbers in  a row ")

#(60000, 28 ,28) -> (60000, 784)

X_train_flat = X_train.reshape(-1, 784)
X_test_flat = X_test.reshape(-1, 784)


#normalize
X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0

X_train_final = X_train_norm[: 50000]
y_train_final = y_train [:50000]
X_val = X_train_norm[50000:]
y_val = y_train[50000:]

print("preprocessing completed..")
print(f" Train: {X_train_final.shape}, Validation: {X_val.shape}, Test: {X_test_norm.shape}")

#step3: ANN
print("\n STEP3 : BUILDING THE ANN...")
print("=" * 40)


model = keras.Sequential([
    #INPUT -> HIDDEN LAYER 1
    keras.layers.Dense(256, activation='relu', input_shape=(784,), name = 'hidden_1'),
    keras.layers.BatchNormalization(name="batchnorm_1"),
    keras.layers.Dropout(0.3, name='dropout_1' ),
    
    
    #HIDDEN LAYER 1  -> HIDDEN LAYER 2
    keras.layers.Dense(128, activation='relu', name = 'hidden_2'),
    keras.layers.BatchNormalization(name="batchnorm_2"),
    keras.layers.Dropout(0.3, name='dropout_2' ),
    
    #HIDDEN LAYER 2 --> HIDDEN LAYER 3
    keras.layers.Dense(64, activation='relu', name = 'hidden_3'),
    #keras.layers.BatchNormalization(name="batchnorm_3"),
    keras.layers.Dropout(0.3, name='dropout_3' ),
    
    #---OUTPUT LAYER---
    keras.layers.Dense(10, activation='softmax', name = 'output'),
    
    
    
    
    
    ])

model.summary()

total_params = model.count_params()
print(f"/n Total parameters the n/w will learns:{total_params}")


#step4: compile and train

print("\n STEP4 : compiling and training")
print("=" * 40)

model.compile(
    optimizer =keras.optimizers.Adam(learning_rate = 0.001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    
    )

early_stop = keras.callbacks.EarlyStopping(
    monitor ='val_accuracy',
    patience = 8,
    restore_best_weights = True,
    verbose = 1)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor = 0.5, #0.0005
    patience = 4,
    min_lr= 1e-6, # 1` * 10 ^-6 = 0.000001
    verbose = 1)

print("\n Training ...(~1-3 minutes depending upon ur hardware_ \n")

history = model.fit(
    
        X_train_final, y_train_final,
        epochs = 50,
        batch_size = 128,
        validation_data= (X_val, y_val),
        callbacks = [early_stop, lr_scheduler],
        verbose = 1
    )

print("\n Traing is completed")




