# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:28:55 2025

@author: Afaq Ahmad
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import AlexNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import cv2
import matplotlib.pyplot as plt
from skimage import filters

# Define dataset path
output_folder = 'Numbereddata_classified_images'
categories = ['Level-01', 'Level-02', 'Level-03', 'Level-04', 'Level-05',
              'Level-06', 'Level-07', 'Level-08', 'Level-09', 'Level-10']

# Load images and assign labels based on folder names
datagen = ImageDataGenerator(rescale=1./255)
imds = datagen.flow_from_directory(output_folder, target_size=(227, 227), batch_size=32, class_mode='categorical')

# Load AlexNet model
net = AlexNet(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

# Extract features using the fully connected layer 'fc7'
feature_layer = 'fc7'
training_features = net.predict(imds)

# Get labels
training_labels = imds.classes

# Train SVM classifier using extracted features
classifier = OneVsRestClassifier(SVC(kernel='linear'))
classifier.fit(training_features, training_labels)

# Load and preprocess test image
def process_image(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray_img

# Corrosion detection
def detect_corrosion(gray_img, technique='Canny'):
    if technique == 'Canny':
        edge_map = filters.canny(gray_img)
    elif technique == 'Sobel':
        edge_map = filters.sobel(gray_img)
    elif technique == 'Prewitt':
        edge_map = filters.prewitt(gray_img)
    elif technique == 'Roberts':
        edge_map = filters.roberts(gray_img)
    else:
        raise ValueError('Unknown technique. Use Canny, Sobel, Prewitt, or Roberts.')

    corrosion_mask = cv2.dilate(edge_map.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    corrosion_mask = cv2.morphologyEx(corrosion_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    corrosion_mask = cv2.morphologyEx(corrosion_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return corrosion_mask, edge_map

# Predict corrosion level
def predict_corrosion_level(img_path):
    img, gray_img = process_image(img_path)
    corrosion_mask, edge_map = detect_corrosion(gray_img)
    corrosion_area_pixels = np.sum(corrosion_mask)
    total_pixels = corrosion_mask.size
    corrosion_percentage = (corrosion_area_pixels / total_pixels) * 100

    img_resized = cv2.resize(img, (227, 227))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_features = net.predict(img_resized)
    predicted_label = classifier.predict(img_features)

    return img, edge_map, corrosion_mask, corrosion_percentage, predicted_label

# Display results
def display_results(img, edge_map, corrosion_mask, corrosion_percentage, predicted_label):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.subplot(2, 2, 2)
    plt.imshow(edge_map, cmap='gray')
    plt.title('Edge Detection')
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.contour(corrosion_mask, colors='r', linewidths=2)
    plt.title(f'Corrosion Area: {corrosion_percentage:.2f}%')
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'AlexNet Classification Result: {predicted_label[0]}')
    plt.show()

# Example usage
img_path = 'testimage.jpg'
img, edge_map, corrosion_mask, corrosion_percentage, predicted_label = predict_corrosion_level(img_path)
display_results(img, edge_map, corrosion_mask, corrosion_percentage, predicted_label)