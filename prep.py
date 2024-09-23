# src/data_preprocessing.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(data_dir, img_size=(100, 100)):
    data = []
    labels = []
    fruit_classes = os.listdir(data_dir)
    
    for fruit in fruit_classes:
        path = os.path.join(data_dir, fruit)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, img_size)
            image = image / 255.0  # Normalize pixel values
            data.append(image)
            labels.append(fruit_classes.index(fruit))  # Assign numeric labels
            
    return np.array(data), np.array(labels), fruit_classes

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
