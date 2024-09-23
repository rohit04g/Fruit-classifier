# src/predict.py

import cv2
import numpy as np
from keras.models import load_model

def predict_fruit(image_path, model_path, img_size=(100, 100)):
    """
    Predict the type of fruit in a given image.

    Parameters:
        - image_path (str): Path to the image file for prediction.
        - model_path (str): Path to the trained model file.
        - img_size (tuple): The size to resize the image (default (100, 100)).
    
    Returns:
        - predicted_class (str): The predicted class of the fruit.
    """
    
    # Load the trained model
    print(f"[INFO] Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load and preprocess the image
    print(f"[INFO] Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Image not found: {image_path}")
        return None
    
    image = cv2.resize(image, img_size)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize image
    
    # Predict the class
    print("[INFO] Making prediction...")
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    
    # Load fruit classes
    from src.prep import load_images
    _, _, fruit_classes = load_images('./fruits-360/Training')  # Path to training data
    predicted_class = fruit_classes[predicted_class_index]
    
    print(f"Predicted Fruit: {predicted_class}")
    return predicted_class

