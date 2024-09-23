# src/evaluate.py

from keras.models import load_model
from src.prep import load_images

def evaluate_model(model_path, test_data_dir, img_size=(100, 100)):
    """
    Evaluate the model on the test dataset.

    Parameters:
        - model_path (str): Path to the trained model file.
        - test_data_dir (str): Path to the test dataset directory.
        - img_size (tuple): The size to resize the images (default (100, 100)).
    """
    
    # Load test data
    print("[INFO] Loading test data...")
    X_test, y_test, _ = load_images(test_data_dir, img_size)
    
    # Load the trained model
    print(f"[INFO] Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Evaluate the model
    print("[INFO] Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

