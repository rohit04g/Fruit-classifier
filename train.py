# src/train.py

from src.prep import load_images, split_data
from src.model import build_model
from keras.callbacks import ModelCheckpoint

def train_model(train_data_dir, model_save_path, img_size=(100, 100), epochs=10, batch_size=32):
    """
    Train a fruit classifier model.

    Parameters:
        - train_data_dir (str): Path to the training data.
        - model_save_path (str): Path to save the trained model.
        - img_size (tuple): The size to resize the images (default (100, 100)).
        - epochs (int): Number of epochs to train the model (default 10).
        - batch_size (int): Batch size for training (default 32).
    """

    # Step 1: Load and preprocess data
    print("[INFO] Loading and preprocessing data...")
    X, y, fruit_classes = load_images(train_data_dir, img_size)

    # Step 2: Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Step 3: Build the model
    print("[INFO] Building the model...")
    input_shape = (img_size[0], img_size[1], 3)  # Image size with 3 channels (RGB)
    num_classes = len(fruit_classes)
    model = build_model(input_shape, num_classes)

    # Step 4: Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Step 5: Set up a checkpoint to save the best model
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)

    # Step 6: Train the model
    print("[INFO] Training the model...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

    # Step 7: Save the final model
    model.save(model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")

    return history
