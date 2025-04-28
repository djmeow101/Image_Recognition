from ultralytics import YOLO
import pytest
import os

# Adjust this path if necessary
MODEL_PATH = 'sets/train5/weights/best.pt'

def load_model(model_path):
    """Load the YOLOv11 classification model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = YOLO(model_path)  # Use the YOLO class from ultralytics
    return model

def test_model_loads():
    """Test if the model loads correctly."""
    model = load_model(MODEL_PATH)
    assert model is not None, "Model failed to load."

def test_model_predicts():
    """Test if the model can make a prediction on an example image."""
    model = load_model(MODEL_PATH)
    results = model("images (1).jpg")  # Use an image path
    assert results is not None, "Model did not return a prediction."
