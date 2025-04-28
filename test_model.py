# test_model.py

import torch
import pytest
import os
from PIL import Image
import torchvision.transforms as transforms

# Adjust this path if necessary
MODEL_PATH = 'sets/train5/weights/best.pt'

def load_model(model_path):
    """Load the YOLO classification model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='github')
    return model

def prepare_image(image_path):
    """Load and preprocess an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def test_model_loads():
    """Test if the model loads correctly."""
    model = load_model(MODEL_PATH)
    assert model is not None, "Model failed to load."

def test_model_predicts():
    """Test if the model can make a prediction on an example image."""
    model = load_model(MODEL_PATH)
    # Use one of the sample images you uploaded
    image_tensor = prepare_image('image.jpg')
    preds = model(image_tensor)
    assert preds is not None, "Model did not return a prediction."
