"""
training using  the Ultralytics library.
"""

from ultralytics import YOLO


def train_model():
    """
    Loads the YOLO model and starts the training process.

    - Model: yolo11n.pt
    - Dataset: dataset_3/data.yaml
    - Epochs: 50
    - Image Size: 640
    - Device: GPU 0
    """
    model = YOLO("yolo11n.pt")  # Replace with the path to your custom model if needed

    results = model.train(
        data="dataset_3/data.yaml",
        epochs=50,
        imgsz=640,
        project="box_sets",
        device='0'
    )
    return results


if __name__ == '__main__':
    train_model()