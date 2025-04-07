from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt") 

# Train the model
results = model.train(data="dataset", epochs=30, imgsz=128,project="sets")

# dataset path ------  project path to save model