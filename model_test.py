from ultralytics import YOLO
model = YOLO("sets\\train5\\weights\\best.pt") # path to trained model
results = model("images (1).jpg") # image path