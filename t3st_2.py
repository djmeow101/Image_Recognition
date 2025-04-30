from ultralytics import YOLO
import cv2

# Load model
model = YOLO("box_sets\\train13\\weights\\best.pt")

# Path to input image
image_path = "dalek.jpg"

# Read the image with OpenCV
img = cv2.imread(image_path)

# Predict
results = model(image_path)

# Loop through detections (there’s usually only one result in the list for a single image)
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class indices
    names = result.names

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[class_id]} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        # Put label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# Save output image
output_path = "output.jpg"
cv2.imwrite(output_path, img)

print(f"Saved image with boxes to: {output_path}")
