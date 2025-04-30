from ultralytics import YOLO
import cv2

# Load model
model = YOLO("box_sets/train12/weights/best.pt")

cap = cv2.VideoCapture("cat.mp4")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter('output_with_boxes.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Get the first result (since it's per-frame)
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()  # xyxy boxes
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    names = result.names

    # Draw bounding boxes
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[cls_id]} {conf:.2f}"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Write frame to output
    out.write(frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video processing complete! Output saved as 'output_with_boxes.mp4'")