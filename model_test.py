import os
import cv2
import pytest
from ultralytics import YOLO

@pytest.mark.skipif(not os.path.exists("best.pt") or not os.path.exists("cat.mp4"), reason="Model or video file not found")
def test_yolo_video_processing(tmp_path):
    # Load YOLO model
    model = YOLO("best.pt")

    # Open video
    cap = cv2.VideoCapture("cat.mp4")
    assert cap.isOpened(), "Failed to open input video."

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert width > 0 and height > 0 and fps > 0, "Invalid video properties."

    # Output path
    output_path = tmp_path / "output_with_boxes.mp4"

    # Output writer
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Process only a few frames (to keep test fast)
    frame_count = 0
    max_frames = 5  # only process a few frames for testing

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names

        # Draw boxes
        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        out.write(frame)
        frame_count += 1

    # Cleanup
    cap.release()
    out.release()

    # Check that output file exists and is not empty
    assert output_path.exists(), "Output video was not created."
    assert output_path.stat().st_size > 0, "Output video is empty."

    print(f"✅ Test completed! Video saved at {output_path}")
