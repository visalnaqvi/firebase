from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained on COCO dataset
model = YOLO("yolov8n.pt")  # you can also use yolov8s.pt, yolov8m.pt, etc.

# Load image from laptop
image_path = "673715637220bc664ed24ba2.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# Run YOLO detection
results = model(img)[0]

# Print all detections
print("\nDetections:")
for box in results.boxes:
    cls_id = int(box.cls[0])       # class index
    label = results.names[cls_id]  # class name
    conf = float(box.conf[0])      # confidence
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box

    print(f"- {label} (class {cls_id}) "
          f"with {conf:.2f} confidence "
          f"at [{x1}, {y1}, {x2}, {y2}]")

# OPTIONAL: Draw boxes and save visualization
for box in results.boxes:
    cls_id = int(box.cls[0])
    label = results.names[cls_id]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("detections.jpg", img)
print("\nSaved annotated image as detections.jpg")
