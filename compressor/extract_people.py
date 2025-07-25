import cv2
from ultralytics import YOLO
import os
from glob import glob

# Load YOLOv8 model
model = YOLO("yolov8x.pt")  # You can also use yolov8s.pt, yolov8m.pt, etc.

# Input and output directories
input_dir = "cl_img"
output_dir = "cropped_people"
os.makedirs(output_dir, exist_ok=True)

# Get all image files from the input directory
image_paths = glob(os.path.join(input_dir, "*.*"))
supported_exts = (".jpg", ".jpeg", ".png", ".bmp")

# Process each image
for image_path in image_paths:
    if not image_path.lower().endswith(supported_exts):
        continue

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to read {image_path}")
        continue

    # Run detection
    results = model(image)[0]

    # Loop through detections
    person_count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf > 0.5:  # Person class ID in COCO
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = image[y1:y2, x1:x2]
            output_filename = f"{image_name}_person_{person_count}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, person_crop)
            person_count += 1

    print(f"{person_count} people cropped from '{image_path}'")
