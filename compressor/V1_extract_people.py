import cv2
from ultralytics import YOLO
import os
from glob import glob

model = YOLO("yolov8x.pt")  

input_dir = "cl_img"
output_dir = "cropped_people"
os.makedirs(output_dir, exist_ok=True)

image_paths = glob(os.path.join(input_dir, "*.*"))
supported_exts = (".jpg", ".jpeg", ".png", ".bmp")

for image_path in image_paths:
    if not image_path.lower().endswith(supported_exts):
        continue

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to read {image_path}")
        continue

    results = model(image)[0]

    person_count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf > 0.5: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = image[y1:y2, x1:x2]
            output_filename = f"{image_name}_person_{person_count}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, person_crop)
            person_count += 1

    print(f"{person_count} people cropped from '{image_path}'")
