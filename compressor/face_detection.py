import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# ==== CONFIG ====
PADDING = 0.25  # 25% padding around the face box
MODEL = "buffalo_l"
INPUT_DIR = "compressed_img"   # input folder
OUTPUT_ANNOTATED_DIR = "face_marked"   # annotated images folder
OUTPUT_FACES_DIR = "faces_cropped"     # cropped faces folder

# Create output dirs if not exist
os.makedirs(OUTPUT_ANNOTATED_DIR, exist_ok=True)
os.makedirs(OUTPUT_FACES_DIR, exist_ok=True)

# Initialize InsightFace
app = FaceAnalysis(name=MODEL, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Get all images in input folder
valid_exts = (".jpg", ".jpeg", ".png", ".jfif", ".bmp")
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]

print(f"Found {len(image_files)} images in {INPUT_DIR}\n")

for img_name in image_files:
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not load {img_name}, skipping...")
        continue

    h, w, _ = img.shape

    # Detect faces
    faces = app.get(img)
    print(f"[{img_name}] Detected {len(faces)} faces")

    # Copy for annotation
    annotated_img = img.copy()

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox)
        score = face.det_score  # detection confidence

        # Add padding
        face_w, face_h = x2 - x1, y2 - y1
        pad_w, pad_h = int(face_w * PADDING), int(face_h * PADDING)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmarks
        if face.landmark is not None:
            for (lx, ly) in face.landmark.astype(int):
                cv2.circle(annotated_img, (lx, ly), 2, (0, 0, 255), -1)

        # Label face index + confidence
        label = f"Face {i+1}: {score:.2f}"
        print(label)
        cv2.putText(annotated_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save cropped face
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size > 0:
            base_name, _ = os.path.splitext(img_name)
            out_face_path = os.path.join(OUTPUT_FACES_DIR, f"{base_name}_face_{i+1}.jpg")
            cv2.imwrite(out_face_path, face_crop)

    # Save annotated image
    annotated_out_path = os.path.join(OUTPUT_ANNOTATED_DIR, f"{img_name}_marked.jpg")
    cv2.imwrite(annotated_out_path, annotated_img)
    print(f"Saved annotated image with faces -> {annotated_out_path}")

print("\n🎉 All images processed!")
