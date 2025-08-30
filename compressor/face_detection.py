import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ==== CONFIG ====
PADDING = 0.25  # 25% padding around the face box
MODEL = "buffalo_l"
INPUT_IMG = "download (1).jfif"
OUTPUT_ANNOTATED = "faces_marked.jpg"
OUTPUT_DIR = "faces_cropped"

# Initialize InsightFace
app = FaceAnalysis(name=MODEL, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load image
img = cv2.imread(INPUT_IMG)
if img is None:
    raise FileNotFoundError(f"Could not load image at {INPUT_IMG}")

h, w, _ = img.shape

# Detect faces
faces = app.get(img)
print(f"\nDetected {len(faces)} faces")

# Process each face
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
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw landmarks
    if face.landmark is not None:
        for (lx, ly) in face.landmark.astype(int):
            cv2.circle(img, (lx, ly), 2, (0, 0, 255), -1)

    # Label face index + confidence
    label = f"Face {i+1}: {score:.2f}"
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save cropped face
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size > 0:
        out_face_path = f"{OUTPUT_DIR}/face_{i+1}.jpg"
        cv2.imwrite(out_face_path, face_crop)

# Save annotated image
cv2.imwrite(OUTPUT_ANNOTATED, img)
print(f"\nSaved annotated image with faces -> {OUTPUT_ANNOTATED}")
