from deepface import DeepFace
from scipy.spatial.distance import cosine
import shutil
import os
import time
import cv2
import numpy as np

def crop_face(img_path):
    try:
        results = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface", enforce_detection=True)
        if not results:
            print(f"No face found in {img_path}")
            return []

        all_faces = []
        for face_obj in results:
            face = face_obj["face"]

            # Convert from float32/float64 range [0,1] to uint8 [0,255]
            if face.max() <= 1.0:
                face = (face * 255).astype("uint8")
            else:
                face = np.clip(face, 0, 255).astype("uint8")

            all_faces.append(face)
        
        return all_faces
    except Exception as e:
        print(f"❌ Error cropping face in {img_path}: {e}")
        return []

# === Main logic ===
IMAGE_FOLDER = "./img"
OUTPUT_FOLDER = "./croped_faces"
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

image_paths = [
    os.path.join(IMAGE_FOLDER, file)
    for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(SUPPORTED_EXTS)
]

# Make sure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

face_count = 0
for img_idx, img in enumerate(image_paths):
    faces = crop_face(img)
    for face_idx, face in enumerate(faces):
        thumb_path = os.path.join(OUTPUT_FOLDER, f"thumb_{img_idx}_{face_idx}.jpg")
        cv2.imwrite(thumb_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        face_count += 1

if face_count > 0:
    print(f"✅ Saved {face_count} face(s) to {OUTPUT_FOLDER}")
else:
    print("❌ No faces detected.")
