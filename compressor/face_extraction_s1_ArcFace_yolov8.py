from deepface import DeepFace
from scipy.spatial.distance import cosine
import shutil
import os
import time
import cv2
import numpy as np

def extract_embedding(image_path, detector_backend="yolov8"):
    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend=detector_backend,
            enforce_detection=True
        )
        results = []
        for obj in embedding_obj:
            results.append({
                "embedding": obj["embedding"],
                "image_path": image_path,
                "seen": False
            })
        return results
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return []

# === Main comparison logic ===
if __name__ == "__main__":
    start_time = time.time()
    
    IMAGE_FOLDER = "./cropped_faces"
    OUTPUT_FOLDER = "./grouped_faces_ArcFace_yolov8"
    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    image_paths = [
        os.path.join(IMAGE_FOLDER, file)
        for file in os.listdir(IMAGE_FOLDER)
        if file.lower().endswith(SUPPORTED_EXTS)
    ]
    
    all_faces = []

    # Step 1: Extract embeddings from all images
    for path in image_paths:
        all_faces.extend(extract_embedding(path))

    threshold = 0.6
    person_id = 1
    groups = []

    # Step 2: Group similar faces
    for i in range(len(all_faces)):
        if all_faces[i]["seen"]:
            continue
        group = set()
        group.add(all_faces[i]["image_path"])
        all_faces[i]["seen"] = True

        for j in range(i + 1, len(all_faces)):
            if not all_faces[j]["seen"]:
                sim = 1 - cosine(all_faces[i]["embedding"], all_faces[j]["embedding"])
                if sim > threshold:
                    group.add(all_faces[j]["image_path"])
                    all_faces[j]["seen"] = True

        groups.append(group)

    # Step 3: Copy images and cropped thumbnails
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for group in groups:
        folder_name = os.path.join(OUTPUT_FOLDER, f"person_img_{person_id}")
        os.makedirs(folder_name, exist_ok=True)
        for img in group:
            img_name = os.path.basename(img)

            # Copy original image
            dest_path = os.path.join(folder_name, img_name)
            shutil.copy(img, dest_path)
        person_id += 1

    print(f"✅ Grouped faces into {len(groups)} people")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ Total time taken: {elapsed_time:.2f} seconds")
