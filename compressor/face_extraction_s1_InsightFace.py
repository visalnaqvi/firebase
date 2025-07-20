import os
import shutil
import time
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# Initialize InsightFace model
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'cuda' if available
face_app.prepare(ctx_id=0)  # Use 0 for CPU or CUDA device index for GPU

def extract_embedding(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Couldn't read image: {image_path}")
            return []

        faces = face_app.get(img)
        results = []

        for face in faces:
            embedding = face.normed_embedding  # Already L2 normalized
            results.append({
                "embedding": embedding,
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
    
    IMAGE_FOLDER = "./img"
    OUTPUT_FOLDER = "./grouped_faces_insightface"
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

    threshold = 0.6  # Adjust based on your tests; InsightFace uses cosine similarity
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

    # Step 3: Copy grouped images
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for group in groups:
        folder_name = os.path.join(OUTPUT_FOLDER, f"person_img_{person_id}")
        os.makedirs(folder_name, exist_ok=True)
        for img in group:
            img_name = os.path.basename(img)
            dest_path = os.path.join(folder_name, img_name)
            shutil.copy(img, dest_path)
        person_id += 1

    print(f"✅ Grouped faces into {len(groups)} people")
    print(f"✅ Total time taken: {time.time() - start_time:.2f} seconds")
