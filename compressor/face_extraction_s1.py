from deepface import DeepFace
from scipy.spatial.distance import cosine
import shutil
import os
import time
import cv2
import numpy as np

def extract_embedding(image_path, detector_backend="retinaface"):
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

def crop_face(img_path):
    try:
        results = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface", enforce_detection=True)
        if not results:
            print(f"No face found in {img_path}")
            return None
        face_obj = results[0]  # Only first face
        face = face_obj["face"]

        # Convert from float32/float64 range [0,1] to uint8 [0,255]
        if face.max() <= 1.0:
            face = (face * 255).astype("uint8")
        else:
            face = np.clip(face, 0, 255).astype("uint8")

        return face
    except Exception as e:
        print(f"❌ Error cropping face in {img_path}: {e}")
        return None

# === Main comparison logic ===
if __name__ == "__main__":
    start_time = time.time()
    
    IMAGE_FOLDER = "./img"
    OUTPUT_FOLDER = "./grouped_faces"
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

            # Save cropped face thumbnail
            cropped_face = crop_face(img)
            if cropped_face is not None:
                thumb_path = os.path.join(folder_name, f"thumb_{img_name}")
                cv2.imwrite(thumb_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
        person_id += 1

    print(f"✅ Grouped faces into {len(groups)} people")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ Total time taken: {elapsed_time:.2f} seconds")
