from deepface import DeepFace
from scipy.spatial.distance import cosine
import shutil
import os

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

# === Main comparison logic ===
if __name__ == "__main__":
    image_paths = ["cp_2.jpg", "cp_4.jpg", "cp_8.jpg" , "cp_1.jpg" , "cp_3.jpg" , "cp_6.jpg"]
    all_faces = []

    # Step 1: Extract embeddings from all images
    for path in image_paths:
        all_faces.extend(extract_embedding(path))

    threshold = 0.7
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

    # Step 3: Copy grouped images into folders
    for group in groups:
        folder_name = f"person_{person_id}"
        os.makedirs(folder_name, exist_ok=True)
        for img in group:
            dest_path = os.path.join(folder_name, os.path.basename(img))
            shutil.copy(img, dest_path)
        person_id += 1

    print(f"✅ Grouped faces into {len(groups)} people")
