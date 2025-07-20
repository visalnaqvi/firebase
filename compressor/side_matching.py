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

def is_same_person(reference_embeddings, query_embedding, threshold=0.55):
    for ref in reference_embeddings:
        sim = 1 - cosine(ref["embedding"], query_embedding["embedding"])
        print(f"Similarity: {sim:.4f}")
        if sim > threshold:
            return True
    return False

# === Setup ===
known_folder = "./grouped_faces_insightface/person_img_1"
query_image = "./grouped_faces_insightface/person_img_7/img.jpeg"
query_image1 = "./grouped_faces_insightface/person_img_7/img2.jpeg"
img = "./grouped_faces_insightface/person_img_1/img_sim.jpeg"
# Get image paths
known_images = [os.path.join(known_folder, f) for f in os.listdir(known_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
reference_embeddings = []
# for img in known_images:
reference_embeddings.extend(extract_embedding(img))


query_embeddings = extract_embedding(query_image)
if not query_embeddings:
    print("❌ Failed to get embedding for query image.")
else:
    result = is_same_person(reference_embeddings, query_embeddings[0])
    if result:
        print("✅ The side pose matches the person.")
    else:
        print("❌ The side pose does NOT match the person.")
