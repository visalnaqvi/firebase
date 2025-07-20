import uuid
import os
from deepface import DeepFace
from scipy.spatial.distance import cosine
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import time
# Constants
COLLECTION_NAME = "faces"
THRESHOLD = 0.7
QDRANT_HOST = "http://localhost:6333"  # or your server IP

client = QdrantClient(QDRANT_HOST)

def extract_embedding(image_path, detector_backend="retinaface"):
    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend=detector_backend,
            enforce_detection=True
        )
        return [obj["embedding"] for obj in embedding_obj]
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return []

def find_similar(embedding):
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=1,
        score_threshold=1 - THRESHOLD  # because cosine similarity in qdrant is distance
    )
    return search_result[0] if search_result else None

def insert_face(image_path, embedding, person_id):
    face_id = str(uuid.uuid4())
    point = PointStruct(
        id=face_id,
        vector=embedding,
        payload={
            "img_path": image_path,
            "person_id": person_id,
            "face_id": face_id
        }
    )
    client.upsert(COLLECTION_NAME, [point])
    return face_id

def process_images(image_paths):
    for image_path in image_paths:
        embeddings = extract_embedding(image_path)
        for embedding in embeddings:
            match = find_similar(embedding)
            if match:
                person_id = match.payload['person_id']
                print(f"🟢 Found similar face, assigning person_id: {person_id} in image {image_path}")
            else:
                person_id = str(uuid.uuid4())
                print(f"🆕 New face, assigning new person_id: {person_id} in image {image_path}")

            insert_face(image_path, embedding, person_id)

def get_max_person_id():
    scroll_result = client.scroll(collection_name=COLLECTION_NAME, limit=10000)
    person_ids = [
        int(p.payload["person_id"]) for p in scroll_result[0] if "person_id" in p.payload
    ]
    return max(person_ids, default=0)
def process_images_accurate(image_paths):
    all_embeddings = []
    for image_path in image_paths:
        embeddings = extract_embedding(image_path)
        for embedding in embeddings:
            all_embeddings.append((image_path, embedding))

    seen = [False] * len(all_embeddings)
    groups = []
    for i, (img_i, emb_i) in enumerate(all_embeddings):
        if seen[i]:
            continue
        group = [img_i]
        seen[i] = True
        for j in range(i + 1, len(all_embeddings)):
            if not seen[j]:
                sim = 1 - cosine(emb_i, all_embeddings[j][1])
                if sim > 0.7:  # stricter threshold
                    group.append(all_embeddings[j][0])
                    seen[j] = True
        groups.append(group)

    # Now insert to Qdrant
    for idx, group in enumerate(groups):
        person_id = str(uuid.uuid4())
        for img_path in group:
            embedding = next(emb for (img, emb) in all_embeddings if img == img_path)
            insert_face(img_path, embedding, person_id)

    print(f"✅ Grouped and inserted {len(groups)} unique persons")

# === Run the logic ===
if __name__ == "__main__":
    start_time = time.time()
    IMAGE_FOLDER = "./images" 
    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    image_paths = [
        os.path.join(IMAGE_FOLDER, file)
        for file in os.listdir(IMAGE_FOLDER)
        if file.lower().endswith(SUPPORTED_EXTS)
    ]
    # image_paths = ["cp_1.jpg", "cp_2.jpg", "cp_3.jpg" , "cp_4.jpg" , "cp_5.jpg" , "cp_6.jpg" ,  "cp_7.jpg" , "cp_8.jpg" , "cp_9.jpg" , "cp_10.jpg"]
    process_images_accurate(image_paths)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"✅ Total time taken: {elapsed_time:.2f} seconds")
