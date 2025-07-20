from deepface import DeepFace
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from scipy.spatial.distance import cosine
import shutil
import os
import uuid

client = QdrantClient("http://localhost:6333")

COLLECTION_NAME = "face_embeddings"

# Initialize or recreate collection
def initialize_qdrant():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE)  # Facenet = 128
    )

def extract_embedding(image_path, detector_backend="retinaface"):
    try:
        print(f"Processing image: {image_path}")
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend=detector_backend,
            enforce_detection=False
        )

        if not isinstance(embedding_obj, list):
            print(f"⚠️ Unexpected embedding format for {image_path}")
            return []

        results = []
        for obj in embedding_obj:
            if isinstance(obj, dict) and "embedding" in obj:
                results.append({
                    "embedding": obj["embedding"],
                    "image_path": image_path,
                    "seen": False
                })
        return results
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return []

# Add vector to Qdrant with payload
def add_to_qdrant(face_obj):
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=face_obj["embedding"],
        payload={"image_path": face_obj["image_path"]}
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])

# Search for similar vectors
def find_similar(embedding, threshold=0.7, top_k=10):
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
        score_threshold=threshold
    )
    return hits

# === Main logic ===
if __name__ == "__main__":
    image_paths = ["cp_1.jpg", "cp_2.jpg", "cp_3.jpg" , "cp_4.jpg" , "cp_5.jpg" , "cp_6.jpg" ,  "cp_7.jpg" , "cp_8.jpg" , "cp_9.jpg" , "cp_10.jpg"]
    # image_paths = ["cp_1.jpg" , "cp_3.jpg" ]
    all_faces = []

    initialize_qdrant()  # Setup DB

    # Step 1: Extract embeddings and insert into Qdrant
    for path in image_paths:
        embeddings = extract_embedding(path)
        all_faces.extend(embeddings)
        for emb in embeddings:
            add_to_qdrant(emb)

    threshold = 0.7
    person_id = 1
    grouped_ids = set()
    groups = []

    # Step 2: Use Qdrant for grouping
    for face in all_faces:
        if face["image_path"] in grouped_ids:
            continue

        group = set()
        group.add(face["image_path"])
        grouped_ids.add(face["image_path"])

        hits = find_similar(face["embedding"], threshold=threshold)

        for hit in hits:
            path = hit.payload["image_path"]
            if path not in grouped_ids:
                group.add(path)
                grouped_ids.add(path)

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
