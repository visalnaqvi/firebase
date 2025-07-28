import os
import time
import cv2
import torch
import uuid
from PIL import Image
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class HybridFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        # Initialize models
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fashion_model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True).to(self.device)
        self.fashion_processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)

        # Initialize Qdrant
        self.qdrant = QdrantClient(host=host, port=port)
        self.collection_name = "gallary-v2"
        self._setup_collection()

    def _setup_collection(self):
        """Setup collection only if it doesn't exist"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            print(f"✅ Using existing collection with {collection_info.points_count} points")
        except:
            print("🔧 Creating new collection...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "face": VectorParams(size=512, distance=Distance.COSINE),
                    "cloth": VectorParams(size=512, distance=Distance.COSINE)
                }
            )

    def extract_face_embedding(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        faces = self.face_app.get(img)
        return faces[0].normed_embedding if faces else None

    def extract_clothing_embedding(self, image_path):
        img = Image.open(image_path).convert('RGB')
        inputs = self.fashion_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.fashion_model.get_image_features(**inputs)
            return emb[0] / emb[0].norm()

    def extract_embeddings(self, image_path):
        face_emb = self.extract_face_embedding(image_path)
        if face_emb is None:
            print(f"⚠️ No face found in {image_path}")
            return None
        cloth_emb = self.extract_clothing_embedding(image_path)
        return {"path": image_path, "face": face_emb, "cloth": cloth_emb}

    def find_similar_candidates(self, item, face_threshold=0.4, limit=5):
        candidates = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=item['face'].tolist(),
            using="face",
            limit=limit,
            score_threshold=face_threshold
        )

        results = []
        ids = [point.id for point in candidates.points]
        if ids:
            full_points = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_vectors=True
            )
            for fp in full_points:
                cloth_tensor = torch.tensor(fp.vector.get("cloth"))
                score = next(p.score for p in candidates.points if p.id == fp.id)
                results.append({
                    "id": fp.id,
                    "score": score,
                    "person_id": fp.payload.get("person_id"),
                    "cloth": cloth_tensor
                })

        return results

    def insert_point(self, path, embeddings, person_id=None):
        if person_id is None:
            person_id = str(uuid.uuid4())
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "face": embeddings["face"].tolist(),
                        "cloth": embeddings["cloth"].cpu().tolist()
                    },
                    payload={
                        "image_path": path,
                        "person_id": person_id
                    }
                )
            ]
        )

    def process_folder(self, input_folder, face_th=0.7, cloth_th=0.85, verbose=True):
        print("🚀 Starting hybrid face grouping pipeline...")
        print(f"📂 Input: {input_folder}")
        print(f"🎯 Thresholds: face={face_th}, cloth={cloth_th}")

        start_time = time.time()
        image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for path in image_paths:
            embeddings = self.extract_embeddings(path)
            if not embeddings:
                continue

            candidates = self.find_similar_candidates(embeddings)
            if not candidates:
                if verbose: print(f"🆕 No candidates → New person for {path}")
                self.insert_point(path, embeddings)
                continue

            # Sort candidates by similarity score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_match = candidates[0]
            best_face_score = best_match['score']

            # Decision logic
            assigned = False
            for cand in candidates[:3]:  # Check top 3 candidates
                face_score = cand['score']
                cloth_score = float((embeddings['cloth'] @ cand['cloth']).cpu())
                combined_score = (face_score * 0.7) + (cloth_score * 0.3)

                if verbose:
                    print(f"📊 Comparing {path} → Face: {face_score:.3f}, Cloth: {cloth_score:.3f}, Combined: {combined_score:.3f}")

                if face_score >= 0.75 or (face_score >= 0.5 and cloth_score >= cloth_th):
                    if verbose: print(f"✅ Assigning to person_id {cand['person_id']}")
                    self.insert_point(path, embeddings, cand['person_id'])
                    assigned = True
                    break

            if not assigned:
                if verbose: print(f"🆕 No strong match → Creating new person for {path}")
                self.insert_point(path, embeddings)

        total_time = time.time() - start_time
        print(f"⏱️ Total time: {total_time:.2f}s")

        return None


# ✅ Usage
if __name__ == "__main__":
    grouper = HybridFaceGrouping()
    grouper.process_folder(
        input_folder="cropped_people",
        face_th=0.7,
        cloth_th=0.85,
        verbose=True
    )
