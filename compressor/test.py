import os
import time
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from insightface.app import FaceAnalysis
from PIL import Image
from transformers import AutoProcessor, AutoModel

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
        self.collection_name = "gallary"
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

    def get_point_by_id(self, point_id):
        """Fetch a single point from Qdrant and return its vectors and metadata."""
        result = self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True
        )
        if not result:
            print(f"❌ No point found for ID: {point_id}")
            return None
        
        fp = result[0]
        return {
            "id": fp.id,
            "face": torch.tensor(fp.vector.get("face")),
            "cloth": torch.tensor(fp.vector.get("cloth")),
            "person_id": fp.payload.get("person_id"),
            "image_path": fp.payload.get("image_path")
        }

    def find_similar_face_candidates(self, face_vector, exclude_id=None, face_threshold=0.4, limit=5):
        """Return similar face candidates excluding the original ID."""
        candidates = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=face_vector.tolist(),
            using="face",
            limit=limit,
            score_threshold=face_threshold
        )

        results = []
        ids = [p.id for p in candidates.points if p.id != exclude_id]  # exclude original
        if ids:
            full_points = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_vectors=True
            )
            for fp in full_points:
                results.append({
                    "id": fp.id,
                    "score": next(p.score for p in candidates.points if p.id == fp.id),
                    "person_id": fp.payload.get("person_id"),
                    "cloth": torch.tensor(fp.vector.get("cloth")),
                    "image_path": fp.payload.get("image_path")
                })

        return results

    def process_point(self, point_id, face_th=0.7, cloth_th=0.85):
        """Check if a given point_id matches existing people or should be new."""
        print(f"🔍 Checking point: {point_id}")
        point = self.get_point_by_id(point_id)
        if not point:
            return

        candidates = self.find_similar_face_candidates(point['face'], exclude_id=point_id)

        if not candidates:
            print("🆕 No similar candidates → New person")
            return

        top_candidate = candidates[0]
        score = top_candidate['score']

        if score >= face_th:
            print(f"✅ High similarity (face): {score:.3f} → Same person")
        elif 0.4 <= score < face_th:
            print(f"⚠️ Medium similarity: {score:.3f} → Checking clothes...")
            matched = False
            for i in range(min(3, len(candidates))):
                cloth_sim = float((point['cloth'] @ candidates[i]['cloth']).cpu())
                print(f"   Candidate {i}: cloth similarity = {cloth_sim:.3f}")
                if cloth_sim >= cloth_th:
                    print(f"✅ Cloth match → Same person")
                    matched = True
            if not matched:
                print("❌ Cloth mismatch → New person")
        else:
            print(f"❌ Low similarity: {score:.3f} → New person")

        print("\nTop candidates:")
        for c in candidates:
            print(f"  ID={c['id']} | face={c['score']:.3f} | person_id={c['person_id']}")

# ✅ Usage Example
if __name__ == "__main__":
    grouper = HybridFaceGrouping()
    grouper.process_point(
        point_id="a4115686-045c-4bcc-bf59-05c7ad4d65ce",  # Replace with actual point ID in Qdrant
        face_th=0.7,
        cloth_th=0.85
    )
