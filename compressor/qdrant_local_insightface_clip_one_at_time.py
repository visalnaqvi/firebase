import os
import shutil
import time
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from collections import deque
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct
)
import uuid
class HybridFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        # Initialize models
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fashion_model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True).to(self.device)
        self.fashion_processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)
        
        # Initialize Qdrant for efficient similarity search
        self.qdrant = QdrantClient(host=host, port=port)
        self.collection_name = "gallary"
        self._setup_collection()
        
    def _setup_collection(self):
        """Setup collection only if it doesn't exist (persistent)"""
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
        """Extract face embedding from image"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        faces = self.face_app.get(img)
        return faces[0].normed_embedding if faces else None
    
    def extract_clothing_embedding(self, image_path):
        """Extract clothing embedding from image"""
        img = Image.open(image_path).convert('RGB')
        inputs = self.fashion_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.fashion_model.get_image_features(**inputs)
            return emb[0] / emb[0].norm()
    
    def extract_embedding(self , image_path):
        face_emb = self.extract_face_embedding(image_path)
        if face_emb is None:
            print(f"⚠️  No face found in {image_path}")
            return
            
        clothing_emb = self.extract_clothing_embedding(image_path)
        return  {
                'path': image_path,
                'face': face_emb,
                'cloth': clothing_emb,
            }
    
    
    def find_similar_face_candidates(self, item, face_threshold=0.4, limit=5):
        """Return a list of up to `limit` matching face candidates (excluding itself)"""
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
                results.append({
                    "id": fp.id,
                    "score": next(p.score for p in candidates.points if p.id == fp.id),
                    "person_id": fp.payload.get("person_id"),
                    "cloth": torch.tensor(fp.vector.get("cloth"))
                })

        return results
    def insert_new_person_in_qdrant(self , path , emb):
        self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector={
                                    "face": emb["face"].tolist(),
                                    "cloth": emb["cloth"].cpu().tolist()
                                },
                                payload={
                                    "image_path": path,
                                    "person_id": str(uuid.uuid4())
                                }
                            )
                        ]
                    )
            
    
    def process_folder(self, input_folder, output_folder="hybrid_grouped", 
                      face_th=0.7, cloth_th=0.85, verbose=True):
        """Complete pipeline combining vector DB efficiency with your superior logic"""
        print("🚀 Starting hybrid face grouping pipeline...")
        print(f"📂 Input: {input_folder}")
        print(f"📂 Output: {output_folder}")
        print(f"🎯 Thresholds: face={face_th}, cloth={cloth_th}")
        
        start_time = time.time()
        
        image_paths = []
        
        for f in os.listdir(input_folder):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            p = os.path.join(input_folder, f)   

            image_paths.append(p)

        for path in image_paths:
            embeddings = self.extract_embedding(path)
            if not embeddings:
                continue
            candidates = self.find_similar_face_candidates(embeddings)
            if candidates:
                score = candidates[0]['score']  # Extract score
                if 0.7 <= score:
                    print(f"Candidate -> High similarity (0.7 - 1): {score}")
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
                                    "person_id": candidates[0]["person_id"]
                                }
                            )
                        ]
                    )
                elif 0.4 <= score < 0.7:
                    print(f"Candidate -> Medium similarity (0.4 - 0.7): {score}")
                    flag = False
                    for i in range(min(3, len(candidates))):
                        print(f"top 3 does not score face in 0.4 so creating new person for {path}")
                        if candidates[i]["score"] < 0.4:
                            self.insert_new_person_in_qdrant(path=path , emb=embeddings)
                            flag = True
                            break
                    if flag:
                        continue
                    for i in range(min(3, len(candidates))):
                        print("checking cloth similarity")
                        cloth_sim = float((embeddings['cloth'] @ candidates[i]['cloth']).cpu())
                        if cloth_sim < cloth_th:
                            print(f"got cloth similary below threshold for 3 candidates creating new person for {path}")
                            self.insert_new_person_in_qdrant(path=path , emb=embeddings)
                            flag = True
                            break
                    if flag:
                        continue
                    print(f"all checks passed cloth sim match for {path} on top three assigning person")
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
                                        "person_id": candidates[0]["person_id"]
                                    }
                                )
                            ]
                        )
                    
                else:
                    self.insert_new_person_in_qdrant(path=path , emb=embeddings)
                    print(f"Candidate -> Low similarity (< 0.4): {score}")
            else:
                self.insert_new_person_in_qdrant(path=path , emb=embeddings)

        
        total_time = time.time() - start_time
        print(f"⏱️  Total time: {total_time:.2f}s")
    
        
        return None


# 🔧 Usage Examples
if __name__ == "__main__":
    grouper = HybridFaceGrouping()
    
    # Your preferred settings
    groups = grouper.process_folder(
        input_folder="cropped_people",
        output_folder="hp_2",
        face_th=0.4,     # Your face threshold
        cloth_th=0.85,   # Your cloth threshold  
        verbose=True     # Show detailed matching process
    )
    
    # Analyze the results
    # grouper.analyze_groups(groups)