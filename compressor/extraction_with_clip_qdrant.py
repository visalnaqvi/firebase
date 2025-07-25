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
        self.collection_name = "hybrid_people_collection"
        self._setup_collection()
        
    def _setup_collection(self):
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)
            
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "face": VectorParams(size=512, distance=Distance.COSINE),
                "clothing": VectorParams(size=512, distance=Distance.COSINE)
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
    
    def index_images(self, folder_path):
        """Index all images and store in both Qdrant and local list"""
        print("🔍 Indexing images...")
        
        items = []
        point_id = 0
        
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(folder_path, filename)
            face_emb = self.extract_face_embedding(image_path)
            
            if face_emb is None:
                print(f"⚠️  No face found in {filename}")
                continue
            
            clothing_emb = self.extract_clothing_embedding(image_path)
            
            # Store in Qdrant for efficient similarity search
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={
                            "face": face_emb.tolist(),
                            "clothing": clothing_emb.cpu().tolist()
                        },
                        payload={
                            "image_path": image_path,
                            "filename": filename
                        }
                    )
                ]
            )
            
            # Also keep in memory for your superior grouping algorithm
            items.append({
                'id': point_id,
                'path': image_path,
                'filename': filename,
                'face': face_emb,
                'cloth': clothing_emb,
                'assigned': False
            })
            
            point_id += 1
            
        print(f"✅ Indexed {len(items)} images with faces")
        return items
    
    def find_similar_candidates(self, item, face_threshold=0.3, limit=20):
        """Use Qdrant to efficiently find potential similar faces"""
        candidates = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=item['face'].tolist(),
            using="face",
            limit=limit,
            score_threshold=face_threshold  # Lower threshold to catch more candidates
        )
        
        return [point.id for point in candidates.points if point.id != item['id']]
    
    def group_images_hybrid(self, items, face_th=0.7, cloth_th=0.85, verbose=True):
        """
        Your superior grouping logic enhanced with vector DB for efficiency
        """
        print(f"🎯 Starting hybrid grouping with face_th={face_th}, cloth_th={cloth_th}")
        groups = []
        
        for idx, item in enumerate(items):
            if item['assigned']:
                continue
            
            if verbose:
                print(f"\n🔄 Processing {item['filename']} ({idx+1}/{len(items)})")
            
            group = [item['path']]
            item['assigned'] = True
            queue = deque([item])
            
            while queue:
                current = queue.popleft()
                
                # Use vector DB to get potential candidates efficiently
                candidate_ids = self.find_similar_candidates(current, face_threshold=0.3)
                
                for candidate_id in candidate_ids:
                    # Find the candidate in our items list
                    other = next((x for x in items if x['id'] == candidate_id), None)
                    if other is None or other['assigned']:
                        continue
                    
                    # Calculate similarities
                    face_sim = 1 - cosine(current['face'], other['face'])
                    cloth_sim = float((current['cloth'] @ other['cloth']).cpu())
                    
                    if verbose:
                        print(f"  📊 {current['filename']} vs {other['filename']}")
                        print(f"     face: {face_sim:.3f}, cloth: {cloth_sim:.3f}")
                    
                    # Your superior matching logic
                    if face_sim >= face_th or (face_sim >= 0.4 and cloth_sim >= cloth_th):
                        other['assigned'] = True
                        group.append(other['path'])
                        queue.append(other)
                        
                        if verbose:
                            condition = "high face sim" if face_sim >= face_th else "face+cloth match"
                            print(f"     ✅ MATCHED ({condition})")
            
            groups.append(group)
            print(f"👥 Group {len(groups)}: {len(group)} images")
        
        return groups
    
    def organize_groups(self, groups, output_folder):
        """Organize groups into folders"""
        print("📁 Organizing images...")
        os.makedirs(output_folder, exist_ok=True)
        
        for idx, group in enumerate(groups, 1):
            group_dir = os.path.join(output_folder, f'person_{idx:03d}')
            os.makedirs(group_dir, exist_ok=True)
            
            for img_path in group:
                shutil.copy2(img_path, os.path.join(group_dir, os.path.basename(img_path)))
            
            print(f"📂 person_{idx:03d}: {len(group)} images")
        
        print(f"✅ Created {len(groups)} person groups")
    
    def process_folder(self, input_folder, output_folder="hybrid_grouped", 
                      face_th=0.7, cloth_th=0.85, verbose=True):
        """Complete pipeline combining vector DB efficiency with your superior logic"""
        print("🚀 Starting hybrid face grouping pipeline...")
        print(f"📂 Input: {input_folder}")
        print(f"📂 Output: {output_folder}")
        print(f"🎯 Thresholds: face={face_th}, cloth={cloth_th}")
        
        start_time = time.time()
        
        # Step 1: Index images (stores in both Qdrant and memory)
        items = self.index_images(input_folder)
        
        # Step 2: Apply your superior grouping algorithm with vector DB acceleration
        groups = self.group_images_hybrid(items, face_th, cloth_th, verbose)
        
        # Step 3: Organize into folders
        self.organize_groups(groups, output_folder)
        
        total_time = time.time() - start_time
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Final result: {len(groups)} groups from {len(items)} images")
        
        return groups

    def analyze_groups(self, groups):
        """Analyze the quality of grouping"""
        total_images = sum(len(group) for group in groups)
        single_image_groups = sum(1 for group in groups if len(group) == 1)
        
        print(f"\n📈 Grouping Analysis:")
        print(f"   Total groups: {len(groups)}")
        print(f"   Total images: {total_images}")
        print(f"   Single-image groups: {single_image_groups}")
        print(f"   Multi-image groups: {len(groups) - single_image_groups}")
        
        group_sizes = [len(group) for group in groups]
        print(f"   Largest group: {max(group_sizes)} images")
        print(f"   Average group size: {sum(group_sizes) / len(group_sizes):.1f}")
        
        size_distribution = {}
        for size in group_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1
        
        print(f"   Size distribution: {dict(sorted(size_distribution.items()))}")


# 🔧 Usage Examples
if __name__ == "__main__":
    grouper = HybridFaceGrouping()
    
    # Your preferred settings
    groups = grouper.process_folder(
        input_folder="cropped_people",
        output_folder="hybrid_grouped_faces",
        face_th=0.7,     # Your face threshold
        cloth_th=0.85,   # Your cloth threshold  
        verbose=True     # Show detailed matching process
    )
    
    # Analyze the results
    grouper.analyze_groups(groups)