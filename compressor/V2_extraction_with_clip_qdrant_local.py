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
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fashion_model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True).to(self.device)
        self.fashion_processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)

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

    

    def find_similar_candidates(self, item, face_threshold=0.7, limit=50):
        candidates = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=item['face'].tolist(),
            using="face",
            limit=limit,
            score_threshold=face_threshold
        )
        return [point.id for point in candidates.points if point.id != item['id']]

    def group_images_hybrid(self, items, face_th=0.7, cloth_th=0.85, verbose=True):
        print(f"🎯 Starting hybrid grouping with face_th={face_th}, cloth_th={cloth_th}")
        groups = []

        for idx, item in enumerate(items):
            if item['assigned']:
                continue

            if verbose:
                print(f"\n🔄 Processing {item['filename']} (Face #{item['face_index']}) ({idx+1}/{len(items)})")

            group = [f"{item['filename']}#face{item['face_index']}"]
            item['assigned'] = True
            queue = deque([item])

            while queue:
                current = queue.popleft()
                candidate_ids = self.find_similar_candidates(current, face_threshold=0.5)

                for candidate_id in candidate_ids:
                    other = next((x for x in items if x['id'] == candidate_id), None)
                    if other is None or other['assigned']:
                        continue

                    face_sim = 1 - cosine(current['face'], other['face'])
                    cloth_sim = float((current['cloth'] @ other['cloth']).cpu())

                    if verbose:
                        print(f"  📊 Comparing with {other['filename']}#face{other['face_index']}")
                        print(f"     face: {face_sim:.3f}, cloth: {cloth_sim:.3f}")

                    if face_sim >= face_th or (face_sim >= 0.4 and cloth_sim >= cloth_th):
                        other['assigned'] = True
                        group.append(f"{other['filename']}#face{other['face_index']}")
                        queue.append(other)
                        if verbose:
                            condition = "high face sim" if face_sim >= face_th else "face+cloth match"
                            print(f"     ✅ MATCHED ({condition})")

            groups.append(group)
            print(f"👥 Group {len(groups)}: {len(group)} faces")

        return groups

    def organize_groups(self, groups, output_folder):
        print("📁 Organizing grouped images...")
        os.makedirs(output_folder, exist_ok=True)

        for idx, group in enumerate(groups, 1):
            group_dir = os.path.join(output_folder, f'person_{idx:03d}')
            os.makedirs(group_dir, exist_ok=True)

            for tag in group:
                filename, face_idx_str = tag.split("#face")
                src_path = os.path.join("cropped_people", filename)
                new_filename = f"{os.path.splitext(filename)[0]}_face{face_idx_str}.jpg"
                shutil.copy2(src_path, os.path.join(group_dir, new_filename))

            print(f"📂 person_{idx:03d}: {len(group)} faces")

        print(f"✅ Created {len(groups)} person groups")

    def process_folder(self, input_folder, output_folder="hybrid_grouped_faces", face_th=0.7, cloth_th=0.85, verbose=True):
        print("🚀 Starting hybrid face grouping pipeline...")
        print(f"📂 Input: {input_folder}")
        print(f"📂 Output: {output_folder}")
        print(f"🎯 Thresholds: face={face_th}, cloth={cloth_th}")

        start_time = time.time()
        items = [] #read items from database table ready_to_group where id in table is point_id in vector db
        groups = self.group_images_hybrid(items, face_th, cloth_th, verbose)
        self.organize_groups(groups, output_folder)

        total_time = time.time() - start_time
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Final result: {len(groups)} groups from {len(items)} faces")

        return groups

    def analyze_groups(self, groups):
        total_faces = sum(len(group) for group in groups)
        single_face_groups = sum(1 for group in groups if len(group) == 1)

        print(f"\n📈 Grouping Analysis:")
        print(f"   Total groups: {len(groups)}")
        print(f"   Total faces: {total_faces}")
        print(f"   Single-face groups: {single_face_groups}")
        print(f"   Multi-face groups: {len(groups) - single_face_groups}")

        group_sizes = [len(group) for group in groups]
        print(f"   Largest group: {max(group_sizes)} faces")
        print(f"   Average group size: {sum(group_sizes) / len(group_sizes):.1f}")

        size_distribution = {}
        for size in group_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1

        print(f"   Size distribution: {dict(sorted(size_distribution.items()))}")


# 🔧 Usage Example
if __name__ == "__main__":
    grouper = HybridFaceGrouping()

    groups = grouper.process_folder(
        input_folder="cropped_people",
        output_folder="hybrid_grouped_faces",
        face_th=0.7,
        cloth_th=0.85,
        verbose=True
    )

    grouper.analyze_groups(groups)