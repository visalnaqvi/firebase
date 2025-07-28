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
import psycopg2
from psycopg2.extras import DictCursor
import uuid
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "admin",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)

class HybridFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        self.qdrant = QdrantClient(host=host, port=port)
        self.collection_name = "hybrid_people_collection"

    # ✅ Fetch items from DB
    def fetch_items_from_db(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, image_path, filename, face_id, person_id, face_emb, clothing_emb, assigned FROM ready_to_group")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        items = []
        for row in rows:
            items.append({
                "id": row["id"],  # Qdrant point ID
                "image_path": row["image_path"],
                "filename": row["filename"],
                "face_id": row["face_id"],
                "person_id": row["person_id"],
                "face": np.array(row["face_emb"], dtype=np.float32),
                "cloth": torch.tensor(row["clothing_emb"], dtype=torch.float32),
                "assigned": row["assigned"]
            })
        return items

    def find_similar_candidates(self, item, face_threshold=0.7, limit=50):
        candidates = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=item['face'].tolist(),
            using="face",
            limit=limit,
            score_threshold=face_threshold
        )
        return [point.id for point in candidates.points if point.id != item['id']]

    def group_and_assign_person_ids(self, items, face_th=0.7, cloth_th=0.85):
        print("🎯 Starting grouping and assigning person IDs")
        groups = []
        updates = []  # For DB updates
        qdrant_updates = []  # For Qdrant payload updates

        for item in items:
            if item['assigned']:
                continue

            group = [item]
            item['assigned'] = True
            queue = deque([item])
            person_id = str(uuid.uuid4())  # ✅ Generate UUID for person_id

            while queue:
                current = queue.popleft()
                candidate_ids = self.find_similar_candidates(current, face_threshold=0.5)

                for candidate_id in candidate_ids:
                    other = next((x for x in items if x['id'] == candidate_id), None)
                    if other is None or other['assigned']:
                        continue

                    face_sim = 1 - cosine(current['face'], other['face'])
                    cloth_sim = float((current['cloth'] @ other['cloth']).cpu())

                    if face_sim >= face_th or (face_sim >= 0.4 and cloth_sim >= cloth_th):
                        other['assigned'] = True
                        queue.append(other)
                        group.append(other)

            # ✅ Assign UUID as person_id
            for member in group:
                member['person_id'] = person_id
                updates.append((person_id, member['id']))
                qdrant_updates.append({
                    "id": member['id'],
                    "payload": {"person_id": person_id}
                })

            groups.append(group)
            print(f"👥 Assigned person_id {person_id} to {len(group)} faces")

        # ✅ Bulk update DB
        self.update_person_ids_in_db(updates)

        # ✅ Bulk update Qdrant
        self.update_person_ids_in_qdrant(qdrant_updates)

        return groups

    def update_person_ids_in_db(self, updates):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.executemany("UPDATE ready_to_group SET person_id = %s WHERE id = %s", updates)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Updated {len(updates)} records in DB with person_id")

    def update_person_ids_in_qdrant(self, updates):
        for update in updates:
            self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload=update['payload'],
                points=[update['id']]
            )
        print(f"✅ Updated {len(updates)} points in Qdrant with person_id")

# 🔧 Usage Example
if __name__ == "__main__":
    grouper = HybridFaceGrouping()
    items = grouper.fetch_items_from_db()
    groups = grouper.group_and_assign_person_ids(items, face_th=0.7, cloth_th=0.85)
