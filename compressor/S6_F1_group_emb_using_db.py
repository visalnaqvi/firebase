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
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
        # or use os.environ.get("POSTGRES_URL") if using env var
    )

class HybridFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        self.qdrant = QdrantClient(host=host, port=port)

    def fetch_groups_from_db(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        print(f"🔃 Fetching groups with status warmed")
        cursor.execute("SELECT id FROM groups where status = 'warmed'")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        items = []
        for row in rows:
            items.append(row["id"])
        print(f"✅ Got {len(items)} groups with warmed status")
        return items
    def fetch_items_from_db(self, group_id):
        conn = get_db_connection()
        print(f"🔃 Fetching face for group {group_id}")
        cursor = conn.cursor()
        cursor.execute("SELECT id, person_id FROM faces2 WHERE group_id = %s", (group_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        point_ids = [row["id"] for row in rows]
        qdrant_points = self.qdrant.retrieve(
            collection_name=group_id,
            ids=point_ids,
            with_payload=True,
            with_vectors=True
        )

        qdrant_map = {str(p.id): p for p in qdrant_points}

        items = []
        for row in rows:
            point_id = str(row["id"])
            q_point = qdrant_map.get(point_id)

            if not q_point:
                print(f"⚠️ Qdrant point not found for ID: {point_id}")
                continue

            items.append({
                "id": row["id"],
                "person_id": row["person_id"],
                "face": q_point.vectors.get("face") if q_point.vectors else None,
                "cloth": q_point.vectors.get("cloth") if q_point.vectors else None,
                "assigned": row["person_id"] != None
            })

        print(f"✅ Got {len(items)} face(s) for group {group_id}")
        return items

    def find_similar_candidates(self, item,group_id ,  face_threshold=0.7 ):
        candidates = self.qdrant.query_points(
            collection_name=group_id,
            query=item['face'].tolist(),
            using="face",
            score_threshold=face_threshold
        )
        return [point.id for point in candidates.points if point.id != item['id']]

    def group_and_assign_person_ids(self, items, g_id , face_th=0.7, cloth_th=0.85 ):
        print("🎯 Starting grouping and assigning person IDs")
        groups = []
        updates = []
        qdrant_updates = []

        conn = get_db_connection()
        cursor = conn.cursor()

        for item in items:
            if item['assigned']:
                continue

            group = [item]
            item['assigned'] = True
            queue = deque([item])
            person_id = str(uuid.uuid4())

            while queue:
                current = queue.popleft()
                candidate_ids = self.find_similar_candidates(current,g_id , face_threshold=0.5 )

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

            for member in group:
                member['person_id'] = person_id
                updates.append((person_id, member['id']))
                qdrant_updates.append({
                    "id": member['id'],
                    "payload": {"person_id": person_id}
                })

            groups.append(group)
            print(f"👥 Assigned person_id {person_id} to {len(group)} faces")

        # ✅ Commit updates
        cursor.executemany("UPDATE faces2 SET person_id = %s WHERE id = %s", updates)
        conn.commit()
        cursor.close()
        conn.close()

       

        return groups


  

# 🔧 Usage Example
if __name__ == "__main__":
    grouper = HybridFaceGrouping()
    groups = grouper.fetch_groups_from_db()
    for id in groups:
        items = grouper.fetch_items_from_db(id)
        grouper.group_and_assign_person_ids(items,id, face_th=0.7, cloth_th=0.85 )
