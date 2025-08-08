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
from qdrant_client.models import VectorParams, Distance, PointStruct
import psycopg2
from psycopg2.extras import DictCursor
from psycopg2 import pool
import uuid
import logging
from contextlib import contextmanager
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Optional
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabasePool:
    def __init__(self, min_conn=1, max_conn=20):
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            min_conn, max_conn,
            host="ballast.proxy.rlwy.net",
            port="56193",
            dbname="railway",
            user="postgres",
            password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
        )

    @contextmanager
    def get_connection(self):
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    def close_all(self):
        self.connection_pool.closeall()

class ProductionFaceGrouping:
    def __init__(self, host="localhost", port=6333, batch_size=100):
        self.qdrant = QdrantClient(host=host, port=port)
        self.db_pool = DatabasePool()
        self.batch_size = batch_size
        
    def fetch_groups_from_db(self) -> List[str]:
        with self.db_pool.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                logger.info("🔃 Fetching groups with status warmed")
                cursor.execute("SELECT id FROM groups WHERE status = 'warmed'")
                rows = cursor.fetchall()
                items = [row["id"] for row in rows]
                logger.info(f"✅ Got {len(items)} groups with warmed status")
                return items

    def fetch_items_batch(self, group_id: str, offset: int = 0, limit: int = None) -> List[Dict]:
        """Fetch items in batches to manage memory"""
        if limit is None:
            limit = self.batch_size
            
        with self.db_pool.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                logger.info(f"🔃 Fetching faces for group {group_id} (offset: {offset}, limit: {limit})")
                cursor.execute(
                    "SELECT id, person_id FROM faces WHERE group_id = %s ORDER BY id LIMIT %s OFFSET %s", 
                    (group_id, limit, offset)
                )
                rows = cursor.fetchall()

        if not rows:
            return []

        point_ids = [row["id"] for row in rows]
        
        try:
            qdrant_points = self.qdrant.retrieve(
                collection_name=group_id,
                ids=point_ids,
                with_payload=True,
                with_vectors=True
            )
        except Exception as e:
            logger.error(f"Failed to retrieve from Qdrant: {e}")
            return []

        qdrant_map = {str(p.id): p for p in qdrant_points}

        items = []
        for row in rows:
            point_id = str(row["id"])
            q_point = qdrant_map.get(point_id)

            if not q_point or not q_point.vectors:
                logger.warning(f"⚠️ Qdrant point not found for ID: {point_id}")
                continue

            items.append({
                "id": row["id"],
                "person_id": row["person_id"],
                "face": np.array(q_point.vectors.get("face", [])),
                "cloth": torch.tensor(q_point.vectors.get("cloth", [])) if q_point.vectors.get("cloth") else None,
                "assigned": row["person_id"] is not None
            })

        logger.info(f"✅ Got {len(items)} face(s) for group {group_id}")
        return items

    def get_total_faces_count(self, group_id: str) -> int:
        """Get total number of faces in group"""
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM faces WHERE group_id = %s", (group_id,))
                return cursor.fetchone()[0]

    def clustering_approach(self, items: List[Dict], face_th: float = 0.7) -> List[List[Dict]]:
        """Use DBSCAN clustering for better scalability"""
        if not items:
            return []
            
        logger.info(f"🎯 Using clustering approach for {len(items)} items")
        
        # Extract face embeddings
        face_embeddings = np.array([item['face'] for item in items if len(item['face']) > 0])
        valid_items = [item for item in items if len(item['face']) > 0]
        
        if len(face_embeddings) == 0:
            return []
        
        # Use DBSCAN clustering
        # Convert similarity threshold to distance (eps parameter)
        eps = 1 - face_th  # cosine distance
        clustering = DBSCAN(eps=eps, min_samples=1, metric='cosine')
        
        try:
            cluster_labels = clustering.fit_predict(face_embeddings)
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return []
        
        # Group items by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_items[idx])
        
        groups = list(clusters.values())
        logger.info(f"✅ Created {len(groups)} clusters")
        return groups

    def batch_update_database(self, updates: List[tuple], batch_size: int = 500):
        """Update database in batches to avoid timeouts"""
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                for i in range(0, len(updates), batch_size):
                    batch = updates[i:i + batch_size]
                    cursor.executemany("UPDATE faces SET person_id = %s WHERE id = %s", batch)
                    conn.commit()
                    logger.info(f"Updated {len(batch)} records (batch {i//batch_size + 1})")

    def process_group_in_batches(self, group_id: str, face_th: float = 0.7, cloth_th: float = 0.85):
        """Process a group in batches to manage memory"""
        total_faces = self.get_total_faces_count(group_id)
        logger.info(f"📊 Processing group {group_id} with {total_faces} faces")
        
        if total_faces > 2000:
            # For very large groups, use clustering approach
            return self.process_large_group_clustering(group_id, face_th)
        else:
            # For medium groups, use optimized BFS
            return self.process_medium_group_optimized(group_id, face_th, cloth_th)

    def process_large_group_clustering(self, group_id: str, face_th: float = 0.7):
        """Process large groups using clustering"""
        all_items = []
        offset = 0
        
        # Load all data in batches
        while True:
            batch_items = self.fetch_items_batch(group_id, offset, self.batch_size)
            if not batch_items:
                break
            all_items.extend(batch_items)
            offset += self.batch_size
            
            # Force garbage collection to manage memory
            if offset % (self.batch_size * 5) == 0:
                gc.collect()
        
        logger.info(f"📊 Loaded {len(all_items)} items for clustering")
        
        # Filter unassigned items
        unassigned_items = [item for item in all_items if not item['assigned']]
        
        if not unassigned_items:
            logger.info("No unassigned items found")
            return []
        
        # Use clustering
        groups = self.clustering_approach(unassigned_items, face_th)
        
        # Assign person IDs and prepare updates
        updates = []
        for group in groups:
            person_id = str(uuid.uuid4())
            for member in group:
                member['person_id'] = person_id
                updates.append((person_id, member['id']))
            logger.info(f"👥 Assigned person_id {person_id} to {len(group)} faces")
        
        # Update database in batches
        if updates:
            self.batch_update_database(updates)
        
        return groups

    def process_medium_group_optimized(self, group_id: str, face_th: float = 0.7, cloth_th: float = 0.85):
        """Optimized BFS for medium-sized groups"""
        all_items = []
        offset = 0
        
        # Load all data
        while True:
            batch_items = self.fetch_items_batch(group_id, offset, self.batch_size)
            if not batch_items:
                break
            all_items.extend(batch_items)
            offset += self.batch_size
        
        unassigned_items = [item for item in all_items if not item['assigned']]
        
        if not unassigned_items:
            return []
        
        groups = []
        updates = []
        processed_ids = set()
        
        for item in unassigned_items:
            if item['id'] in processed_ids:
                continue
                
            group = [item]
            processed_ids.add(item['id'])
            queue = deque([item])
            person_id = str(uuid.uuid4())
            
            while queue:
                current = queue.popleft()
                
                # Find similar candidates using Qdrant
                try:
                    candidates = self.qdrant.query_points(
                        collection_name=group_id,
                        query=current['face'].tolist(),
                        using="face",
                        score_threshold=0.3,  # Lower threshold for initial filtering
                        limit=50  # Limit candidates to manage performance
                    )
                    candidate_ids = [point.id for point in candidates.points 
                                   if point.id != current['id'] and point.id not in processed_ids]
                except Exception as e:
                    logger.warning(f"Qdrant query failed: {e}")
                    continue
                
                for candidate_id in candidate_ids:
                    other = next((x for x in unassigned_items if x['id'] == candidate_id), None)
                    if other is None or other['id'] in processed_ids:
                        continue
                    
                    try:
                        face_sim = 1 - cosine(current['face'], other['face'])
                        cloth_sim = float((current['cloth'] @ other['cloth']).cpu()) if current['cloth'] is not None and other['cloth'] is not None else 0
                        
                        if face_sim >= face_th or (face_sim >= 0.4 and cloth_sim >= cloth_th):
                            processed_ids.add(other['id'])
                            queue.append(other)
                            group.append(other)
                    except Exception as e:
                        logger.warning(f"Similarity calculation failed: {e}")
                        continue
            
            for member in group:
                member['person_id'] = person_id
                updates.append((person_id, member['id']))
            
            groups.append(group)
            logger.info(f"👥 Assigned person_id {person_id} to {len(group)} faces")
        
        # Update database in batches
        if updates:
            self.batch_update_database(updates)
        
        return groups

    def cleanup(self):
        """Cleanup resources"""
        self.db_pool.close_all()

# 🔧 Usage Example
if __name__ == "__main__":
    grouper = ProductionFaceGrouping(batch_size=200)
    
    try:
        groups = grouper.fetch_groups_from_db()
        
        for group_id in groups:
            logger.info(f"🔄 Processing group: {group_id}")
            start_time = time.time()
            
            result_groups = grouper.process_group_in_batches(
                group_id, 
                face_th=0.7, 
                cloth_th=0.85
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Completed group {group_id} in {elapsed_time:.2f}s, created {len(result_groups)} person groups")
    
    except Exception as e:
        logger.error(f"❌ Error processing groups: {e}")
    finally:
        grouper.cleanup()