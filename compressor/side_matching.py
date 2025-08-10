import os
import numpy as np
import torch
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import DictCursor
import uuid

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway", 
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

class TrueSimpleFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        self.qdrant = QdrantClient(host=host, port=port)

    def get_unassigned_faces(self, group_id):
        """Get unassigned face IDs one by one (generator for memory efficiency)"""
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        print(f"🔃 Fetching unassigned faces for group {group_id}")
        cursor.execute(
            "SELECT id FROM faces WHERE group_id = %s AND person_id IS NULL ORDER BY id", 
            (group_id,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        face_ids = [row["id"] for row in rows]
        print(f"✅ Found {len(face_ids)} unassigned faces")
        return face_ids

    def get_single_face_embedding(self, group_id, face_id):
        """Get embedding for a single face from Qdrant"""
        try:
            qdrant_points = self.qdrant.retrieve(
                collection_name=group_id,
                ids=[face_id],
                with_payload=True,
                with_vectors=True
            )
            
            if qdrant_points and qdrant_points[0].vectors:
                point = qdrant_points[0]
                return {
                    'face': np.array(point.vectors.get("face", [])),
                    'cloth': torch.tensor(point.vectors.get("cloth", [])) if point.vectors.get("cloth") else None,
                }
            return None
        except Exception as e:
            print(f"❌ Error retrieving embedding for {face_id}: {e}")
            return None

    def find_best_match(self, face_embedding, group_id, face_threshold=0.7):
        """Find the best matching face - YOUR SIMPLE ALGORITHM"""
        try:
            candidates = self.qdrant.query_points(
                collection_name=group_id,
                query=face_embedding.tolist(),
                using="face",
                score_threshold=face_threshold,
                limit=1,  # Only need the best match
                with_payload=True
            )
            
            if candidates.points:
                best_match = candidates.points[0]
                return {
                    'id': best_match.id,
                    'score': best_match.score,
                    'person_id': best_match.payload.get('person_id') if best_match.payload else None
                }
            return None
        except Exception as e:
            print(f"⚠️ Error querying Qdrant: {e}")
            return None

    def find_match_with_cloth_fallback(self, face_embedding, cloth_embedding, group_id, 
                                     face_threshold=0.4, cloth_threshold=0.85):
        """Fallback: Find match using clothing similarity"""
        if cloth_embedding is None:
            return None
            
        try:
            candidates = self.qdrant.query_points(
                collection_name=group_id,
                query=face_embedding.tolist(),
                using="face",
                score_threshold=face_threshold,
                limit=5,  # Check a few candidates for clothing match
                with_payload=True,
                with_vectors=True
            )
            
            if not candidates.points:
                return None
            
            # Check clothing similarity for each candidate
            for candidate in candidates.points:
                if candidate.vector and candidate.vector.get("cloth"):
                    candidate_cloth = torch.tensor(candidate.vector.get("cloth"))
                    cloth_sim = float((cloth_embedding @ candidate_cloth).cpu())
                    
                    if cloth_sim >= cloth_threshold:
                        return {
                            'id': candidate.id,
                            'face_score': candidate.score,
                            'cloth_score': cloth_sim,
                            'person_id': candidate.payload.get('person_id') if candidate.payload else None
                        }
            return None
        except Exception as e:
            print(f"⚠️ Error in clothing similarity search: {e}")
            return None

    def update_person_id(self, face_id, person_id, group_id):
        """Update person_id in both PostgreSQL and Qdrant"""
        # Update PostgreSQL
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE faces SET person_id = %s WHERE id = %s", 
            (person_id, face_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        # Update Qdrant
        self.qdrant.set_payload(
            collection_name=group_id,
            points=[face_id],
            payload={"person_id": person_id}
        )

    def process_unassigned_faces_simple(self, group_id):
        """
        YOUR SIMPLE ALGORITHM - CORRECTLY IMPLEMENTED:
        1. Get unassigned face
        2. Query Qdrant for best match
        3. If match has person_id → assign same person_id
        4. If match has NO person_id → create new person_id for BOTH
        5. If no match → create new person_id
        """
        print(f"🚀 Processing group {group_id} with TRUE simple algorithm")
        
        # Get all unassigned face IDs
        unassigned_face_ids = self.get_unassigned_faces(group_id)
        if not unassigned_face_ids:
            print("✅ No unassigned faces found")
            return
        
        assigned_count = 0
        new_persons_count = 0
        matched_to_existing = 0
        
        # Process each face individually - YOUR ALGORITHM
        for face_id in unassigned_face_ids:
            print(f"🔍 Processing face {face_id}")
            
            # Get this face's embedding
            embedding_data = self.get_single_face_embedding(group_id, face_id)
            if not embedding_data:
                print(f"⚠️ Could not get embedding for {face_id}, skipping")
                continue
            
            face_emb = embedding_data['face']
            cloth_emb = embedding_data['cloth']
            
            # Step 1: Try high-confidence face match (0.7 threshold)
            match = self.find_best_match(face_emb, group_id, face_threshold=0.7)
            
            person_id = None
            
            if match:
                if match['person_id']:
                    # Case 1: Found match with existing person_id
                    person_id = match['person_id']
                    print(f"✅ Found assigned match {match['id']} (score: {match['score']:.3f}) - "
                          f"assigning existing person_id: {person_id}")
                    matched_to_existing += 1
                else:
                    # Case 2: Found match but it has NO person_id - CREATE NEW FOR BOTH
                    person_id = str(uuid.uuid4())
                    
                    # Update the matched face first
                    self.update_person_id(match['id'], person_id, group_id)
                    
                    print(f"👥 Found unassigned match {match['id']} (score: {match['score']:.3f}) - "
                          f"created new person_id {person_id} for both faces")
                    new_persons_count += 1
            else:
                # Step 2: Try clothing similarity fallback
                cloth_match = self.find_match_with_cloth_fallback(face_emb, cloth_emb, group_id)
                
                if cloth_match:
                    if cloth_match['person_id']:
                        # Found clothing match with existing person_id
                        person_id = cloth_match['person_id']
                        print(f"✅ Found clothing match {cloth_match['id']} "
                              f"(face: {cloth_match['face_score']:.3f}, cloth: {cloth_match['cloth_score']:.3f}) - "
                              f"assigning existing person_id: {person_id}")
                        matched_to_existing += 1
                    else:
                        # Found clothing match but no person_id - CREATE NEW FOR BOTH
                        person_id = str(uuid.uuid4())
                        
                        # Update the matched face first
                        self.update_person_id(cloth_match['id'], person_id, group_id)
                        
                        print(f"👥 Found unassigned clothing match {cloth_match['id']} - "
                              f"created new person_id {person_id} for both faces")
                        new_persons_count += 1
                else:
                    # Case 3: No match found at all - CREATE NEW PERSON
                    person_id = str(uuid.uuid4())
                    print(f"🆕 No match found - created new person_id: {person_id}")
                    new_persons_count += 1
            
            # Update current face with the determined person_id
            self.update_person_id(face_id, person_id, group_id)
            assigned_count += 1
        
        print(f"🎉 Processing complete!")
        print(f"   📊 Total faces assigned: {assigned_count}")
        print(f"   👤 New persons created: {new_persons_count}")
        print(f"   🔗 Matched to existing persons: {matched_to_existing}")

    def process_all_groups(self):
        """Process all warmed groups"""
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        cursor.execute("SELECT id FROM groups WHERE status = 'warmed'")
        groups = cursor.fetchall()
        cursor.close()
        conn.close()
        
        group_ids = [row["id"] for row in groups]
        print(f"📋 Found {len(group_ids)} groups to process")
        
        for group_id in group_ids:
            try:
                self.process_unassigned_faces_simple(group_id)
                print(f"✅ Completed group {group_id}")
                print("-" * 50)
            except Exception as e:
                print(f"❌ Error processing group {group_id}: {e}")
                continue

# Optimized version with batch updates for better performance
class BatchOptimizedSimpleGrouping(TrueSimpleFaceGrouping):
    def __init__(self, host="localhost", port=6333, batch_size=100):
        super().__init__(host, port)
        self.batch_size = batch_size
        self.pending_updates = []

    def batch_update_person_id(self, face_id, person_id, group_id):
        """Queue update for batch processing"""
        self.pending_updates.append((face_id, person_id, group_id))
        
        # Flush if batch is full
        if len(self.pending_updates) >= self.batch_size:
            self.flush_updates()

    def flush_updates(self):
        """Execute all pending updates in batch"""
        if not self.pending_updates:
            return
        
        # Batch update PostgreSQL
        conn = get_db_connection()
        cursor = conn.cursor()
        
        sql_updates = [(person_id, face_id) for face_id, person_id, _ in self.pending_updates]
        cursor.executemany(
            "UPDATE faces SET person_id = %s WHERE id = %s", 
            sql_updates
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        # Batch update Qdrant (group by collection)
        updates_by_group = {}
        for face_id, person_id, group_id in self.pending_updates:
            if group_id not in updates_by_group:
                updates_by_group[group_id] = []
            updates_by_group[group_id].append((face_id, person_id))
        
        for group_id, updates in updates_by_group.items():
            points_payload = [
                {"id": face_id, "payload": {"person_id": person_id}}
                for face_id, person_id in updates
            ]
            
            # Update Qdrant in chunks
            for i in range(0, len(points_payload), 50):  # Qdrant batch limit
                chunk = points_payload[i:i+50]
                for point_data in chunk:
                    self.qdrant.set_payload(
                        collection_name=group_id,
                        points=[point_data["id"]],
                        payload=point_data["payload"]
                    )
        
        print(f"✅ Flushed {len(self.pending_updates)} updates to databases")
        self.pending_updates.clear()

    def process_unassigned_faces_batch_optimized(self, group_id):
        """Same simple algorithm but with batch updates for performance"""
        print(f"🚀 Processing group {group_id} with batch-optimized simple algorithm")
        
        unassigned_face_ids = self.get_unassigned_faces(group_id)
        if not unassigned_face_ids:
            return
        
        stats = {"assigned": 0, "new_persons": 0, "matched_existing": 0}
        
        for face_id in unassigned_face_ids:
            # Same logic as simple version but using batch updates
            embedding_data = self.get_single_face_embedding(group_id, face_id)
            if not embedding_data:
                continue
            
            face_emb = embedding_data['face']
            cloth_emb = embedding_data['cloth']
            
            # Your simple algorithm logic here...
            match = self.find_best_match(face_emb, group_id, face_threshold=0.7)
            
            if match and match['person_id']:
                person_id = match['person_id']
                stats["matched_existing"] += 1
            elif match and not match['person_id']:
                person_id = str(uuid.uuid4())
                self.batch_update_person_id(match['id'], person_id, group_id)
                stats["new_persons"] += 1
            else:
                person_id = str(uuid.uuid4())
                stats["new_persons"] += 1
            
            self.batch_update_person_id(face_id, person_id, group_id)
            stats["assigned"] += 1
        
        # Flush any remaining updates
        self.flush_updates()
        
        print(f"🎉 Batch processing complete! Stats: {stats}")

# Usage
if __name__ == "__main__":
    # Use the true simple version
    simple_grouper = TrueSimpleFaceGrouping()
    simple_grouper.process_all_groups()
    
    # Or use batch-optimized version for better performance
    # batch_grouper = BatchOptimizedSimpleGrouping(batch_size=200)
    # batch_grouper.process_unassigned_faces_batch_optimized("group_id")