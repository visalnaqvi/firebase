import os
import numpy as np
from scipy.spatial.distance import cosine
import torch
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import DictCursor
import uuid
from collections import defaultdict

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

class SimplifiedFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        self.qdrant = QdrantClient(host=host, port=port)

    def get_unassigned_faces_batch(self, group_id, limit=100):
        """Get 100 unassigned faces from PostgreSQL"""
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        print(f"🔃 Fetching {limit} unassigned faces for group {group_id}")
        cursor.execute(
            "SELECT id FROM faces WHERE group_id = %s AND person_id IS NULL LIMIT %s", 
            (group_id, limit)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        face_ids = [row["id"] for row in rows]
        print(f"✅ Found {len(face_ids)} unassigned faces")
        return face_ids

    def get_face_embedding(self, group_id, face_id):
        """Get single face embedding from Qdrant"""
        try:
            points = self.qdrant.retrieve(
                collection_name=group_id,
                ids=[face_id],
                with_payload=True,
                with_vectors=True
            )
            
            if points and points[0].vectors:
                return {
                    'face': np.array(points[0].vectors.get("face", [])),
                    'cloth': torch.tensor(points[0].vectors.get("cloth", [])) if points[0].vectors.get("cloth") else None,
                }
            return None
        except Exception as e:
            print(f"⚠️ Error retrieving embedding for face {face_id}: {e}")
            return None

    def find_face_candidates(self, face_embedding, group_id, threshold=0.4, limit=50):
        """Find face candidates using face similarity"""
        try:
            candidates = self.qdrant.query_points(
                collection_name=group_id,
                query=face_embedding.tolist(),
                using="face",
                score_threshold=threshold,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            
            face_matches = []       # >= 0.7
            face_matches_less = []  # < 0.7

            for candidate in candidates.points:
                match_data = {
                    'id': candidate.id,
                    'score': candidate.score,
                    'person_id': candidate.payload.get('person_id') if candidate.payload else None,
                    'cloth_ids': set(candidate.payload.get('cloth_ids', [])) if candidate.payload else set()
                }

                if candidate.score >= 0.7:
                    face_matches.append(match_data)
                else:
                    face_matches_less.append(match_data)
            
            return face_matches, face_matches_less

        except Exception as e:
            print(f"❌ Error finding face candidates: {e}")
            return [], []

    def find_cloth_candidates(self, cloth_embedding, face_matches_less, group_id, threshold=0.85, limit=50):
        """Find cloth candidates that also appear in face_matches_less."""
        if cloth_embedding is None:
            return []

        try:
            # Create a set of IDs from face_matches_less for quick lookup
            face_ids = {match['id'] for match in face_matches_less}

            # Cloth similarity search
            candidates = self.qdrant.query_points(
                collection_name=group_id,
                query=cloth_embedding.tolist(),
                using="cloth",
                score_threshold=threshold,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )

            matching_candidates = []
            for candidate in candidates.points:
                if candidate.id in face_ids:  # Must also be in face_matches_less
                    matching_candidates.append({
                        'id': candidate.id,
                        'score_face': next(m['score'] for m in face_matches_less if m['id'] == candidate.id),
                        'score_cloth': candidate.score,
                        'person_id': candidate.payload.get('person_id') if candidate.payload else None,
                        'cloth_ids': set(candidate.payload.get('cloth_ids', [])) if candidate.payload else set()
                    })

            return matching_candidates

        except Exception as e:
            print(f"⚠️ Error finding cloth candidates: {e}")
            return []

    def analyze_face_candidates(self, face_matches):
        """Analyze face candidates and categorize by person_id assignment"""
        unassigned = []
        assigned = defaultdict(list)
        
        for match in face_matches:
            if match['person_id'] is None:
                unassigned.append(match)
            else:
                assigned[match['person_id']].append(match)
        
        return unassigned, assigned

    def handle_face_matching(self, face_matches, new_person_id):
        """Handle face matching cases a, b, c"""
        unassigned, assigned = self.analyze_face_candidates(face_matches)
        similar_faces = []
        final_person_id = new_person_id
        
        print(f"   📊 Face analysis: {len(unassigned)} unassigned, {len(assigned)} assigned groups")
        
        if not assigned:
            # Case A: All candidates have unassigned person_id
            print("   📝 Case A: All unassigned - assigning new UUID to all")
            faces_to_update = [match['id'] for match in unassigned]
            
        elif len(assigned) == 1:
            # Case B: Some unassigned, some assigned to one person
            person_a = list(assigned.keys())[0]
            person_a_matches = assigned[person_a]
            
            # Check if any assigned match has score > 0.8
            high_score_matches = [m for m in person_a_matches if m['score'] > 0.8]
            
            if high_score_matches:
                print(f"   📝 Case B: High score match found (>{0.8}) - assigning existing person_id: {person_a}")
                final_person_id = person_a
                faces_to_update = [match['id'] for match in unassigned]
            else:
                print(f"   📝 Case B: No high score match - assigning new UUID, adding to similar faces")
                faces_to_update = [match['id'] for match in unassigned]
                similar_faces.append(person_a)
                
        else:
            # Case C: Multiple assigned person_ids
            print(f"   📝 Case C: Multiple assigned persons - checking scores")
            best_person_id = None
            best_score = 0
            
            # Find the person with highest score > 0.8
            for person_id, matches in assigned.items():
                for match in matches:
                    if match['score'] > 0.8 and match['score'] > best_score:
                        best_score = match['score']
                        best_person_id = person_id
            
            if best_person_id:
                print(f"   📝 Case C: High score match found - assigning person_id: {best_person_id}")
                final_person_id = best_person_id
                faces_to_update = [match['id'] for match in unassigned]
                # Add other person_ids to similar faces
                for person_id in assigned.keys():
                    if person_id != best_person_id:
                        similar_faces.append(person_id)
            else:
                print(f"   📝 Case C: No high score match - assigning new UUID, adding all to similar faces")
                faces_to_update = [match['id'] for match in unassigned]
                similar_faces.extend(assigned.keys())
        
        return final_person_id, faces_to_update, similar_faces

    def update_embeddings_batch(self, group_id, updates):
        """Batch update embeddings in Qdrant"""
        points_to_update = []
        
        for face_id, person_id, cloth_ids in updates:
            points_to_update.append({
                "id": face_id,
                "payload": {
                    "person_id": person_id,
                    "cloth_ids": list(cloth_ids)
                }
            })
        
        if points_to_update:
            try:
                self.qdrant.upsert(
                    collection_name=group_id,
                    points=points_to_update
                )
                print(f"   ✅ Updated {len(points_to_update)} embeddings in Qdrant")
            except Exception as e:
                print(f"   ❌ Error updating Qdrant: {e}")

    def update_postgres_batch(self, face_assignments):
        """Batch update PostgreSQL faces table"""
        if not face_assignments:
            return
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Batch update faces
            update_data = [(person_id, face_id) for face_id, person_id in face_assignments.items()]
            cursor.executemany(
                "UPDATE faces SET person_id = %s WHERE id = %s",
                update_data
            )
            conn.commit()
            print(f"✅ Updated {len(update_data)} faces in PostgreSQL")
            
        except Exception as e:
            print(f"❌ Error updating PostgreSQL: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def insert_similar_faces(self, group_id, similar_faces_data):
        """Insert similar faces into similar_faces table"""
        if not similar_faces_data:
            return
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS similar_faces (
                    id SERIAL PRIMARY KEY,
                    group_id VARCHAR(255),
                    face_id VARCHAR(255),
                    similar_person_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert similar faces data
            insert_data = []
            for face_id, similar_person_ids in similar_faces_data.items():
                for similar_person_id in similar_person_ids:
                    insert_data.append((group_id, face_id, similar_person_id))
            
            if insert_data:
                cursor.executemany(
                    "INSERT INTO similar_faces (group_id, face_id, similar_person_id) VALUES (%s, %s, %s)",
                    insert_data
                )
                conn.commit()
                print(f"✅ Inserted {len(insert_data)} similar face records")
            
        except Exception as e:
            print(f"❌ Error inserting similar faces: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def process_face_batch(self, group_id, batch_size=100):
        """Process a batch of unassigned faces"""
        print(f"🚀 Processing batch of {batch_size} faces for group {group_id}")
        
        # Get batch of unassigned faces
        unassigned_face_ids = self.get_unassigned_faces_batch(group_id, batch_size)
        if not unassigned_face_ids:
            print("No unassigned faces found")
            return
        
        face_assignments = {}  # face_id -> person_id
        similar_faces_data = {}  # face_id -> [similar_person_ids]
        qdrant_updates = []  # (face_id, person_id, cloth_ids)
        
        for face_id in unassigned_face_ids:
            print(f"\n🔍 Processing face {face_id}")
            
            # Step 1: Generate new UUID for this face
            new_person_id = str(uuid.uuid4())
            
            # Step 2: Get face embedding
            embedding_data = self.get_face_embedding(group_id, face_id)
            if not embedding_data:
                print(f"   ⚠️ Could not get embedding for face {face_id}")
                continue
            
            face_emb = embedding_data['face']
            cloth_emb = embedding_data['cloth']
            
            # Step 3: Find face and cloth candidates
            face_matches , face_matches_less  = self.find_face_candidates(face_emb, group_id)
            cloth_matches = self.find_cloth_candidates(cloth_emb,face_matches_less , group_id)
            
            print(f"   📋 Found {len(face_matches)} face matches, {len(cloth_matches)} cloth matches")
            
            # Step 4: Handle face matching logic
            final_person_id, faces_to_update, similar_faces = self.handle_face_matching(
                face_matches, new_person_id
            )
            
            # Step 5: Store assignments and similar faces
            face_assignments[face_id] = final_person_id
            
            if similar_faces:
                similar_faces_data[face_id] = similar_faces
            
            # Step 6: Prepare cloth_ids updates for all matches
            cloth_ids = set()
            
            # Add final_person_id to cloth_ids for all cloth matches
            for cloth_match in cloth_matches:
                cloth_ids_to_update = cloth_match['cloth_ids'].copy()
                cloth_ids_to_update.add(final_person_id)
                qdrant_updates.append((cloth_match['id'], cloth_match['person_id'], cloth_ids_to_update))
            
            # Update the current face
            qdrant_updates.append((face_id, final_person_id, cloth_ids))
            
            # Update faces that should get the same person_id
            for update_face_id in faces_to_update:
                if update_face_id != face_id:  # Don't double-update the current face
                    qdrant_updates.append((update_face_id, final_person_id, set()))
                    face_assignments[update_face_id] = final_person_id
        
        # Step 7: Batch updates
        print(f"\n💾 Performing batch updates...")
        self.update_embeddings_batch(group_id, qdrant_updates)
        self.update_postgres_batch(face_assignments)
        self.insert_similar_faces(group_id, similar_faces_data)
        
        print(f"🎉 Batch processing complete! Processed {len(unassigned_face_ids)} faces")
        print(f"   📊 Assignments: {len(face_assignments)}, Similar faces: {len(similar_faces_data)}")

    def process_unassigned_faces(self, group_id, batch_size=100):
        """Process all unassigned faces in batches"""
        print(f"🚀 Starting face processing for group {group_id}")
        
        while True:
            # Process one batch
            initial_count = len(self.get_unassigned_faces_batch(group_id, 1))
            if initial_count == 0:
                print("✅ All faces processed!")
                break
                
            self.process_face_batch(group_id, batch_size)
            
            # Check if there are more faces to process
            remaining_count = len(self.get_unassigned_faces_batch(group_id, 1))
            if remaining_count == initial_count:
                print("⚠️ No progress made, stopping to avoid infinite loop")
                break

    def process_all_groups(self, batch_size=100):
        """Process all warmed groups"""
        # Get all warmed groups
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
                self.process_unassigned_faces(group_id, batch_size)
                print(f"✅ Completed group {group_id}")
            except Exception as e:
                print(f"❌ Error processing group {group_id}: {e}")
                continue

# 🔧 Usage Example
if __name__ == "__main__":
    grouper = SimplifiedFaceGrouping()
    
    # Process all groups with batch size of 100
    grouper.process_all_groups(batch_size=100)
    
    # Or process a specific group
    # grouper.process_unassigned_faces("specific_group_id", batch_size=100)