import os
import numpy as np
from scipy.spatial.distance import cosine
import torch
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import DictCursor
import uuid
import json
from collections import defaultdict
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import time
from typing import Optional
class ProcessingError(Exception):
    def __init__(self, message, group_id=None, reason=None, retryable=True):
        super().__init__(message)
        self.group_id = group_id
        self.reason = reason
        self.retryable = retryable

    def __str__(self):
        return f"ProcessingError: {self.args[0]} (group_id={self.group_id}, reason={self.reason}, retryable={self.retryable})"
def get_or_assign_group_id():
    """
    Fetch the active group_id for extraction task.
    - If processing_group has a value → return it
    - Else if next_group_in_queue has a value → move it to processing_group,
    set next_group_in_queue = NULL, return it
    - Else return None
    """
    conn = None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Fetch both columns
                cur.execute(
                    """
                    SELECT processing_group, next_group_in_queue
                    FROM process_status
                    WHERE task = 'grouping'
                    LIMIT 1
                    """
                )
                row = cur.fetchone()

                if not row:
                    return None

                processing_group, next_group_in_queue = row

                if processing_group:
                    return processing_group

                if next_group_in_queue:
                    # Promote next_group_in_queue → processing_group
                    cur.execute(
                        """
                        UPDATE process_status
                        SET processing_group = %s,
                            next_group_in_queue = NULL
                        WHERE task = 'grouping'
                        """,
                        (next_group_in_queue,)
                    )
                    conn.commit()
                    return next_group_in_queue

                return None
    except Exception as e:
        print("❌ Error in get_or_assign_group_id:", e)
        return None


def update_status_history(
    run_id: int,
    task: str,
    sub_task: str,
    totalImagesInitialized: int,
    totalImagesFailed: int,
    totalImagesProcessed: int,
    groupId: Optional[str],
    fail_reason: Optional[str]
) -> bool:
    """
    Insert a record into process_history.
    Returns True if insert succeeded, False otherwise.
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO process_history
                        (worker_id, run_id, task, sub_task,
                        initialized_count, success_count, failed_count,
                        group_id, ended_at, fail_reason)
                    VALUES
                        (%s, %s, %s, %s,
                        %s, %s, %s,
                        %s, NOW(), %s)
                    """,
                    (
                        1,                       # worker_id
                        run_id,
                        task,
                        sub_task,
                        totalImagesInitialized,
                        totalImagesProcessed,
                        totalImagesFailed,
                        groupId,
                        fail_reason,
                    )
                )
                conn.commit()
                return True
    except Exception as e:
        print(f"❌ Error inserting into process_history: {e}")
        return False
    
    
def update_status(group_id, fail_reason, is_ideal , status):
    """
    Updates process_status table where task = 'extraction'
    Returns a dict with success flag and optional error.
    """
    conn = None

    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if (status=='failed'):
                    cur.execute(
                        """
                        UPDATE process_status
                        SET task_status = %s,
                            fail_reason = %s,
                            ended_at = NOW(),
                            is_ideal = %s
                        WHERE task = 'grouping'
                        """,
                        (status , fail_reason, is_ideal)
                    )
                else:
                    cur.execute(
                        """
                        UPDATE process_status
                        SET task_status = %s,
                            processing_group = %s,
                            fail_reason = %s,
                            ended_at = NOW(),
                            is_ideal = %s
                        WHERE task = 'grouping'
                        """,
                        (status , group_id, fail_reason, is_ideal)
                    )
            conn.commit()
            return {"success": True}
    except Exception as e:
        print("❌ Error updating process status:", e)
        if conn:
            conn.rollback()
        return {"success": False, "errorReason": "updating status", "error": str(e)}
    finally:
        if conn:
            conn.close()
def get_db_connection():
    return psycopg2.connect(
         host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )
def update_last_provrssed_group_column(group_id):
        """
        Updates process_status table where task = 'extraction'
        Returns a dict with success flag and optional error.
        """
        conn = None
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE process_status
                        SET last_group_processed = %s
                        WHERE task = 'grouping'
                        """,
                        (group_id,)
                    )
                    cur.execute(
                        """
                        UPDATE process_status
                        SET next_group_in_queue = %s
                        WHERE task = 'insertion' and next_group_in_queue is null 
                        """,
                        (group_id,)
                    )
                    if cur.rowcount == 0:
                            raise Exception("No rows updated for quality_assignment (next_group_in_queue was not NULL)")
                conn.commit()
                return {"success": True}
        except Exception as e:
            print("❌ Error updating process status:", e)
            if conn:
                conn.rollback()
            return {"success": False, "errorReason": "updating status", "error": str(e)}
        finally:
            if conn:
                conn.close()
class SimplifiedFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        self.qdrant = QdrantClient(host=host, port=port)
        self.faces_cache = {}  # In-memory cache for faces data
        self.error_faces = set()  # Track faces with errors

    
    def load_faces_json(self, group_id):
        """Load faces data from JSON file into memory cache"""
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "warm-images",str(group_id), "faces", "faces.json")

            if not os.path.exists(json_path):
                print(f"[ERROR] JSON file not found: {json_path}")
                raise ProcessingError(f"[ERROR] JSON file not found: {json_path}")
            with open(json_path, 'r') as f:
                faces_data = json.load(f)
            
            # Convert to dict with face_id as key for faster lookups
            faces_dict = {face['id']: face for face in faces_data}
            self.faces_cache[group_id] = faces_dict
            print(f"[SUCCESS] Loaded {len(faces_dict)} faces from JSON into cache")
            return faces_dict
            
        except Exception as e:
            print(f"[ERROR] Error loading faces JSON: {e}")
            raise

    def save_faces_json(self, group_id):
        """Save faces data from memory cache to JSON file"""
        if group_id not in self.faces_cache:
            print(f"[WARNING] No cache data found for group {group_id}")
            return
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "warm-images",  str(group_id),"faces",  "faces.json")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            # Convert dict back to list format
            faces_list = list(self.faces_cache[group_id].values())
            
            with open(json_path, 'w') as f:
                json.dump(faces_list, f, indent=2)
            
            print(f"[SUCCESS] Saved {len(faces_list)} faces to JSON file at {json_path}")
            
        except Exception as e:
            print(f"[ERROR] Error saving faces JSON: {e}")



    def get_unassigned_faces_batch(self, group_id, limit=10):
        try:
            """Get unassigned faces from memory cache"""
            if group_id not in self.faces_cache:
                faces_dict = self.load_faces_json(group_id)
            else:
                faces_dict = self.faces_cache[group_id]
            
            # Find unassigned faces that are not in error state
            unassigned_faces = [
                face_id for face_id, face_data in faces_dict.items()
                if face_data.get('person_id') is None 
                and face_data.get('status') != 'error_groupping'
                and face_id not in self.error_faces
            ]
            
            # Return limited batch
            batch = unassigned_faces[:limit]
            print(f"[SUCCESS] Found {len(batch)} unassigned faces from cache")
            return batch
        except Exception as e:
            raise

    def update_face_in_cache(self, group_id, face_id, updates):
        try:
            """Update face data in memory cache"""
            if group_id not in self.faces_cache:
                return
            
            if face_id in self.faces_cache[group_id]:
                self.faces_cache[group_id][face_id].update(updates)
        except Exception as e:
            raise
        
    def mark_faces_error_batch(self, group_id, error_face_ids):
        try:
            """Mark multiple faces as error in cache"""
            if not error_face_ids or group_id not in self.faces_cache:
                return
            
            for face_id in error_face_ids:
                self.update_face_in_cache(group_id, face_id, {'status': 'error_groupping'})
            
            print(f"[SUCCESS] Marked {len(error_face_ids)} faces as error in cache")
        except Exception as e:
            raise

    def get_face_embedding(self, group_id, face_id):
        """Get single face embedding from Qdrant"""
        try:
            points = self.qdrant.retrieve(
                collection_name=group_id,
                ids=[face_id],
                with_payload=True,
                with_vectors=True
            )
            if not points or len(points) == 0:
                print(f"[WARNING] No points found for face {face_id}")
                self.error_faces.add(face_id)
                return None

            point = points[0]
            vectors = getattr(points[0], "vectors", None) or getattr(points[0], "vector", None)
            payload = getattr(point, "payload", {}) or {}
            person_id = payload.get("person_id")
            
            if vectors:
                return {
                    "face": np.array(vectors.get("face", [])),
                    "cloth": torch.tensor(vectors.get("cloth", [])) if vectors.get("cloth") else None,
                    "person_id": person_id
                }

            print(f"[WARNING] No vectors found for face {face_id}")
            self.error_faces.add(face_id)
            return None

        except Exception as e:
            print(f"[WARNING] Error retrieving embedding for face {face_id}: {e}")
            self.error_faces.add(face_id)
            return None

    def find_face_candidates(self, face_embedding, group_id, person_id, threshold=0.4, limit=1000):
        """Find face candidates using face similarity excluding given person_id"""
        try:
            if face_embedding is None:
                print("[WARNING] find_face_candidates called with None embedding")
                return [], []

            # Build filter to exclude given person_id if provided
            filter_obj = None
            if person_id:
                filter_obj = Filter(
                    must_not=[
                        FieldCondition(key="person_id", match=MatchValue(value=person_id))
                    ])

            candidates = self.qdrant.query_points(
                collection_name=group_id,
                query=face_embedding.tolist(),
                using="face",
                score_threshold=threshold,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                query_filter=filter_obj 
            )

            points = getattr(candidates, "points", None)
            if points is None:
                if isinstance(candidates, list):
                    points = candidates
                elif isinstance(candidates, dict) and "points" in candidates:
                    points = candidates["points"]
                else:
                    print("[WARNING] qdrant.query_points returned no points")
                    return [], []

            face_matches = []
            face_matches_less = []
            print(f"found {len(points)} face candidate(s)")

            for candidate in points:
                payload = candidate.payload or {}
                raw_cloth_ids = payload.get("cloth_ids") or []
                try:
                    cloth_ids_set = set(raw_cloth_ids)
                except TypeError:
                    cloth_ids_set = set()

                match_data = {
                    "id": candidate.id,
                    "score": candidate.score,
                    "person_id": payload.get("person_id"),
                    "cloth_ids": cloth_ids_set
                }

                if candidate.score >= 0.6:
                    face_matches.append(match_data)
                else:
                    face_matches_less.append(match_data)

            return face_matches, face_matches_less

        except Exception as e:
            print(f"[ERROR] Error finding face candidates: {e}")
            return [], []

    def find_cloth_candidates(self, cloth_embedding, face_matches_less, group_id, person_id, threshold=0.85, limit=1000):
        """Find cloth candidates that also appear in face_matches_less (defensive)"""
        if cloth_embedding is None:
            return []

        # Ensure face_matches_less is iterable
        if face_matches_less is None:
            face_matches_less = []

        try:
            face_ids = {m["id"] for m in face_matches_less}

            # Build filter to exclude given person_id if provided
            filter_obj = None
            if person_id:
                filter_obj = Filter(
                    must_not=[
                        FieldCondition(key="person_id", match=MatchValue(value=person_id))
                    ])

            candidates = self.qdrant.query_points(
                collection_name=group_id,
                query=cloth_embedding.tolist(),
                using="cloth",
                score_threshold=threshold,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                query_filter=filter_obj
            )

            points = getattr(candidates, "points", None)
            if points is None:
                if isinstance(candidates, list):
                    points = candidates
                elif isinstance(candidates, dict) and "points" in candidates:
                    points = candidates["points"]
                else:
                    return []

            matching_candidates = []
            for candidate in points:
                if candidate.id in face_ids:
                    payload = candidate.payload or {}
                    raw_cloth_ids = payload.get("cloth_ids") or []
                    try:
                        cloth_ids_set = set(raw_cloth_ids)
                    except TypeError:
                        cloth_ids_set = set()

                    score_face = next((m["score"] for m in face_matches_less if m["id"] == candidate.id), None)

                    matching_candidates.append({
                        "id": candidate.id,
                        "score": score_face,
                        "score_cloth": candidate.score,
                        "person_id": payload.get("person_id"),
                        "cloth_ids": cloth_ids_set
                    })

            return matching_candidates

        except Exception as e:
            print(f"[WARNING] Error finding cloth candidates: {e}")
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

    def handle_face_matching(self, face_matches, new_person_id, is_new):
        try:
            """Handle face matching cases a, b, c"""
            unassigned, assigned = self.analyze_face_candidates(face_matches)
            similar_faces = []
            final_person_id = new_person_id
            
            print(f"   [INFO] Face analysis: {len(unassigned)} unassigned, {len(assigned)} assigned groups")
            
            if not assigned:
                # Case A: All candidates have unassigned person_id
                print("   [INFO] Case A: All unassigned - assigning new UUID to all")
                faces_to_update = [match['id'] for match in unassigned]
                
            elif len(assigned) == 1:
                # Case B: Some unassigned, some assigned to one person
                person_a = list(assigned.keys())[0]
                person_a_matches = assigned[person_a]
                
                # Check if any assigned match has score > 0.8
                high_score_matches = [m for m in person_a_matches if m['score'] > 0.8]
                
                if is_new:
                    print(f"   [INFO] Case B: High score match found (>{0.8}) - is_new is true assigning existing person_id: {person_a}")
                    final_person_id = person_a
                    faces_to_update = [match['id'] for match in unassigned]
                else:
                    print(f"   [INFO] Case B: high score match - is_new is false assigning new_person_id UUID")
                    faces_to_update = []
                    for matches_list in assigned.values():
                        faces_to_update.extend([match['id'] for match in matches_list])
                    
            else:
                # Case C: Multiple assigned person_ids
                print(f"   [INFO] Case C: Multiple assigned persons - checking scores")
                best_person_id = None
                best_score = 0
                
                # Find the person with highest score > 0.8
                for person_id, matches in assigned.items():
                    for match in matches:
                        if match['score'] > 0.8 and match['score'] > best_score:
                            best_score = match['score']
                            best_person_id = person_id
                
                if best_person_id:
                    print(f"   [INFO] Case C: High score match found - assigning person_id: {best_person_id}")
                    final_person_id = best_person_id
                    faces_to_update = [match['id'] for match in unassigned]
                    # Add other person_ids to similar faces
                    for person_id in assigned.keys():
                        if person_id != best_person_id:
                            similar_faces.append(person_id)
                else:
                    print(f"   [INFO] Case C: No high score match - assigning new UUID, adding all to similar faces")
                    faces_to_update = [match['id'] for match in unassigned]
                    similar_faces.extend(assigned.keys())
            
            return final_person_id, faces_to_update, similar_faces
        except Exception as e:
            raise

    def update_embeddings_batch(self, group_id, updates):
        try:
            """Batch update metadata (payload) in Qdrant."""
            if not updates:
                return

            print(f"   [PROCESSING] Preparing to update {len(updates)} Qdrant payloads for group {group_id}")

            for face_id, person_id, cloth_ids in updates:
                payload = {
                    "person_id": person_id,
                    "cloth_ids": list(cloth_ids) if cloth_ids else []
                }
                try:
                    self.qdrant.set_payload(
                        collection_name=group_id,
                        payload=payload,
                        points=[face_id]
                    )
                except Exception as e:
                    print(f"   [ERROR] Error setting payload for {face_id}: {e}")

            print(f"   [SUCCESS] Payload updates attempted for {len(updates)} points")
        except Exception as e:
            raise

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
                    person_id VARCHAR(255),
                    similar_person_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert similar faces data
            insert_data = []
            for person_id, similar_person_ids in similar_faces_data.items():
                for similar_person_id in similar_person_ids:
                    insert_data.append((group_id, person_id, similar_person_id))
            
            if insert_data:
                cursor.executemany(
                    "INSERT INTO similar_faces (group_id, person_id, similar_person_id) VALUES (%s, %s, %s)",
                    insert_data
                )
                conn.commit()
                print(f"[SUCCESS] Inserted {len(insert_data)} similar face records")
            
        except Exception as e:
            print(f"[ERROR] Error inserting similar faces: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def process_face_batch(self, group_id, batch_size=10):
        try:
            """Process a batch of unassigned faces"""
            print(f"[PROCESSING] Processing batch of {batch_size} faces for group {group_id}")
            
            # Clear error faces set at start of each batch
            self.error_faces.clear()
            
            # Get batch of unassigned faces
            unassigned_face_ids = self.get_unassigned_faces_batch(group_id, batch_size)
            if not unassigned_face_ids:
                print("No unassigned faces found")
                return
            
            face_assignments = {}  # face_id -> person_id
            similar_faces_data = {}  # face_id -> [similar_person_ids]
            all_qdrant_updates = []  # Collect all Qdrant updates
            
            for face_id in unassigned_face_ids:
                print(f"\n[PROCESSING] Processing face {face_id}")
                
                # Step 1: Get face embedding
                embedding_data = self.get_face_embedding(group_id, face_id)
                if not embedding_data:
                    print(f"   [WARNING] Could not get embedding for face {face_id}")
                    continue
                
                face_emb = embedding_data['face']
                cloth_emb = embedding_data['cloth']
                person_id = embedding_data['person_id']
                
                # Step 2: Generate or use person_id
                if not person_id:
                    new_person_id = str(uuid.uuid4())
                    is_new = True
                else:
                    new_person_id = person_id
                    is_new = False
                
                # Step 3: Find face and cloth candidates
                face_matches, face_matches_less = self.find_face_candidates(face_emb, group_id, person_id)
                cloth_matches = self.find_cloth_candidates(cloth_emb, face_matches_less, group_id, person_id)
                
                print(f"   [INFO] Found {len(face_matches)} face matches, {len(cloth_matches)} cloth matches")
                
                # Step 4: Handle face matching logic
                final_person_id, faces_to_update, similar_faces = self.handle_face_matching(
                    face_matches + cloth_matches, new_person_id, is_new
                )
                
                # Step 5: Store assignments and similar faces
                face_assignments[face_id] = final_person_id
                
                if similar_faces:
                    similar_faces_data[final_person_id] = similar_faces
                
                # Step 6: Prepare cloth_ids updates for all matches
                cloth_ids = set()
                
                # Add final_person_id to cloth_ids for all cloth matches
                for cloth_match in cloth_matches:
                    cloth_ids_to_update = cloth_match['cloth_ids'].copy()
                    cloth_ids_to_update.add(final_person_id)
                    all_qdrant_updates.append((cloth_match['id'], cloth_match['person_id'], cloth_ids_to_update))
                
                # Update the current face
                all_qdrant_updates.append((face_id, final_person_id, cloth_ids))
                
                # Update faces that should get the same person_id
                for update_face_id in faces_to_update:
                    if update_face_id != face_id:  # Don't double-update the current face
                        all_qdrant_updates.append((update_face_id, final_person_id, set()))
            
            # Step 7: Batch updates
            print(f"Performing batch updates...")
            
            # Update Qdrant
            self.update_embeddings_batch(group_id, all_qdrant_updates)
            
            # Update faces in cache
            for face_id, person_id in face_assignments.items():
                self.update_face_in_cache(group_id, face_id, {'person_id': person_id})
            
            # Mark error faces in cache
            if self.error_faces:
                self.mark_faces_error_batch(group_id, list(self.error_faces))
            
            # Insert similar faces to DB
            self.insert_similar_faces(group_id, similar_faces_data)
            
            # Save updated cache to JSON file (single write operation)
            self.save_faces_json(group_id)
            
            print(f" Batch processing complete! Processed {len(unassigned_face_ids)} faces")
            print(f"   [INFO] Assignments: {len(face_assignments)}, Similar faces: {len(similar_faces_data)}")
            print(f"   [ERROR] Error faces: {len(self.error_faces)}")
        except Exception as e:
            raise

    def mark_group_processed(self, group_id) -> None:
        """Mark group_id as processed"""
        if not group_id:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    UPDATE groups
                    SET status = 'warmed',
                        last_processed_at = NOW(),
                        last_processed_step = 'grouping'
                    WHERE id = %s AND status = 'warming'
                """
                cur.execute(query, (group_id,))
                conn.commit()
                print(f"Marked {group_id} group_id as processed")

    def process_unassigned_faces(self, group_id, batch_size=10):
        """Process all unassigned faces in batches"""
        print(f"[PROCESSING] Starting face processing for group {group_id}")
        try:
            # Load faces data once at the beginning
            if group_id not in self.faces_cache:
                self.load_faces_json(group_id)
            
            processed_batches = 0
            while True:
                # Process one batch
                initial_unassigned = self.get_unassigned_faces_batch(group_id, 1)
                if not initial_unassigned:
                    print("[SUCCESS] All faces processed!")
                    break
                    
                self.process_face_batch(group_id, batch_size)
                processed_batches += 1
                
                # Safety check to avoid infinite loops
                if processed_batches > 100:  # Adjust as needed
                    print("[WARNING] Maximum batch limit reached, stopping")
                    break
        except Exception as e:
            raise

    def process_all_groups(self, batch_size=10):
        """Process all warming groups"""
       
        try:
            run_id = int(time.time())
            group_id = get_or_assign_group_id()
            if not group_id:
                update_status(None , "No Group Found for processing" , True , "waiting")
                update_status_history(run_id , "grouping" , "group" , None , None , None , group_id , "no_group")
                return True
            update_status(group_id , "running" , False , "healthy")
            update_status_history(run_id , "grouping" , "group" , None , None , None , group_id , "started")
            print(f"[INFO] Found {group_id} groups to process")
            self.process_unassigned_faces(group_id, batch_size)
            self.mark_group_processed(group_id)
            print(f"[SUCCESS] Completed group {group_id}")
            
            # Clear cache for this group to free memory
            if group_id in self.faces_cache:
                del self.faces_cache[group_id]
                
            update_status(None , "" , True , "done")
            update_status_history(run_id , "grouping" , "group" , None , None , None , group_id , "done")
            update_last_provrssed_group_column(group_id)
            return True
        except Exception as e:
            print(f"[ERROR] Error processing group {group_id}: {e}")
            update_status(group_id , f"Error while trying grouping {e}" , True , "failed")
            update_status_history(run_id , "grouping" , "group" , None , None , None , group_id , f"error while trying grouping {e}")
            return False
            

#  Usage Example
if __name__ == "__main__":
    
    grouper = SimplifiedFaceGrouping()
    # Process all groups with batch size of 10
    success = grouper.process_all_groups(batch_size=10)
    exit(0 if success else 1)
    