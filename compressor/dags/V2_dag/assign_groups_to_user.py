import psycopg2
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
import numpy as np
import cv2
import logging
import time
from typing import Optional, List, Tuple
from insightface.app import FaceAnalysis
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    def __init__(self, message, user_id=None, reason=None, retryable=True):
        super().__init__(message)
        self.user_id = user_id
        self.reason = reason
        self.retryable = retryable

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            host="ballast.proxy.rlwy.net",
            port="56193",
            dbname="railway",
            user="postgres",
            password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
        )
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

class UserFaceMatching:
    def __init__(self):
        logger.info("Initializing InsightFace model...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0)
        
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        logger.info("Initialization complete")
    
    def fetch_users_with_unmatched_groups(self) -> List[Tuple]:
        """Fetch users who have groups to match"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, email, face_image_bytes, face_embedding, un_matched_groups
                        FROM users
                        WHERE un_matched_groups IS NOT NULL 
                        AND array_length(un_matched_groups, 1) > 0
                    """)
                    results = cur.fetchall()
                    logger.info(f"Found {len(results)} users with unmatched groups")
                    return results
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            raise
    
    def generate_face_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Generate face embedding from image bytes"""
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Detect face
            faces = self.face_app.get(img)
            
            if len(faces) == 0:
                raise ValueError("No face detected in image")
            
            if len(faces) > 1:
                logger.warning("Multiple faces detected, using first face")
            
            # Extract embedding
            face = faces[0]
            embedding = face.normed_embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def search_similar_person_in_qdrant(self, embedding: np.ndarray, group_id: int, threshold: float = 0.5) -> Optional[str]:
        """Search for similar person in Qdrant collection"""
        try:
            collection_name = str(group_id)
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            if not any(c.name == collection_name for c in collections):
                logger.warning(f"Collection {collection_name} does not exist")
                return None
            
            # Search for similar faces
            results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector={"face": embedding.tolist()},
                limit=1,
                score_threshold=threshold
            )
            
            if results and len(results) > 0:
                best_match = results[0]
                logger.info(f"Found match with score {best_match.score} for group {group_id}")
                
                # Get person_id from payload
                person_id = best_match.payload.get('person_id')
                return person_id
            
            logger.info(f"No match found above threshold {threshold} for group {group_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            raise
    
    def get_person_by_id(self, person_id: str, group_id: int) -> Optional[int]:
        """Get person record from persons table"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id FROM persons
                        WHERE person_id = %s AND group_id = %s
                    """, (person_id, group_id))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            logger.error(f"Error fetching person: {e}")
            raise
    
    def update_user_match(self, user_id: int, person_db_id: int, group_id: int, embedding: Optional[np.ndarray] = None):
        """Update user with matched person and remove from unmatched groups"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Update user - remove group from un_matched_groups and optionally save embedding
                    if embedding is not None:
                        cur.execute("""
                            UPDATE users
                            SET un_matched_groups = array_remove(un_matched_groups, %s),
                                face_embedding = %s,
                                matched_at = NOW()
                            WHERE id = %s
                        """, (group_id, embedding.tolist(), user_id))
                    else:
                        cur.execute("""
                            UPDATE users
                            SET un_matched_groups = array_remove(un_matched_groups, %s),
                                matched_at = NOW()
                            WHERE id = %s
                        """, (group_id, user_id))
                    
                    # Update persons table with user_id
                    cur.execute("""
                        UPDATE persons
                        SET user_id = %s,
                            matched_at = NOW()
                        WHERE id = %s
                    """, (user_id, person_db_id))
                    
                    conn.commit()
                    logger.info(f"User {user_id} matched with person {person_db_id} for group {group_id}")
                    
        except Exception as e:
            logger.error(f"Error updating user match: {e}")
            raise
    
    def process_single_user(self, user_data: Tuple) -> dict:
        """Process a single user for face matching"""
        user_id, email, image_bytes, existing_embedding, unmatched_groups = user_data
        
        stats = {
            "user_id": user_id,
            "email": email,
            "groups_processed": 0,
            "groups_matched": 0,
            "groups_failed": 0
        }
        
        try:
            # Step 1: Get or generate embedding
            if existing_embedding is not None:
                logger.info(f"Using existing embedding for user {email}")
                embedding = np.array(existing_embedding, dtype=np.float32)
                save_embedding = False
            else:
                if image_bytes is None:
                    logger.error(f"No image bytes found for user {email}")
                    return stats
                
                logger.info(f"Generating new embedding for user {email}")
                embedding = self.generate_face_embedding(image_bytes)
                save_embedding = True
            
            # Step 2: Process each unmatched group
            for group_id in unmatched_groups:
                stats["groups_processed"] += 1
                
                try:
                    # Search in Qdrant
                    person_id = self.search_similar_person_in_qdrant(embedding, group_id)
                    
                    if person_id:
                        # Get person from database
                        person_db_id = self.get_person_by_id(person_id, group_id)
                        
                        if person_db_id:
                            # Update user match
                            self.update_user_match(
                                user_id, 
                                person_db_id, 
                                group_id, 
                                embedding if save_embedding else None
                            )
                            stats["groups_matched"] += 1
                            logger.info(f"✓ Matched user {email} to person in group {group_id}")
                        else:
                            logger.warning(f"Person {person_id} not found in database for group {group_id}")
                            stats["groups_failed"] += 1
                    else:
                        logger.info(f"No match found for user {email} in group {group_id}")
                        # Still remove from unmatched groups
                        with get_db_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    UPDATE users
                                    SET un_matched_groups = array_remove(un_matched_groups, %s)
                                    WHERE id = %s
                                """, (group_id, user_id))
                                conn.commit()
                        stats["groups_failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing group {group_id} for user {email}: {e}")
                    stats["groups_failed"] += 1
                    continue
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing user {email}: {e}")
            return stats
    
    def process_all_users(self):
        """Main processing loop"""
        try:
            users = self.fetch_users_with_unmatched_groups()
            
            if not users:
                logger.info("No users to process")
                return
            
            total_stats = {
                "users_processed": 0,
                "total_groups_processed": 0,
                "total_groups_matched": 0,
                "total_groups_failed": 0
            }
            
            for user_data in users:
                logger.info(f"Processing user {user_data[1]}...")
                stats = self.process_single_user(user_data)
                
                total_stats["users_processed"] += 1
                total_stats["total_groups_processed"] += stats["groups_processed"]
                total_stats["total_groups_matched"] += stats["groups_matched"]
                total_stats["total_groups_failed"] += stats["groups_failed"]
            
            logger.info("=" * 50)
            logger.info("PROCESSING SUMMARY")
            logger.info(f"Users processed: {total_stats['users_processed']}")
            logger.info(f"Groups processed: {total_stats['total_groups_processed']}")
            logger.info(f"Groups matched: {total_stats['total_groups_matched']}")
            logger.info(f"Groups failed: {total_stats['total_groups_failed']}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error in process_all_users: {e}")
            raise

def main():
    try:
        logger.info("Starting user face matching process...")
        matcher = UserFaceMatching()
        matcher.process_all_users()
        logger.info("User face matching completed successfully")
        return True
    except Exception as e:
        logger.error(f"User face matching failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)