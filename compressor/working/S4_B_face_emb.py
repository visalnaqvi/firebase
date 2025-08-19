import os
import uuid
import cv2
import torch
import psycopg2
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import concurrent.futures
import numpy as np

@dataclass
class Config:
    BATCH_SIZE: int = 50
    PARALLEL_LIMIT: int = 4
    MAX_RETRIES: int = 3
    
    # Qdrant config
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

config = Config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper cleanup"""
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            dbname="postgres",
            user="postgres",
            password="admin"
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

class DatabaseManager:
    """Handles all database operations for face embeddings"""
    
    @staticmethod
    def fetch_face_extracted_images(group_id: int, batch_size: int) -> List[int]:
        """Fetch images that have faces extracted but need embeddings"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT id FROM images WHERE status = 'face_extracted' AND group_id = %s LIMIT %s", 
                    (group_id, batch_size)
                )
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def fetch_faces_for_images(image_ids: List[int]) -> List[Tuple[str, int, bytes, float]]:
        """Fetch faces for given image IDs"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, image_id, face_thumb_bytes, quality_score FROM faces WHERE image_id = ANY(%s::uuid[]) AND face_thumb_bytes IS NOT NULL", 
                    (image_ids,)
                )
                return cur.fetchall()

    @staticmethod
    def fetch_groups_with_faces() -> List[int]:
        """Fetch groups that have faces extracted"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT group_id FROM images WHERE status = 'face_extracted'"
                )
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def mark_images_embeddings_generated(image_ids: List[int]) -> None:
        """Mark images as having face embeddings generated"""
        if not image_ids:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = "UPDATE images SET status = 'face_embeddings_generated' WHERE id = ANY(%s::uuid[])"
                cur.execute(query, (image_ids,))
                conn.commit()
                logger.info(f"Marked {len(image_ids)} images as face_embeddings_generated")

class FaceEmbeddingGenerator:
    def __init__(self):
        logger.info("Initializing Face Embedding Generator...")
        
        # Initialize models
        try:
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0)
            
            self.qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            
            logger.info("Face embedding models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face embedding models: {e}")
            raise

    def setup_collection(self, collection_name: str) -> None:
        """Setup Qdrant collection for face embeddings"""
        try:
            if not self.qdrant.collection_exists(collection_name):
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "face": VectorParams(size=512, distance=Distance.COSINE),
                        "cloth": VectorParams(size=512, distance=Distance.COSINE)
                    }
                )
            logger.info(f"Collection {collection_name} setup completed")
        except Exception as e:
            logger.error(f"Failed to setup collection {collection_name}: {e}")
            raise

    def generate_face_embedding(self, face_thumb_bytes: bytes) -> Optional[np.ndarray]:
        """Generate face embedding from thumbnail bytes"""
        try:
            # Decode image from bytes
            nparr = np.frombuffer(face_thumb_bytes, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if face_img is None:
                logger.warning("Failed to decode face thumbnail")
                return None

            # Extract face embedding
            faces = self.face_app.get(face_img)
            if not faces:
                logger.warning("No face detected in thumbnail")
                return None
            
            # Return the embedding of the first (and should be only) face
            return faces[0].normed_embedding
            
        except Exception as e:
            logger.error(f"Failed to generate face embedding: {e}")
            return None

    def process_faces_batch(self, faces_data: List[Tuple[str, int, bytes, float]], collection_name: str) -> List[int]:
        """Process batch of faces to generate embeddings"""
        logger.info(f"Processing batch of {len(faces_data)} faces for embedding generation")
        
        processed_image_ids = set()
        successful_embeddings = 0
        
        for face_id, image_id, face_thumb_bytes, quality_score in faces_data:
            try:
                if face_thumb_bytes is None:
                    logger.warning(f"No face thumbnail for face {face_id}")
                    continue
                
                # Generate face embedding
                face_embedding = self.generate_face_embedding(face_thumb_bytes)
                if face_embedding is None:
                    logger.warning(f"Failed to generate embedding for face {face_id}")
                    continue
                
                # Insert/Update face embedding in Qdrant (cloth embedding will be added later)
                self.qdrant.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=face_id,
                            vector={
                                "face": face_embedding.tolist(),
                                "cloth": [0.0] * 512  # Placeholder for cloth embedding
                            },
                            payload={
                                "person_id": None,
                                "image_id": image_id,
                                "cloth_ids": None,
                                "quality_score": quality_score,
                                "has_face_embedding": True,
                                "has_cloth_embedding": False
                            }
                        )
                    ]
                )
                
                processed_image_ids.add(image_id)
                successful_embeddings += 1
                
            except Exception as e:
                logger.error(f"Failed to process face {face_id}: {e}")
                continue
        
        logger.info(f"Generated {successful_embeddings} face embeddings from {len(faces_data)} faces")
        return list(processed_image_ids)

def process_group_face_embeddings(group_id: int, generator: FaceEmbeddingGenerator) -> None:
    """Process a single group for face embedding generation"""
    try:
        # Setup collection
        generator.setup_collection(str(group_id))
        logger.info(f"Processing group {group_id} for face embedding generation")
        
        processed_count = 0
        
        while True:
            # Fetch batch of images that have faces extracted
            image_ids = DatabaseManager.fetch_face_extracted_images(group_id, config.BATCH_SIZE)
            
            if not image_ids:
                logger.info(f"No more face_extracted images for group {group_id}")
                break
            
            logger.info(f"Found {len(image_ids)} face_extracted images for group {group_id}")
            
            # Fetch faces for these images
            faces_data = DatabaseManager.fetch_faces_for_images(image_ids)
            
            if not faces_data:
                logger.warning(f"No faces found for images in group {group_id}")
                # Still mark images as processed to avoid infinite loop
                DatabaseManager.mark_images_embeddings_generated(image_ids)
                continue
            
            # Process faces to generate embeddings
            processed_image_ids = generator.process_faces_batch(faces_data, str(group_id))
            
            # Mark images as having face embeddings generated
            if processed_image_ids:
                DatabaseManager.mark_images_embeddings_generated(processed_image_ids)
            
            processed_count += len(image_ids)
            logger.info(f"Group {group_id}: Processed {processed_count} images so far for face embeddings")
            
            # If we got fewer images than batch size, we're done
            if len(image_ids) < config.BATCH_SIZE:
                break
                
        logger.info(f"Completed face embedding generation for group {group_id}: {processed_count} total images processed")
        
    except Exception as e:
        logger.error(f"Failed to process group {group_id} for face embeddings: {e}")
        raise

def main():
    """Main execution function for face embedding generation"""
    try:
        logger.info("Initializing face embedding generator...")
        generator = FaceEmbeddingGenerator()
        
        # Fetch groups that have faces extracted
        groups = DatabaseManager.fetch_groups_with_faces()
        logger.info(f"Found {len(groups)} groups with extracted faces to process")
        
        if not groups:
            logger.info("No groups with extracted faces found, exiting")
            return
        
        # Process each group
        for group_id in groups:
            try:
                process_group_face_embeddings(group_id, generator)
            except Exception as e:
                logger.error(f"Failed to process group {group_id}, continuing with next group: {e}")
                continue
        
        logger.info("Face embedding generation completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in face embedding generation: {e}")
        raise

if __name__ == "__main__":
    main()