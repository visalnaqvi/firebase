import os
import torch
import psycopg2
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import concurrent.futures
import numpy as np
import cv2
import open_clip
@dataclass
class Config:
    BATCH_SIZE: int = 50
    PARALLEL_LIMIT: int = 4
    PERSON_CONFIDENCE_THRESHOLD: float = 0.5
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
    """Handles all database operations for clothing embeddings"""
    
    @staticmethod
    def fetch_face_embedding_images(group_id: int, batch_size: int) -> List[Tuple[int, bytes]]:
        """Fetch images that have face embeddings but need clothing embeddings"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, image_byte FROM images WHERE status = 'face_embeddings_generated' AND group_id = %s LIMIT %s", 
                    (group_id, batch_size)
                )
                return cur.fetchall()

    @staticmethod
    def fetch_groups_with_face_embeddings() -> List[int]:
        """Fetch groups that have face embeddings generated"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT group_id FROM images WHERE status = 'face_embeddings_generated'"
                )
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def fetch_face_ids_for_image(image_id: int) -> List[str]:
        """Fetch face IDs for a given image"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM faces WHERE image_id = %s", 
                    (image_id,)
                )
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def mark_images_completed(image_ids: List[int]) -> None:
        """Mark images as completely processed (warmed)"""
        if not image_ids:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = "UPDATE images SET status = 'warmed', image_byte = NULL WHERE id = ANY(%s::uuid[])"
                cur.execute(query, (image_ids,))
                conn.commit()
                logger.info(f"Marked {len(image_ids)} images as completed and cleared image_byte")

class ClothingEmbeddingGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.qdrant = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
        # Initialize models
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_name = "ViT-B-32"
            pretrained = "laion2b_s34b_b79k"  # or another CLIP pretrained set

            self.fashion_model, _, self.fashion_processor = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            raise

    def extract_clothing_embedding(self, image_input) -> torch.Tensor:
        """Extract clothing embeddings with proper error handling"""
        try:
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            else:
                img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

            # Transform to tensor
            image_tensor = self.fashion_processor(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.fashion_model.encode_image(image_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb[0]
        except Exception as e:
            logger.error(f"Failed to extract clothing embedding: {e}")
            raise

    def process_image_for_clothing(self, image_id, image_bytes: bytes, yolo_model, collection_name: str) -> List[str]:
        """Process single image to extract clothing embeddings for detected persons"""
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.warning(f"Failed to decode image {image_id}")
                return []

            # Get face IDs for this image
            face_ids = DatabaseManager.fetch_face_ids_for_image(image_id)
            if not face_ids:
                logger.warning(f"No faces found for image {image_id}")
                return []

            # YOLO detection for persons
            results = yolo_model(img)[0]
            updated_face_ids = []
            face_idx = 0  # Index to match persons with faces

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > config.PERSON_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_crop = img[y1:y2, x1:x2]
                    
                    if person_crop.size == 0:
                        continue
                    person_crop_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))

                    # Save
                    filename = f"{image_id}.jpg"
                    crop_path = os.path.join("crp", filename)
                    person_crop_pil.save(crop_path)
                    # Extract clothing embedding for this person
                    try:
                        clothing_emb = self.extract_clothing_embedding(person_crop)
                    except Exception as e:
                        logger.warning(f"Failed to extract clothing embedding for image {image_id}: {e}")
                        continue
                    
                    # Update corresponding face(s) in Qdrant with clothing embedding
                    # Assuming one person can have multiple faces, we'll update all faces for this image
                    for face_id in face_ids[face_idx:]:  # Update remaining faces
                        try:
                            # Get existing point to preserve face embedding
                            existing_points = self.qdrant.retrieve(
                                collection_name=collection_name,
                                ids=[face_id]
                            )
                            
                            if existing_points:
                                existing_point = existing_points[0]
                                # Update with clothing embedding
                                self.qdrant.upsert(
                                    collection_name=collection_name,
                                    points=[
                                        PointStruct(
                                            id=face_id,
                                            vector={
                                                "face": existing_point.vector.get("face", [0.0] * 512),
                                                "cloth": clothing_emb.cpu().tolist()
                                            },
                                            payload={
                                                **existing_point.payload,
                                                "has_cloth_embedding": True
                                            }
                                        )
                                    ]
                                )
                                updated_face_ids.append(face_id)
                                face_idx += 1
                                break  # Assuming one person per detected face for now
                            
                        except Exception as e:
                            logger.error(f"Failed to update face {face_id} with clothing embedding: {e}")
                            continue

            logger.info(f"Processed image {image_id}: Updated {len(updated_face_ids)} faces with clothing embeddings")
            return updated_face_ids
            
        except Exception as e:
            logger.error(f"Failed to process image {image_id} for clothing: {e}")
            return []

    def process_images_batch_for_clothing(self, images_batch: List[Tuple[int, bytes]], yolo_model, collection_name: str) -> List[int]:
        """Process batch of images for clothing embeddings"""
        logger.info(f"Processing batch of {len(images_batch)} images for clothing embeddings")
        
        processed_image_ids = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.PARALLEL_LIMIT) as executor:
            futures = {
                executor.submit(self.process_image_for_clothing, img_id, img_bytes, yolo_model, collection_name): img_id
                for img_id, img_bytes in images_batch
            }
            
            for future in concurrent.futures.as_completed(futures):
                img_id = futures[future]
                try:
                    updated_faces = future.result(timeout=60)
                    if updated_faces:  # If any faces were updated
                        processed_image_ids.append(img_id)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout processing image {img_id} for clothing")
                except Exception as e:
                    logger.error(f"Error processing image {img_id} for clothing: {e}")
        
        return processed_image_ids

def process_group_clothing_embeddings(group_id: int, generator: ClothingEmbeddingGenerator, yolo_model) -> None:
    """Process a single group for clothing embedding generation"""
    try:
        logger.info(f"Processing group {group_id} for clothing embedding generation")
        
        processed_count = 0
        
        while True:
            # Fetch batch of images that have face embeddings generated
            images_batch = DatabaseManager.fetch_face_embedding_images(group_id, config.BATCH_SIZE)
            
            if not images_batch:
                logger.info(f"No more face_embeddings_generated images for group {group_id}")
                break
            
            logger.info(f"Found {len(images_batch)} images with face embeddings for group {group_id}")
            
            # Process images for clothing embeddings
            processed_image_ids = generator.process_images_batch_for_clothing(images_batch, yolo_model, str(group_id))
            
            # Mark all images in batch as completed (regardless of success) and clear image_byte
            all_image_ids = [img_id for img_id, _ in images_batch]
            DatabaseManager.mark_images_completed(all_image_ids)
            
            processed_count += len(images_batch)
            logger.info(f"Group {group_id}: Processed {processed_count} images so far for clothing embeddings")
            
            # If we got fewer images than batch size, we're done
            if len(images_batch) < config.BATCH_SIZE:
                break
                
        logger.info(f"Completed clothing embedding generation for group {group_id}: {processed_count} total images processed")
        
    except Exception as e:
        logger.error(f"Failed to process group {group_id} for clothing embeddings: {e}")
        raise

def main():
    """Main execution function for clothing embedding generation"""
    try:
        # Initialize YOLO model (needed for person detection)
        logger.info("Initializing YOLO model...")
        yolo_model = YOLO("yolov8x.pt")
        
        logger.info("Initializing clothing embedding generator...")
        generator = ClothingEmbeddingGenerator()
        
        # Fetch groups that have face embeddings generated
        groups = DatabaseManager.fetch_groups_with_face_embeddings()
        logger.info(f"Found {len(groups)} groups with face embeddings to process")
        
        if not groups:
            logger.info("No groups with face embeddings found, exiting")
            return
        
        # Process each group
        for group_id in groups:
            try:
                process_group_clothing_embeddings(group_id, generator, yolo_model)
            except Exception as e:
                logger.error(f"Failed to process group {group_id}, continuing with next group: {e}")
                continue
        
        logger.info("Clothing embedding generation completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in clothing embedding generation: {e}")
        raise

if __name__ == "__main__":
    main()