import os
import uuid
import cv2
import torch
import psycopg2
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator
import logging
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from psycopg2.extras import execute_values
import concurrent.futures
import numpy as np

# Configuration
@dataclass
class Config:
    BATCH_SIZE: int = 50  # Increased for better efficiency
    PARALLEL_LIMIT: int = 4  # Reduced to prevent resource exhaustion
    PERSON_CONFIDENCE_THRESHOLD: float = 0.5
    MAX_RETRIES: int = 3
    
    # Database config
    DB_HOST: str = os.getenv("DB_HOST", "ballast.proxy.rlwy.net")
    DB_PORT: str = os.getenv("DB_PORT", "56193")
    DB_NAME: str = os.getenv("DB_NAME", "railway")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "AfldldzckDWtkskkAMEhMaDXnMqknaPY")
    
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
    """Handles all database operations"""
    
    @staticmethod
    def fetch_unprocessed_images(group_id: int, batch_size: int) -> List[Tuple[int, bytes]]:
        """Fetch unprocessed images with proper error handling"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, image_byte FROM images WHERE status = 'warm' AND group_id = %s LIMIT %s", 
                    (group_id, batch_size)
                )
                return cur.fetchall()

    @staticmethod
    def fetch_warm_groups() -> List[int]:
        """Fetch warm groups from database"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM groups WHERE status = 'warm'")
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def insert_faces_batch(records: List[dict], group_id: int) -> None:
        """Insert detected faces with transaction safety"""
        if not records:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                INSERT INTO faces (id, image_id, group_id, person_id, face_thumb_bytes)
                VALUES %s
                """
                values = [
                    (r['id'], r['image_id'], group_id, r['person_id'], r['face_thumb_bytes'])
                    for r in records
                ]
                execute_values(cur, query, values)
                conn.commit()
                logger.info(f"Inserted {len(records)} faces for group {group_id}")

    @staticmethod
    def mark_images_processed_batch(image_ids: List[int]) -> None:
        """Mark images as processed and clear image_byte"""
        if not image_ids:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = "UPDATE images SET status = 'warmed' WHERE id = ANY(%s::uuid[])"
                cur.execute(query, (image_ids,))
                conn.commit()
                logger.info(f"Marked {len(image_ids)} images as processed")

class HybridFaceIndexer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models with error handling
        try:
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0)
            
            self.fashion_model = AutoModel.from_pretrained(
                'Marqo/marqo-fashionCLIP', 
                trust_remote_code=True
            ).to(self.device)
            self.fashion_processor = AutoProcessor.from_pretrained(
                'Marqo/marqo-fashionCLIP', 
                trust_remote_code=True
            )
            
            self.qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def setup_collection(self, collection_name: str) -> None:
        """Setup Qdrant collection with proper error handling"""
        try:
            if not self.qdrant.collection_exists(collection_name):
                # self.qdrant.delete_collection(collection_name)

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

    def extract_clothing_embedding(self, image_input) -> torch.Tensor:
        """Extract clothing embeddings with proper error handling"""
        try:
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            else:
                img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

            inputs = self.fashion_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.fashion_model.get_image_features(**inputs)
                return emb[0] / emb[0].norm()
        except Exception as e:
            logger.error(f"Failed to extract clothing embedding: {e}")
            raise

    def image_to_bytes(self, cv_image: np.ndarray) -> bytes:
        """Convert OpenCV image to bytes with error handling"""
        try:
            success, buffer = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise ValueError("Could not encode image")
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            raise

    def process_image(self, image_id: int, image_bytes: bytes, yolo_model, collection_name: str) -> List[dict]:
        """Process single image with comprehensive error handling"""
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.warning(f"Failed to decode image {image_id}")
                return []

            # YOLO detection
            results = yolo_model(img)[0]
            records = []

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > config.PERSON_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_crop = img[y1:y2, x1:x2]
                    
                    if person_crop.size == 0:
                        continue
                    
                    # Extract faces
                    faces = self.face_app.get(person_crop)
                    if not faces:
                        continue
                    
                    # Extract clothing embedding
                    try:
                        clothing_emb = self.extract_clothing_embedding(person_crop)
                    except Exception as e:
                        logger.warning(f"Failed to extract clothing embedding for image {image_id}: {e}")
                        continue
                    
                    # Process each face
                    for face in faces:
                        try:
                            face_emb = face.normed_embedding
                            point_id = str(uuid.uuid4())
                            
                            # Extract face thumbnail
                            face_thumb_bytes = None
                            if len(faces) == 1:
                                x1_f, y1_f, x2_f, y2_f = map(int, face.bbox)
                                face_crop = person_crop[y1_f:y2_f, x1_f:x2_f]
                                if face_crop.size > 0:
                                    face_thumb_bytes = self.image_to_bytes(face_crop)
                            
                            # Insert into Qdrant
                            self.qdrant.upsert(
                                collection_name=collection_name,
                                points=[
                                    PointStruct(
                                        id=point_id,
                                        vector={
                                            "face": face_emb.tolist(),
                                            "cloth": clothing_emb.cpu().tolist()
                                        },
                                        payload={
                                            "person_id": None,
                                            "image_id": image_id,
                                            "cloth_ids":None
                                        }
                                    )
                                ]
                            )
                            
                            records.append({
                                "id": point_id,
                                "image_id": image_id,
                                "person_id": None,
                                "face_thumb_bytes": face_thumb_bytes
                            })
                            
                        except Exception as e:
                            logger.error(f"Failed to process face in image {image_id}: {e}")
                            continue

            logger.info(f"Processed image {image_id}: {len(records)} faces detected")
            return records
            
        except Exception as e:
            logger.error(f"Failed to process image {image_id}: {e}")
            return []

    def process_images_batch(self, images_batch: List[Tuple[int, bytes]], yolo_model, collection_name: str) -> List[dict]:
        """Process batch of images with controlled parallelism"""
        logger.info(f"Processing batch of {len(images_batch)} images")
        
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.PARALLEL_LIMIT) as executor:
            futures = {
                executor.submit(self.process_image, img_id, img_bytes, yolo_model, collection_name): img_id
                for img_id, img_bytes in images_batch
            }
            
            for future in concurrent.futures.as_completed(futures):
                img_id = futures[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    all_results.extend(result)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout processing image {img_id}")
                except Exception as e:
                    logger.error(f"Error processing image {img_id}: {e}")
        
        return all_results

def process_group(group_id: int, indexer: HybridFaceIndexer, yolo_model) -> None:
    """Process a single group with proper error handling and transaction management"""
    try:
        indexer.setup_collection(group_id)
        logger.info(f"Processing group {group_id}")
        
        processed_count = 0
        
        while True:
            # Fetch batch of unprocessed images
            unprocessed = DatabaseManager.fetch_unprocessed_images(group_id, config.BATCH_SIZE)
            
            if not unprocessed:
                logger.info(f"No more unprocessed images for group {group_id}")
                break
            
            logger.info(f"Found {len(unprocessed)} unprocessed images for group {group_id}")
            
            # Process the batch
            records = indexer.process_images_batch(unprocessed, yolo_model, group_id)
            
            # Extract processed image IDs (all images in batch are considered processed)
            processed_image_ids = [img_id for img_id, _ in unprocessed]
            
            # Insert faces and mark images as processed in transaction
            if records:
                DatabaseManager.insert_faces_batch(records, group_id)
            
            DatabaseManager.mark_images_processed_batch(processed_image_ids)
            
            processed_count += len(unprocessed)
            logger.info(f"Group {group_id}: Processed {processed_count} images so far, {len(records)} faces indexed")
            
            # If we got fewer images than batch size, we're done
            if len(unprocessed) < config.BATCH_SIZE:
                break
                
        logger.info(f"Completed processing group {group_id}: {processed_count} total images processed")
        
    except Exception as e:
        logger.error(f"Failed to process group {group_id}: {e}")
        raise

def main():
    """Main execution function with proper error handling"""
    try:
        # Initialize models
        logger.info("Initializing YOLO model...")
        yolo_model = YOLO("yolov8x.pt")
        
        logger.info("Initializing face indexer...")
        indexer = HybridFaceIndexer()
        
        # Fetch groups to process
        groups = DatabaseManager.fetch_warm_groups()
        logger.info(f"Found {len(groups)} warm groups to process")
        
        if not groups:
            logger.info("No warm groups found, exiting")
            return
        
        # Process each group
        for group_id in groups:
            try:
                process_group(group_id, indexer, yolo_model)
            except Exception as e:
                logger.error(f"Failed to process group {group_id}, continuing with next group: {e}")
                continue
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()