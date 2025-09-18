import os
import uuid
import cv2
import torch
import psycopg2
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator, Dict, Set
import logging
import requests
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from psycopg2.extras import execute_values
import concurrent.futures
import numpy as np
import math
import open_clip
import firebase_admin
from firebase_admin import credentials, storage
import time
from queue import Queue
import threading
from collections import defaultdict
import json
import argparse
import sys
class ProcessingError(Exception):
    def __init__(self, message, group_id=None, reason=None, retryable=True):
        super().__init__(message)
        self.group_id = group_id
        self.reason = reason
        self.retryable = retryable

    def __str__(self):
        return f"ProcessingError: {self.args[0]} (group_id={self.group_id}, reason={self.reason}, retryable={self.retryable})"
# Initialize Firebase once
try:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "gallery-585ee.firebasestorage.app"
    })
except Exception as e:
    logging.warning(f"Firebase initialization failed: {e}")

def calculate_overlap_ratio(box1, box2):
    """Calculate the overlap ratio between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate area of smaller box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    smaller_area = min(area1, area2)
    
    if smaller_area == 0:
        return 0.0
    
    return intersection_area / smaller_area

def expand_face_bbox_for_clothing(face_bbox, person_bbox, expansion_factor=2.0):
    """
    Expand face bounding box to include more body area for clothing detection
    """
    # Convert face bbox from person-relative to image coordinates
    face_x1, face_y1, face_x2, face_y2 = face_bbox
    person_x1, person_y1, person_x2, person_y2 = person_bbox
    
    # Convert to absolute coordinates
    abs_face_x1 = person_x1 + face_x1
    abs_face_y1 = person_y1 + face_y1
    abs_face_x2 = person_x1 + face_x2
    abs_face_y2 = person_y1 + face_y2
    
    # Calculate face dimensions
    face_width = abs_face_x2 - abs_face_x1
    face_height = abs_face_y2 - abs_face_y1
    
    # Expand downward and slightly outward to capture clothing
    expanded_x1 = max(person_x1, abs_face_x1 - face_width * 0.5)
    expanded_y1 = abs_face_y1  # Start from face top
    expanded_x2 = min(person_x2, abs_face_x2 + face_width * 0.5)
    expanded_y2 = min(person_y2, abs_face_y1 + face_height * expansion_factor)
    
    return [int(expanded_x1), int(expanded_y1), int(expanded_x2), int(expanded_y2)]

@dataclass
class Config:
    BATCH_SIZE: int = 25  # Increased batch size
    PARALLEL_LIMIT: int = 1  # Increased parallelism
    PERSON_CONFIDENCE_THRESHOLD: float = 0.5
    MAX_RETRIES: int = 3
    FACE_OVERLAP_THRESHOLD: float = 0.7
    QDRANT_BATCH_SIZE: int = 25  # Batch insert to Qdrant
    DB_BATCH_SIZE: int = 25  # Batch insert to DB
    QDRANT_MAX_WORKERS:int = 1
    
    # Connection configs
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    
    # DB Config
    DB_HOST: str = "ballast.proxy.rlwy.net"
    DB_PORT: str = "56193"
    DB_NAME: str = "railway"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "AfldldzckDWtkskkAMEhMaDXnMqknaPY"

config = Config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConnectionValidator:
    """Validates all connections before starting processing"""
    
    @staticmethod
    def validate_database() -> bool:
        """Test database connection"""
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                dbname=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD
            )
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.close()
            logger.info("✓ Database connection validated")
            return True
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            return False
    
    @staticmethod
    def validate_qdrant() -> bool:
        """Test Qdrant connection"""
        try:
            client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            client.get_collections()
            logger.info("✓ Qdrant connection validated")
            return True
        except Exception as e:
            logger.error(f"✗ Qdrant connection failed: {e}")
            return False
    
    @staticmethod
    def validate_firebase() -> bool:
        """Test Firebase connection"""
        try:
            bucket = storage.bucket()
            # Try to list a few blobs to test connection
            blobs = list(bucket.list_blobs(max_results=1))
            logger.info("✓ Firebase connection validated")
            return True
        except Exception as e:
            logger.error(f"✗ Firebase connection failed: {e}")
            return False
    
    @staticmethod
    def validate_all() -> bool:
        """Validate all connections"""
        logger.info("Validating connections...")
        db_ok = ConnectionValidator.validate_database()
        # qdrant_ok = ConnectionValidator.validate_qdrant()
        firebase_ok = ConnectionValidator.validate_firebase()
        
        if db_ok and firebase_ok:
            logger.info("✓ All connections validated successfully")
            return True
        else:
            logger.error("✗ Some connections failed validation")
            return False

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper cleanup"""
    conn = None
    try:
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD
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
    """Handles all database operations with improved batching"""
    
    @staticmethod
    def fetch_unprocessed_images(group_id: int, batch_size: int):
        """Fetch unprocessed images with proper error handling"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, location FROM images WHERE status = 'warm' AND group_id = %s LIMIT %s", 
                        (group_id, batch_size)
                    )
                    return {"success": True, "data": cur.fetchall()}
        except Exception as e:
            logger.error(f"❌ Error in fetch_unprocessed_images: {e}")
            return {"success": False, "error": str(e)}
    @staticmethod
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
                        WHERE task = 'extraction'
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
                            WHERE task = 'extraction'
                            """,
                            (next_group_in_queue,)
                        )
                        conn.commit()
                        return next_group_in_queue

                    return None
        except Exception as e:
            print("❌ Error in get_or_assign_group_id:", e)
            return None

    @staticmethod
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
                            WHERE task = 'extraction'
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
                            WHERE task = 'extraction'
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
                        WHERE task = 'extraction'
                        """,
                        (group_id,)
                    )
                    cur.execute(
                        """
                        UPDATE process_status
                        SET next_group_in_queue = %s
                        WHERE task = 'quality_assignment' and next_group_in_queue is null  
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

    @staticmethod
    def load_processed_images_from_json(group_id: int) -> Set[str]:
        """
        Load already processed image IDs from faces.json file
        Returns set of processed image_ids for quick lookup
        """
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "warm-images", str(group_id), "faces", "faces.json")
            
            if not os.path.exists(json_path):
                logger.info(f"No existing faces.json found for group {group_id}")
                return set()
            
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        logger.warning(f"Invalid faces.json format for group {group_id}")
                        return set()
                    
                    # Extract unique image_ids from the records
                    processed_image_ids = set()
                    for record in data:
                        if isinstance(record, dict) and "image_id" in record:
                            processed_image_ids.add(str(record["image_id"]))
                    
                    logger.info(f"Found {len(processed_image_ids)} already processed images in faces.json for group {group_id}")
                    return processed_image_ids
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse faces.json for group {group_id}: {e}")
                    return set()
                    
        except Exception as e:
            logger.warning(f"Failed to load faces.json for group {group_id}: {e}")
            return set()

    @staticmethod
    def filter_already_processed_images(images_batch: List[Tuple[int, str]], processed_image_ids: Set[str]) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str, Optional[str]]]]:
        """
        Filter out already processed images and return them as 'warming' status
        Returns: (unprocessed_images, already_processed_statuses)
        """
        unprocessed = []
        already_processed_statuses = []
        
        for img_id, location in images_batch:
            if str(img_id) in processed_image_ids:
                already_processed_statuses.append((img_id, "warming", None))
                logger.debug(f"Skipping already processed image {img_id}")
            else:
                unprocessed.append((img_id, location))
        
        if already_processed_statuses:
            logger.info(f"Skipped {len(already_processed_statuses)} already processed images")
        
        return unprocessed, already_processed_statuses

    @staticmethod
    def mark_group_processed(group_id) -> None:
        """Mark group_id as processed"""
        if not group_id:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    UPDATE groups
                    SET status = 'warming',
                        last_processed_at = NOW(),
                        last_processed_step = 'extraction'
                    WHERE id = %s
                """
                cur.execute(query, (group_id,))
                conn.commit()
                logger.info(f"Marked group {group_id} as processed")
    
    @staticmethod
    def mark_image_status(image_id: int, status: str, error_message: str = None) -> None:
        """Mark single image with specific status and error message"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if error_message:
                    query = """
                        UPDATE images 
                        SET status = %s, error_message = %s, last_processed_at = NOW()
                        WHERE id = %s
                    """
                    cur.execute(query, (status, error_message, image_id))
                else:
                    query = """
                        UPDATE images 
                        SET status = %s, last_processed_at = NOW()
                        WHERE id = %s
                    """
                    cur.execute(query, (status, image_id))
                conn.commit()
                logger.info(f"Marked image {image_id} as {status}")

    @staticmethod
    def mark_images_status_batch(image_statuses: List[Tuple[int, str, Optional[str]]]) -> None:
        """Mark multiple images with their respective statuses and error messages"""
        if not image_statuses:
            return
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Separate images with and without error messages
                    with_errors = [(img_id, status, error_msg) for img_id, status, error_msg in image_statuses if error_msg]
                    without_errors = [(img_id, status) for img_id, status, error_msg in image_statuses if not error_msg]
                    
                    # Update images with error messages
                    if with_errors:
                        query_with_error = """
                            UPDATE images 
                            SET status = data.status, error_message = data.error_message, last_processed_at = NOW()
                            FROM (VALUES %s) AS data(id, status, error_message)
                            WHERE images.id = data.id::uuid
                        """
                        execute_values(cur, query_with_error, with_errors)
                    
                    # Update images without error messages
                    if without_errors:
                        query_without_error = """
                            UPDATE images 
                            SET status = data.status, last_processed_at = NOW()
                            FROM (VALUES %s) AS data(id, status)
                            WHERE images.id = data.id::uuid
                        """
                        execute_values(cur, query_without_error, without_errors)
                    
                    conn.commit()
                    logger.info(f"Updated status for {len(image_statuses)} images")
        except Exception as e:
            logger.error(f"Database error in mark_images_status_batch: {e}")
            raise ProcessingError(f"Cannot update images status in db: {str(e)}", retryable=False)

    @staticmethod
    def save_face_image(face_record: dict, group_id: int):
        """Save only the thumbnail image to disk"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            group_folder = os.path.join(script_dir,"warm-images", str(group_id),"faces")
            os.makedirs(group_folder, exist_ok=True)

            if face_record.get("face_thumb_bytes"):
                image_path = os.path.join(group_folder, f"{face_record['id']}.jpg")
                with open(image_path, "wb") as f:
                    f.write(face_record["face_thumb_bytes"])
        except Exception as e:
            logger.error(f"Failed to save face image for group {group_id}: {e}")
    
    @staticmethod
    def save_faces_json_batch(face_records: list[dict], group_id: int):
        """Append a batch of face metadata (without bytes) to faces.json"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            group_folder = os.path.join(script_dir,"warm-images", str(group_id),"faces")
            os.makedirs(group_folder, exist_ok=True)

            json_path = os.path.join(group_folder, "faces.json")

            # Load existing
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = []
                    except Exception:
                        existing_data = []
            else:
                existing_data = []

            # Convert batch (remove raw bytes)
            new_records = [
                {k: v for k, v in r.items() if k != "face_thumb_bytes"}
                for r in face_records
            ]

            # Merge and save once
            all_data = existing_data + new_records
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update faces.json for group {group_id}: {e}")
    
    @staticmethod
    def insert_faces_batch(records: List[dict], group_id: int) -> None:
        """Insert detected faces in batches for better performance"""
        if not records:
            return
        for r in records:
            DatabaseManager.save_face_image(r, group_id)

        DatabaseManager.save_faces_json_batch(records, group_id)

class OptimizedFaceIndexer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Create connection pool for Qdrant
        # self.qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        
        # Cache for Firebase bucket
        self.firebase_bucket = storage.bucket()
        
    def _initialize_models(self):
        """Initialize all models with error handling"""
        try:
            logger.info("Loading InsightFace model...")
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0)
            
            logger.info("Loading CLIP model...")
            try:
                model_name = "ViT-B-32"
                pretrained = "laion2b_s34b_b79k"
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                    device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
            except Exception as meta_error:
                logger.warning(f"OpenCLIP error, trying FashionCLIP: {meta_error}")
                self.fashion_model = AutoModel.from_pretrained(
                    'Marqo/marqo-fashionCLIP', 
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                if str(self.device) != "cpu":
                    self.fashion_model = self.fashion_model.to(self.device)
            
            self.fashion_processor = AutoProcessor.from_pretrained(
                'Marqo/marqo-fashionCLIP', 
                trust_remote_code=True
            )
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def setup_collection(self, collection_name: str) -> None:
        """Setup Qdrant collection with proper error handling"""
        try:
            if not self.qdrant.collection_exists(collection_name):
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "face": VectorParams(size=512, distance=Distance.COSINE),
                        "cloth": VectorParams(size=512, distance=Distance.COSINE)
                    }
                )
            logger.info(f"Collection {collection_name} ready")
        except Exception as e:
            logger.error(f"Failed to setup collection {collection_name}: {e}")
            raise

    def extract_clothing_embedding(self, image_input) -> torch.Tensor:
        """Extract clothing embeddings with proper error handling"""
        try:
            img_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb[0]
        except Exception as e:
            logger.error(f"Failed to extract clothing embedding: {e}")
            raise

    def image_to_bytes(self, cv_image: np.ndarray, target_height: int = 150) -> bytes:
        """Convert OpenCV image to bytes with consistent sizing"""
        try:
            if cv_image is None or cv_image.size == 0:
                raise ValueError("Empty or invalid image")
            
            original_height, original_width = cv_image.shape[:2]
            aspect_ratio = original_width / original_height
            new_width = int(target_height * aspect_ratio)
            
            resized_image = cv2.resize(cv_image, (new_width, target_height), interpolation=cv2.INTER_AREA)
            
            success, buffer = cv2.imencode('.jpg', resized_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise ValueError("Could not encode image")
            
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            raise

    def read_image(self, group_id, path: str):
        """Read image from /warm-images cache, fallback to Firebase"""
        try:
            # Local path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(script_dir, "warm-images", f"{group_id}", f"compressed_{path}.jpg")

            if os.path.exists(local_path):
                # ✅ Read from local cache
                img = cv2.imread(local_path, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
                else:
                    logger.warning(f"Corrupted local file {local_path}, refetching from Firebase")

            # [WARNING] Not in local cache (or corrupted), fetch from Firebase
            blob = self.firebase_bucket.blob("compressed_" + path)
            if not blob.exists():
                raise FileNotFoundError(f"Image not found in Firebase: compressed_{path}")
            
            img_bytes = blob.download_as_bytes()
            image_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(f"Could not decode image: compressed_{path}")

            return img

        except FileNotFoundError as e:
            logger.error(f"File not found for image {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read image {path}: {e}")
            raise
        
    def delete_local_image(self, group_id, path: str):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(script_dir, "warm-images", f"{group_id}", f"compressed_{path}.jpg")
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.debug(f"🗑️ Deleted {local_path}")
        except Exception as e:
            logger.warning(f"Failed to delete {local_path}: {e}")

    def deduplicate_faces(self, faces_with_bboxes):
        """Remove duplicate faces based on overlap ratio"""
        if len(faces_with_bboxes) <= 1:
            return faces_with_bboxes
        
        faces_sorted = sorted(faces_with_bboxes, 
                             key=lambda x: getattr(x[0], 'det_score', 0.0), 
                             reverse=True)
        
        unique_faces = []
        
        for face, bbox in faces_sorted:
            is_duplicate = False
            
            for unique_face, unique_bbox in unique_faces:
                overlap = calculate_overlap_ratio(bbox, unique_bbox)
                if overlap > config.FACE_OVERLAP_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append((face, bbox))
        
        return unique_faces

    def process_single_image(self, group_id, image_id: int, location: str, yolo_model) -> Tuple[List[dict], str, Optional[str]]:
        """
        Optimized single image processing with detailed error handling
        Returns: (records, status, error_message)
        """
        try:
            # Load image
            try:
                img = self.read_image(group_id, image_id)
                if img is None:
                    return [], "extraction_failed", "Could not decode image"
            except FileNotFoundError:
                return [], "extraction_failed", "File not found"
            except Exception as e:
                return [], "extraction_failed", f"Image read error: {str(e)}"

            # Detect faces
            try:
                all_faces = self.face_app.get(img)
                if not all_faces:
                    return [], "no_face", "No faces detected"
            except Exception as e:
                logger.error(f"Face detection failed for image {image_id}: {e}")
                return [], "extraction_failed", f"Face detection error: {str(e)}"

            # YOLO person detection
            try:
                results = yolo_model(img)[0]
                person_boxes = []
                
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == 0 and conf > config.PERSON_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_boxes.append([x1, y1, x2, y2])

                if not person_boxes:
                    return [], "no_face", "No persons detected"
            except Exception as e:
                logger.error(f"YOLO detection failed for image {image_id}: {e}")
                return [], "extraction_failed", f"Person detection error: {str(e)}"

            # Process faces
            try:
                records = []
                faces_with_bboxes = [(face, face.bbox) for face in all_faces]
                unique_faces = self.deduplicate_faces(faces_with_bboxes)
                
                for face, face_bbox in unique_faces:
                    record = self._process_single_face(face, face_bbox, person_boxes, img, image_id)
                    if record:
                        records.append(record)
                
                if not records:
                    return [], "no_face", "No valid faces found after processing"
                    
                return records, "warming", None
                
            except Exception as e:
                logger.error(f"Face processing failed for image {image_id}: {e}")
                return [], "extraction_failed", f"Face processing error: {str(e)}"
            
        except Exception as e:
            logger.error(f"Unexpected error processing image {image_id}: {e}")
            return [], "extraction_failed", f"Unexpected error: {str(e)}"

    def _process_single_face(self, face, face_bbox, person_boxes, img, image_id) -> Optional[dict]:
        """Process a single face within an image"""
        try:
            face_x1, face_y1, face_x2, face_y2 = map(int, face_bbox)
            face_center_x = (face_x1 + face_x2) // 2
            face_center_y = (face_y1 + face_y2) // 2
            
            # Find containing person box
            best_person_box = None
            for person_box in person_boxes:
                px1, py1, px2, py2 = person_box
                if px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2:
                    best_person_box = person_box
                    break
            
            if best_person_box is None:
                return None
            
            # Extract clothing region
            clothing_bbox = expand_face_bbox_for_clothing(
                [face_x1 - best_person_box[0], face_y1 - best_person_box[1], 
                 face_x2 - best_person_box[0], face_y2 - best_person_box[1]], 
                best_person_box
            )
            
            # Validate clothing bbox
            h, w = img.shape[:2]
            clothing_bbox = [max(0, clothing_bbox[0]), max(0, clothing_bbox[1]),
                           min(w, clothing_bbox[2]), min(h, clothing_bbox[3])]
            
            if clothing_bbox[2] <= clothing_bbox[0] or clothing_bbox[3] <= clothing_bbox[1]:
                return None
            
            clothing_crop = img[clothing_bbox[1]:clothing_bbox[3], clothing_bbox[0]:clothing_bbox[2]]
            if clothing_crop.size == 0:
                return None
            
            # Extract embeddings
            pil_img = Image.fromarray(cv2.cvtColor(clothing_crop, cv2.COLOR_BGR2RGB))
            clothing_emb = self.extract_clothing_embedding(pil_img)
            face_emb = face.normed_embedding
            
            # Generate thumbnail
            face_thumb_bytes = self._extract_face_thumbnail(img, face_x1, face_y1, face_x2, face_y2)
            
            # Get confidence
            insight_face_confidence = float(face.det_score) if hasattr(face, 'det_score') and face.det_score is not None else 0.0
            
            return {
                "id": str(uuid.uuid4()),
                "image_id": image_id,
                "person_id": None,
                "face_thumb_bytes": face_thumb_bytes,
                "quality_score": -1,
                "insight_face_confidence": insight_face_confidence,
                "face_embedding": face_emb.tolist(),
                "clothing_embedding": clothing_emb.cpu().tolist(),
                "status":"p"
            }
            
        except Exception as e:
            logger.error(f"Failed to process face in image {image_id}: {e}")
            return None

    def _extract_face_thumbnail(self, img, face_x1, face_y1, face_x2, face_y2):
        """Extract face thumbnail with padding"""
        try:
            h, w = img.shape[:2]
            pad_x = int((face_x2 - face_x1) * 0.4)
            pad_y = int((face_y2 - face_y1) * 0.4)
            
            padded_x1 = max(0, face_x1 - pad_x)
            padded_y1 = max(0, face_y1 - pad_y)
            padded_x2 = min(w, face_x2 + pad_x)
            padded_y2 = min(h, face_y2 + pad_y)
            
            if padded_x2 > padded_x1 and padded_y2 > padded_y1:
                face_crop = img[padded_y1:padded_y2, padded_x1:padded_x2]
                if face_crop.size > 0:
                    return self.image_to_bytes(face_crop)
        except Exception:
            pass
        return None

    def batch_upsert_to_qdrant(self, records: List[dict], collection_name: str) -> None:
        """Batch insert records to Qdrant in parallel for better performance"""
        if not records:
            return

        batch_size = config.QDRANT_BATCH_SIZE
        max_workers = config.QDRANT_MAX_WORKERS if hasattr(config, "QDRANT_MAX_WORKERS") else 8

        def upsert_batch(batch):
            points = [
                PointStruct(
                    id=record["id"],
                    vector={
                        "face": record["face_embedding"],
                        "cloth": record["clothing_embedding"]
                    },
                    payload={
                        "person_id": record["person_id"],
                        "image_id": record["image_id"],
                        "cloth_ids": None,
                    }
                )
                for record in batch
            ]
            try:
                self.qdrant.upsert(collection_name=collection_name, points=points)
            except Exception as e:
                logger.error(f"Failed to upsert batch to Qdrant: {e}")
                # fallback: try individual points
                for point in points:
                    try:
                        self.qdrant.upsert(collection_name=collection_name, points=[point])
                    except Exception as inner_e:
                        logger.error(f"Failed to upsert single point: {inner_e}")
                        raise

        # Split records into batches
        batches = [records[i:i+batch_size] for i in range(0, len(records), batch_size)]

        # Run parallel uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(upsert_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Parallel upsert task failed: {e}")

    def process_images_batch(self, group_id, images_batch: List[Tuple[int, str]], yolo_model) -> Tuple[List[dict], List[Tuple[int, str, Optional[str]]]]:
        """
        Process batch of images with optimized parallelism
        Returns: (all_records, image_statuses)
        """
        logger.info(f"Processing batch of {len(images_batch)} images")
        
        all_records = []
        image_statuses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.PARALLEL_LIMIT) as executor:
            futures = {
                executor.submit(self.process_single_image, group_id, img_id, location, yolo_model): (img_id, location)
                for img_id, location in images_batch
            }
           
            for future in concurrent.futures.as_completed(futures):
                img_id, location = futures[future]
                try:
                    records, status, error_message = future.result(timeout=120)  # Increased timeout
                    all_records.extend(records)
                    image_statuses.append((img_id, status, error_message))
                    
                    if status == "warming":
                        logger.info(f"✓ Image {img_id}: {len(records)} faces extracted")
                    elif status == "no_face":
                        logger.info(f"⚠ Image {img_id}: No faces detected")
                    elif status == "extraction_failed":
                        logger.warning(f"✗ Image {img_id}: Extraction failed - {error_message}")
                        
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout processing image {img_id}")
                    image_statuses.append((img_id, "extraction_failed", "Processing timeout"))
                except Exception as e:
                    logger.error(f"Error processing image {img_id}: {e}")
                    image_statuses.append((img_id, "extraction_failed", f"Unexpected error: {str(e)}"))
        
        return all_records, image_statuses

def process_group_optimized(group_id: int, indexer: OptimizedFaceIndexer, yolo_model , run_id) -> None:
    """Optimized group processing with batched operations and JSON check"""
    try:
        # indexer.setup_collection(str(group_id))
        logger.info(f"Processing group {group_id}")
        
        # Load already processed image IDs from JSON once at the beginning
        processed_image_ids = DatabaseManager.load_processed_images_from_json(group_id)
        
        processed_count = 0
        total_faces = 0
        skipped_count = 0
        all_success_images = []
        all_failed_images = []
        while True:
            # Fetch batch of images
            fetchUnprocessedResponse = DatabaseManager.fetch_unprocessed_images(group_id, config.BATCH_SIZE)
            if(not fetchUnprocessedResponse["success"]):
                raise ProcessingError(
                    f"Failed to fetch unprocessed images: {fetchUnprocessedResponse['error']}", 
                    group_id=group_id, 
                    reason="database_fetch_error", 
                    retryable=False
                )            
                
            unprocessed = fetchUnprocessedResponse["data"]
            if not unprocessed:
                logger.info(f"No more unprocessed images for group {group_id}")
                DatabaseManager.update_status_history(run_id , "extraction" , "group" , processed_count , len(all_failed_images) , len(all_success_images) , group_id , "error Failed for "+", \n".join(all_failed_images))
                break
            
            logger.info(f"Found {len(unprocessed)} unprocessed images for group {group_id}")
            
            start_time = time.time()
            
            # Step 0: Filter out already processed images from JSON
            logger.info("Step 0: Checking against already processed images...")
            images_to_process, already_processed_statuses = DatabaseManager.filter_already_processed_images(
                unprocessed, processed_image_ids
            )
            
            # Update skipped count
            skipped_count += len(already_processed_statuses)
            
            # If all images were already processed, just update their status and continue
            if not images_to_process:
                if already_processed_statuses:
                    logger.info("All images in this batch were already processed, updating database status...")
                    try:
                        DatabaseManager.mark_images_status_batch(already_processed_statuses)
                        processed_count += len(already_processed_statuses)
                    except ProcessingError:
                        # Re-raise ProcessingError as-is
                        raise
                continue
            
            logger.info(f"Processing {len(images_to_process)} new images, skipping {len(already_processed_statuses)} already processed")
            
            # Step 1: Process only the unprocessed images (extract embeddings)
            logger.info("Step 1: Processing new images and extracting embeddings...")
            try:
                all_records, image_statuses = indexer.process_images_batch(group_id, images_to_process, yolo_model)
            except Exception as e:
                # Convert any processing errors to ProcessingError
                raise ProcessingError(
                    f"Image batch processing failed: {str(e)}", 
                    group_id=group_id, 
                    reason="image_processing_error", 
                    retryable=True
                )
            # Combine statuses from already processed and newly processed images
            combined_statuses = already_processed_statuses + image_statuses
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f}s")
            
            # Step 2: Update image statuses in batch
            logger.info("Step 2: Updating image statuses...")
            try:
                DatabaseManager.mark_images_status_batch(combined_statuses)
            except ProcessingError:
                # Re-raise ProcessingError as-is
                raise
            if all_records:
                # Step 3: Batch insert to Qdrant
                logger.info("Step 3: Inserting to Qdrant...")
                qdrant_start = time.time()
                # try:
                #     indexer.batch_upsert_to_qdrant(all_records, str(group_id))
                # except Exception as e:
                #     raise ProcessingError(
                #         f"Qdrant insertion failed: {str(e)}", 
                #         group_id=group_id, 
                #         reason="qdrant_error", 
                #         retryable=True
                #     )
                qdrant_time = time.time() - qdrant_start
                logger.info(f"Qdrant insert completed in {qdrant_time:.2f}s")
                
                # Step 4: Batch insert to database
                logger.info("Step 4: Inserting to database...")
                db_start = time.time()
                # Remove embeddings from records before DB insert (they're already in Qdrant)
                try:
                    # Remove embeddings from records before DB insert
                    db_records = []
                    for record in all_records:
                        db_record = {k: v for k, v in record.items() 
                                if k not in ['face_embedding', 'clothing_embedding']}
                        db_records.append(db_record)
                    
                    DatabaseManager.insert_faces_batch(db_records, group_id)
                except Exception as e:
                    raise ProcessingError(
                        f"Database face insertion failed: {str(e)}", 
                        group_id=group_id, 
                        reason="face_db_insertion_error", 
                        retryable=False
                    )
                db_time = time.time() - db_start
                logger.info(f"Database insert completed in {db_time:.2f}s")
                
                # Step 5: Update processed_image_ids set for next iteration
                newly_processed_image_ids = set(str(record["image_id"]) for record in all_records)
                processed_image_ids.update(newly_processed_image_ids)
                
                total_faces += len(all_records)
            
            processed_count += len(unprocessed)  # Count all images in batch
            total_time = time.time() - start_time
            
            # Log statistics
            successful_images = [s[0] for s in image_statuses if s[1] == "warming"]
            all_success_images.extend(successful_images)
            failed_images = [s[0] for s in image_statuses if s[1] == "extraction_failed"]
            all_failed_images.extend(failed_images)
            no_face_images = [s for s in image_statuses if s[1] == "no_face"]
            all_success_images.extend(no_face_images)
            logger.info(f"Group {group_id}: Processed {processed_count} images so far")
            logger.info(f"  ✓ Successful: {len(successful_images)}, ✗ Failed: {len(failed_images)}, ⚠ No faces: {len(no_face_images)}, 🔄 Skipped: {len(already_processed_statuses)}")
            logger.info(f"  {total_faces} faces indexed total in {total_time:.2f}s")
            
            if images_to_process:  # Only calculate performance for actually processed images
                logger.info(f"Performance: {len(images_to_process)/processing_time:.2f} images/sec, "
                           f"{len(all_records)/processing_time:.2f} faces/sec")
            
            if len(unprocessed) < config.BATCH_SIZE:
                DatabaseManager.update_status_history(run_id , "extraction" , "group" , processed_count , len(all_failed_images) , len(all_success_images) , group_id , "done Extraction Failed for "+", \n".join(all_failed_images))
                break
        
        logger.info(f"Completed processing group {group_id}: {processed_count} total images processed, {skipped_count} images skipped (already processed), {total_faces} faces indexed")
        
    except ProcessingError:
        # Re-raise ProcessingError as-is to preserve error details
        raise
    except Exception as e:
        # Convert any unexpected errors to ProcessingError
        logger.error(f"Unexpected error in process_group_optimized: {e}", exc_info=True)
        raise ProcessingError(
            f"Group processing failed with unexpected error: {str(e)}", 
            group_id=group_id, 
            reason="unexpected_error", 
            retryable=False
        )

def main_optimized():
    """Optimized main execution with connection validation and model pre-loading"""
    run_id = int(time.time())
    
    # Step 1: Validate all connections
    logger.info("=== STEP 1: VALIDATING CONNECTIONS ===")
    if not ConnectionValidator.validate_all():
        DatabaseManager.update_status(None , "error Validation Failed",True , "failed" )
        DatabaseManager.update_status_history(run_id , "extraction" , "run" , None , None , None , None , "error Validation Failed")
        logger.error("Connection validation failed. Exiting.")
        return False
    
    # Step 2: Check if group exists and has work to do
    logger.info("=== STEP 2: VALIDATING GROUP ===")
    group_id = DatabaseManager.get_or_assign_group_id()
    if not group_id:
        logger.error(f"Group {group_id} not found or not in 'warm' status")
        DatabaseManager.update_status(None , "No Group Found To Process",True , "waiting" )
        DatabaseManager.update_status_history(run_id , "extraction" , "run" , None , None , None , None , "no_group")
        return False
    DatabaseManager.update_status(group_id , "Running",False , "healthy" )
    DatabaseManager.update_status_history(run_id , "extraction" , "run" , None , None , None , group_id , "started")
    # Check if there are images to process
    unprocessed_count = len(DatabaseManager.fetch_unprocessed_images(group_id, 1))
    if unprocessed_count == 0:
        logger.info(f"No warm images found for group {group_id}, exiting")
        DatabaseManager.update_status(group_id , "No Images Found To Process",True , "failed" )
        DatabaseManager.update_status_history(run_id , "extraction" , "run" , None , None , None , group_id , "error Validation Failed")
        return False
 

    logger.info(f"Group {group_id} validated and has images to process")
    
    # Step 3: Load models (expensive operation, do once)
    logger.info("=== STEP 3: LOADING MODELS ===")
    start_model_load = time.time()
    
    logger.info("Loading YOLO model...")
    yolo_model = YOLO("yolov8x.pt")
    
    logger.info("Loading face indexer...")
    indexer = OptimizedFaceIndexer()
    
    model_load_time = time.time() - start_model_load
    logger.info(f"All models loaded in {model_load_time:.2f}s")
    
    # Step 4: Process the group
    logger.info("=== STEP 4: PROCESSING GROUP ===")
    
    try:
        group_start = time.time()
        
        # Process the group
        process_group_optimized(group_id, indexer, yolo_model,run_id)
        
        # Mark group as completed
        DatabaseManager.mark_group_processed(group_id)
        DatabaseManager.update_status(None , "Waiting",True , "done" )
        DatabaseManager.update_status_history(run_id , "extraction" , "run" , None , None , None , group_id , "done")
        DatabaseManager.update_last_provrssed_group_column(group_id)
        group_time = time.time() - group_start
        logger.info(f"Group {group_id} completed in {group_time:.2f}s")
    except ProcessingError as e:
        msg = f"error in group {group_id}: {e}"
        logger.error(msg)
        DatabaseManager.update_status(group_id, msg, True, "failed")
        DatabaseManager.update_status_history(
            run_id, "extraction", "run", None, None, None, group_id, msg
        )
        return False
    except Exception as e:
        msg = f"error while processing group {group_id}: {e}"
        logger.error(msg, exc_info=True)  # logs full traceback
        DatabaseManager.update_status(group_id, msg, True, "failed")
        DatabaseManager.update_status_history(
            run_id, "extraction", "run", None, None, None, group_id, msg
        )
        return False
    
    logger.info("=== PROCESSING COMPLETED SUCCESSFULLY ===")
    return True
def main_with_monitoring():
    """Main function with performance monitoring and command line arguments"""
    # Parse command line arguments
    
    total_start = time.time()
    
    try:
        success = main_optimized()
        total_time = time.time() - total_start
        
        if success:
            logger.info(f"Processing completed successfully for group in {total_time:.2f}s")
        else:
            logger.error(f"[WARNING] Processing failed for group after {total_time:.2f}s")
            
        return success
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return False
    except Exception as e:
        total_time = time.time() - total_start
        logger.error(f"Critical error after {total_time:.2f}s: {e}")
        return False

if __name__ == "__main__":
    success = main_with_monitoring()
    exit(0 if success else 1)