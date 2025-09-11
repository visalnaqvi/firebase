import os
import uuid
import cv2
import torch
import psycopg2
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator, Dict
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
def _normalize(value, min_val, max_val):
    if max_val <= min_val:
        return 0.0
    return float(min(max((value - min_val) / (max_val - min_val), 0.0), 1.0))

# Initialize Firebase once
try:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "gallery-585ee.firebasestorage.app"
    })
except Exception as e:
    logging.warning(f"Firebase initialization failed: {e}")

def estimate_frontalness_from_landmarks(face, face_crop):
    """
    Try to estimate frontalness using landmarks if explicit yaw/pitch not available.
    Returns a score 0..1 where 1 is frontal.
    """
    kps = None
    if hasattr(face, "kps") and face.kps is not None:
        kps = np.array(face.kps)
    elif hasattr(face, "landmark") and face.landmark is not None:
        kps = np.array(face.landmark)
    if kps is None or kps.size == 0:
        return 1.0

    try:
        if kps.shape[0] >= 5:
            left_eye = kps[0]
            right_eye = kps[1]
        else:
            xs = kps[:, 0]
            left_eye = kps[np.argmin(xs)]
            right_eye = kps[np.argmax(xs)]

        dx = abs(right_eye[0] - left_eye[0])
        width = face_crop.shape[1] if face_crop is not None else dx
        ratio = dx / (width + 1e-6)
        return _normalize(ratio, 0.18, 0.45)
    except Exception:
        return 1.0

def compute_face_quality(face_crop: np.ndarray, face) -> float:
    """
    Compute a 0..1 quality score for the crop and the face object returned by insightface.
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0

    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_norm = _normalize(sharpness, 50, 1500)
    except Exception:
        sharp_norm = 0.0

    det_conf = 0.0
    if hasattr(face, "det_score") and face.det_score is not None:
        try:
            det_conf = float(face.det_score)
            det_conf = _normalize(det_conf, 0.3, 1.0)
        except Exception:
            det_conf = 0.0

    pose_score = 1.0
    if hasattr(face, "yaw") or hasattr(face, "pose"):
        try:
            yaw = getattr(face, "yaw", 0.0)
            pitch = getattr(face, "pitch", 0.0)
            yaw = float(yaw) if yaw is not None else 0.0
            pitch = float(pitch) if pitch is not None else 0.0
            yaw_score = max(0.0, 1.0 - abs(yaw) / 40.0)
            pitch_score = max(0.0, 1.0 - abs(pitch) / 40.0)
            pose_score = (yaw_score + pitch_score) / 2.0
        except Exception:
            pose_score = 1.0
    else:
        try:
            pose_score = estimate_frontalness_from_landmarks(face, face_crop)
        except Exception:
            pose_score = 1.0

    completeness = 1.0
    try:
        if hasattr(face, "bbox") and face.bbox is not None:
            x1, y1, x2, y2 = map(int, face.bbox)
            h, w = face_crop.shape[:2]
            margin_x = min(x1, w - x2)
            margin_y = min(y1, h - y2)
            min_margin = min(margin_x, margin_y)
            completeness = _normalize(min_margin, 0, 10)
        else:
            completeness = 1.0
    except Exception:
        completeness = 1.0

    score = 0.45 * sharp_norm + 0.25 * pose_score + 0.15 * det_conf + 0.15 * completeness
    return float(min(max(score, 0.0), 1.0))

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
        qdrant_ok = ConnectionValidator.validate_qdrant()
        firebase_ok = ConnectionValidator.validate_firebase()
        
        if db_ok and qdrant_ok and firebase_ok:
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
    def fetch_unprocessed_images(group_id: int, batch_size: int) -> List[Tuple[int, str]]:
        """Fetch unprocessed images with proper error handling"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, location FROM images WHERE status = 'warm' AND group_id = %s LIMIT %s", 
                    (group_id, batch_size)
                )
                return cur.fetchall()

    @staticmethod
    def fetch_warm_groups() -> List[int]:
        """Fetch warm groups from database"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM groups WHERE status = 'warm' order by last_processed_at limit 1")
                return [row[0] for row in cur.fetchall()]

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
                    WHERE id = %s AND status = 'warm'
                """
                cur.execute(query, (group_id,))
                conn.commit()
                logger.info(f"Marked group {group_id} as processed")

    @staticmethod
    def mark_group_process_status(group_id) -> None:
        """Mark group_id as being processed"""
        if not group_id:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    UPDATE process_status
                    SET group_id = %s
                    WHERE status = 'extraction'
                """
                cur.execute(query, (group_id,))
                conn.commit()
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
        # Process in smaller batches to avoid memory issues
        # batch_size = config.DB_BATCH_SIZE
        
        # with get_db_connection() as conn:
        #     with conn.cursor() as cur:
        #         for i in range(0, len(records), batch_size):
        #             batch = records[i:i+batch_size]
        #             query = """
        #             INSERT INTO faces (id, image_id, group_id, person_id, face_thumb_bytes, quality_score, insight_face_confidence)
        #             VALUES %s
        #             """
        #             values = [
        #                 (r['id'], r['image_id'], group_id, r['person_id'], r['face_thumb_bytes'], r['quality_score'], r['insight_face_confidence'])
        #                 for r in batch
        #             ]
        #             execute_values(cur, query, values)
        #         conn.commit()
        #         logger.info(f"Inserted {len(records)} faces for group {group_id}")

    @staticmethod
    def mark_images_processed_batch(image_ids: List[int]) -> None:
        """Mark images as processed in batch"""
        if not image_ids:
            return
        
        # Process in smaller batches
        batch_size = config.DB_BATCH_SIZE
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(image_ids), batch_size):
                    batch = image_ids[i:i+batch_size]
                    query = "UPDATE images SET status = 'warming' WHERE id = ANY(%s::uuid[])"
                    cur.execute(query, (batch,))
                conn.commit()
                logger.info(f"Marked {len(image_ids)} images as processed")

class OptimizedFaceIndexer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Create connection pool for Qdrant
        self.qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        
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

    def read_image(self,group_id, path: str):
        """Read image from /warm-images cache, fallback to Firebase"""
        try:
            # Local path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(script_dir, "warm-images", f"{group_id}" , f"compressed_{path}.jpg")

            if os.path.exists(local_path):
                # ✅ Read from local cache
                img = cv2.imread(local_path, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
                else:
                    logger.warning(f"Corrupted local file {local_path}, refetching from Firebase")

            # ❌ Not in local cache (or corrupted), fetch from Firebase
            blob = self.firebase_bucket.blob("compressed_" + path)
            img_bytes = blob.download_as_bytes()
            image_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            return img

        except Exception as e:
            logger.error(f"Failed to read image {path}: {e}")
            return None
        
    def delete_local_image(self,group_id, path: str):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(script_dir, "warm-images", f"{group_id}" ,  f"compressed_{path}.jpg")
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                print(f"🗑️ Deleted {local_path}")
            else:
                print(f"⚠️ File not found: {local_path}")
        except Exception as e:
            print(f"❌ Failed to delete {local_path}: {e}")

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

    def process_single_image(self, group_id ,image_id: int, location: str, yolo_model) -> List[dict]:
        """Optimized single image processing"""
        try:
            # Load image
            img = self.read_image(group_id,image_id)
            if img is None:
                return []

            # Detect faces
            all_faces = self.face_app.get(img)
            if not all_faces:
                return []

            # YOLO person detection
            results = yolo_model(img)[0]
            person_boxes = []
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > config.PERSON_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append([x1, y1, x2, y2])

            if not person_boxes:
                return []

            # Process faces
            records = []
            faces_with_bboxes = [(face, face.bbox) for face in all_faces]
            unique_faces = self.deduplicate_faces(faces_with_bboxes)
            
            for face, face_bbox in unique_faces:
                record = self._process_single_face(face, face_bbox, person_boxes, img, image_id)
                if record:
                    records.append(record)
            # self.delete_local_image(group_id,image_id)
            return records
            
        except Exception as e:
            logger.error(f"Failed to process image {image_id}: {e}")
            return []

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
                "clothing_embedding": clothing_emb.cpu().tolist()
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

    def process_images_batch(self,group_id, images_batch: List[Tuple[int, str]], yolo_model) -> List[dict]:
        """Process batch of images with optimized parallelism"""
        logger.info(f"Processing batch of {len(images_batch)} images")
        
        all_records = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.PARALLEL_LIMIT) as executor:
            futures = {
                executor.submit(self.process_single_image,group_id, img_id, location, yolo_model): img_id
                for img_id, location in images_batch
            }
            
            for future in concurrent.futures.as_completed(futures):
                img_id = futures[future]
                try:
                    records = future.result(timeout=120)  # Increased timeout
                    all_records.extend(records)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout processing image {img_id}")
                except Exception as e:
                    logger.error(f"Error processing image {img_id}: {e}")
        
        return all_records

def process_group_optimized(group_id: int, indexer: OptimizedFaceIndexer, yolo_model) -> None:
    """Optimized group processing with batched operations"""
    try:
        indexer.setup_collection(group_id)
        logger.info(f"Processing group {group_id}")
        
        processed_count = 0
        
        while True:
            # Fetch batch of images
            unprocessed = DatabaseManager.fetch_unprocessed_images(group_id, config.BATCH_SIZE)
            
            if not unprocessed:
                logger.info(f"No more unprocessed images for group {group_id}")
                break
            
            logger.info(f"Found {len(unprocessed)} unprocessed images for group {group_id}")
            
            start_time = time.time()
            
            # Step 1: Process all images in the batch (extract embeddings)
            logger.info("Step 1: Processing images and extracting embeddings...")
            all_records = indexer.process_images_batch(group_id, unprocessed, yolo_model)
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f}s")
            
            if all_records:
                # Step 2: Batch insert to Qdrant
                logger.info("Step 2: Inserting to Qdrant...")
                qdrant_start = time.time()
                indexer.batch_upsert_to_qdrant(all_records, group_id)
                qdrant_time = time.time() - qdrant_start
                logger.info(f"Qdrant insert completed in {qdrant_time:.2f}s")
                
                # Step 3: Batch insert to database
                logger.info("Step 3: Inserting to database...")
                db_start = time.time()
                # Remove embeddings from records before DB insert (they're already in Qdrant)
                db_records = []
                for record in all_records:
                    db_record = {k: v for k, v in record.items() 
                               if k not in ['face_embedding', 'clothing_embedding']}
                    db_records.append(db_record)
                
                DatabaseManager.insert_faces_batch(db_records, group_id)
                db_time = time.time() - db_start
                logger.info(f"Database insert completed in {db_time:.2f}s")
            
            # Step 4: Mark images as processed
            processed_image_ids = [img_id for img_id, _ in unprocessed]
            DatabaseManager.mark_images_processed_batch(processed_image_ids)
            
            processed_count += len(unprocessed)
            total_time = time.time() - start_time
            
            logger.info(f"Group {group_id}: Processed {processed_count} images so far, "
                       f"{len(all_records)} faces indexed in {total_time:.2f}s total")
            logger.info(f"Performance: {len(unprocessed)/total_time:.2f} images/sec, "
                       f"{len(all_records)/total_time:.2f} faces/sec")
            
            if len(unprocessed) < config.BATCH_SIZE:
                break
        
        logger.info(f"Completed processing group {group_id}: {processed_count} total images processed")
        
    except Exception as e:
        logger.error(f"Failed to process group {group_id}: {e}")
        raise

def main_optimized():
    """Optimized main execution with connection validation and model pre-loading"""
    
    # Step 1: Validate all connections
    logger.info("=== STEP 1: VALIDATING CONNECTIONS ===")
    if not ConnectionValidator.validate_all():
        logger.error("Connection validation failed. Exiting.")
        return False
    
    # Step 2: Check if there's work to do
    logger.info("=== STEP 2: CHECKING FOR WORK ===")
    groups = DatabaseManager.fetch_warm_groups()
    if not groups:
        logger.info("No warm groups found, exiting")
        return True
    
    logger.info(f"Found {len(groups)} warm groups to process")
    
    # Step 3: Load models (expensive operation, do once)
    logger.info("=== STEP 3: LOADING MODELS ===")
    start_model_load = time.time()
    
    logger.info("Loading YOLO model...")
    yolo_model = YOLO("yolov8x.pt")
    
    logger.info("Loading face indexer...")
    indexer = OptimizedFaceIndexer()
    
    model_load_time = time.time() - start_model_load
    logger.info(f"All models loaded in {model_load_time:.2f}s")
    
    # Step 4: Process each group
    logger.info("=== STEP 4: PROCESSING GROUPS ===")
    
    for group_id in groups:
        try:
            group_start = time.time()
            
            # Mark group as being processed
            DatabaseManager.mark_group_process_status(group_id)
            
            # Process the group
            process_group_optimized(group_id, indexer, yolo_model)
            
            # Mark group as completed
            DatabaseManager.mark_group_processed(group_id)
            DatabaseManager.mark_group_process_status(0)
            
            group_time = time.time() - group_start
            logger.info(f"Group {group_id} completed in {group_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to process group {group_id}, continuing with next group: {e}")
            DatabaseManager.mark_group_process_status(0)  # Reset status on error
            continue
    
    logger.info("=== PROCESSING COMPLETED SUCCESSFULLY ===")
    return True

def main_with_monitoring():
    """Main function with performance monitoring"""
    total_start = time.time()
    
    try:
        success = main_optimized()
        total_time = time.time() - total_start
        
        if success:
            logger.info(f"🎉 Processing completed successfully in {total_time:.2f}s")
        else:
            logger.error(f"❌ Processing failed after {total_time:.2f}s")
            
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