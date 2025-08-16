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
import math
import open_clip
def _normalize(value, min_val, max_val):
    if max_val <= min_val:
        return 0.0
    return float(min(max((value - min_val) / (max_val - min_val), 0.0), 1.0))

def estimate_frontalness_from_landmarks(face, face_crop):
    """
    Try to estimate frontalness using landmarks if explicit yaw/pitch not available.
    Returns a score 0..1 where 1 is frontal.
    """
    # many insightface outputs include 'kps' or 'landmark' as an array of 5 or 106 points
    kps = None
    if hasattr(face, "kps") and face.kps is not None:
        kps = np.array(face.kps)  # usually shape (5,2)
    elif hasattr(face, "landmark") and face.landmark is not None:
        kps = np.array(face.landmark)  # maybe (106,2)
    if kps is None or kps.size == 0:
        return 1.0  # give neutral credit if no landmarks

    # Use eye-center symmetry as a cheap proxy:
    try:
        # pick left/right eye points depending on format
        if kps.shape[0] >= 5:
            left_eye = kps[0]
            right_eye = kps[1]
        else:
            # fallback: use extremes in x
            xs = kps[:, 0]
            left_eye = kps[np.argmin(xs)]
            right_eye = kps[np.argmax(xs)]

        # horizontal displacement between eyes relative to width
        dx = abs(right_eye[0] - left_eye[0])
        width = face_crop.shape[1] if face_crop is not None else dx
        ratio = dx / (width + 1e-6)
        # if ratio is too small, face might be turned (one eye far behind)
        # normalize expected ratio ~ 0.25..0.45 typical -> map to 0..1
        return _normalize(ratio, 0.18, 0.45)
    except Exception:
        return 1.0

def compute_face_quality(face_crop: np.ndarray, face) -> float:
    """
    Compute a 0..1 quality score for the crop and the face object returned by insightface.
    face_crop: the actual image crop (BGR numpy array).
    face: the insightface face object (has bbox, det_score, maybe kps, maybe pose).
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0

    # 1) Sharpness (Laplacian variance)
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Typical values vary; tune min/max on sample data
        sharp_norm = _normalize(sharpness, 50, 1500)  # tune these bounds
    except Exception:
        sharp_norm = 0.0

    # 2) Detection confidence (if available)
    det_conf = 0.0
    if hasattr(face, "det_score") and face.det_score is not None:
        try:
            det_conf = float(face.det_score)
            det_conf = _normalize(det_conf, 0.3, 1.0)
        except Exception:
            det_conf = 0.0

    # 3) Frontalness / pose
    pose_score = 1.0
    # some models provide yaw/pitch/roll, or face.pose
    if hasattr(face, "yaw") or hasattr(face, "pose"):
        try:
            yaw = getattr(face, "yaw", 0.0)
            pitch = getattr(face, "pitch", 0.0)
            # assume degrees; clamp large values
            yaw = float(yaw) if yaw is not None else 0.0
            pitch = float(pitch) if pitch is not None else 0.0
            yaw_score = max(0.0, 1.0 - abs(yaw) / 40.0)  # 40° tolerance
            pitch_score = max(0.0, 1.0 - abs(pitch) / 40.0)
            pose_score = (yaw_score + pitch_score) / 2.0
        except Exception:
            pose_score = 1.0
    else:
        # fallback to landmarks-based estimate
        try:
            pose_score = estimate_frontalness_from_landmarks(face, face_crop)
        except Exception:
            pose_score = 1.0

    # 4) Completeness / bbox margin: ensure bbox isn't tightly cut at edges
    completeness = 1.0
    try:
        if hasattr(face, "bbox") and face.bbox is not None:
            x1, y1, x2, y2 = map(int, face.bbox)
            h, w = face_crop.shape[:2]
            # If bbox hits crop border, penalize
            margin_x = min(x1, w - x2)
            margin_y = min(y1, h - y2)
            min_margin = min(margin_x, margin_y)
            # normalize margin (0 -> bad, >= 10px -> good). Tune if needed.
            completeness = _normalize(min_margin, 0, 10)
        else:
            completeness = 1.0
    except Exception:
        completeness = 1.0

    # Weighted sum (tune weights)
    # Give sharpness and pose the most weight
    score = 0.45 * sharp_norm + 0.25 * pose_score + 0.15 * det_conf + 0.15 * completeness
    return float(min(max(score, 0.0), 1.0))
@dataclass
class Config:
    BATCH_SIZE: int = 10  # Increased for better efficiency
    PARALLEL_LIMIT: int = 2  # Reduced to prevent resource exhaustion
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
                cur.execute("SELECT id FROM groups WHERE status = 'warm' order by last_processed_at")
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def insert_faces_batch(records: List[dict], group_id: int) -> None:
        """Insert detected faces with transaction safety"""
        if not records:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                INSERT INTO faces (id, image_id, group_id, person_id, face_thumb_bytes , quality_score)
                VALUES %s
                """
                values = [
                    (r['id'], r['image_id'], group_id, r['person_id'], r['face_thumb_bytes'] , r['quality_score'])
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
                query = "UPDATE images SET status = 'warming' WHERE id = ANY(%s)"
                cur.execute(query, (image_ids,))
                conn.commit()
                logger.info(f"Marked {len(image_ids)} images as processed")
                
    @staticmethod
    def mark_group_processed(group_id) -> None:
        """Mark group_id as processed and clear image_byte"""
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
                logger.info(f"Marked {group_id} group_id as processed")

class HybridFaceIndexer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models with error handling
        try:
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0)
            
            # Fix 1: Load model with proper device handling for meta tensors
            try:
                model_name = "ViT-B-32"
                pretrained = "laion2b_s34b_b79k"  # or another CLIP pretrained set

                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                    device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
                # Move to device after loading
                # self.fashion_model = self.fashion_model.to(self.device)
            except Exception as meta_error:
                logger.warning(f"Meta tensor error, trying alternative loading: {meta_error}")
                # Alternative: Load on CPU first, then move
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
            
            img_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
                return emb[0]
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
                        pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                        clothing_emb = self.extract_clothing_embedding(pil_img)
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
                            face_crop = None
                            try:
                                x1_f, y1_f, x2_f, y2_f = map(int, face.bbox)
                                pad_x = int((x2_f - x1_f) * 0.4)
                                pad_y = int((y2_f - y1_f) * 0.4)
                                # clip bbox to person_crop bounds to avoid errors
                                x1_f = max(0, x1_f - pad_x) ; 
                                y1_f = max(0, y1_f - pad_y)
                                x2_f = min(person_crop.shape[1], x2_f + pad_x); 
                                y2_f = min(person_crop.shape[0], y2_f + pad_y)
                                if x2_f > x1_f and y2_f > y1_f:
                                    face_crop = person_crop[y1_f:y2_f, x1_f:x2_f]
                                    if face_crop.size > 0:
                                        face_thumb_bytes = self.image_to_bytes(face_crop)
                            except Exception:
                                face_crop = None
                                face_thumb_bytes = None

                            # compute quality score (use helper). fallback to 0.0
                            try:
                                quality_score = compute_face_quality(face_crop, face)
                            except Exception as e:
                                logger.warning(f"Quality scoring failed for image {image_id}, face: {e}")
                                quality_score = 0.0
                            
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
                                            "cloth_ids":None,
                                            "quality_score": quality_score
                                        }
                                    )
                                ]
                            )
                            
                            records.append({
                                "id": point_id,
                                "image_id": image_id,
                                "person_id": None,
                                "face_thumb_bytes": face_thumb_bytes,
                                "quality_score": quality_score
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
                DatabaseManager.mark_group_processed(group_id)
            except Exception as e:
                logger.error(f"Failed to process group {group_id}, continuing with next group: {e}")
                continue
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()