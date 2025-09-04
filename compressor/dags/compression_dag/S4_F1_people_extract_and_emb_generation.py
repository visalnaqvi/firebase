import os
import uuid
import cv2
import torch
import psycopg2
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator
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

def _normalize(value, min_val, max_val):
    if max_val <= min_val:
        return 0.0
    return float(min(max((value - min_val) / (max_val - min_val), 0.0), 1.0))

cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "gallery-585ee.firebasestorage.app"
})

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
    BATCH_SIZE: int = 20
    PARALLEL_LIMIT: int = 1
    PERSON_CONFIDENCE_THRESHOLD: float = 0.5
    MAX_RETRIES: int = 3
    FACE_OVERLAP_THRESHOLD: float = 0.7  # Threshold for considering faces as duplicates
    
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

class DatabaseManager:
    """Handles all database operations"""
    
    @staticmethod
    def fetch_unprocessed_images(group_id: int, batch_size: int) -> List[Tuple[int, bytes]]:
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

    @staticmethod
    def mark_group_process_status(group_id) -> None:
        """Mark group_id as processed and clear image_byte"""
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
                logger.info(f"Marked {group_id} group_id as processed")

    @staticmethod
    def insert_faces_batch(records: List[dict], group_id: int) -> None:
        """Insert detected faces with transaction safety"""
        if not records:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                INSERT INTO faces (id, image_id, group_id, person_id, face_thumb_bytes, quality_score , insight_face_confidence)
                VALUES %s
                """
                values = [
                    (r['id'], r['image_id'], group_id, r['person_id'], r['face_thumb_bytes'], r['quality_score'] , r['insight_face_confidence'])
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
                query = "UPDATE images SET status = 'warming' WHERE id = ANY(%s::uuid[])"
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
                logger.warning(f"Meta tensor error, trying alternative loading: {meta_error}")
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
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "face": VectorParams(size=512, distance=Distance.COSINE),
                        "cloth": VectorParams(size=512, distance=Distance.COSINE)
                    }
                )
                
            image_collection_name = f"image_{collection_name}"
            if not self.qdrant.collection_exists(image_collection_name):
                self.qdrant.create_collection(
                    collection_name=image_collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
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
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb[0]
        except Exception as e:
            logger.error(f"Failed to extract clothing embedding: {e}")
            raise

    # def image_to_bytes(self, cv_image: np.ndarray) -> bytes:
    #     """Convert OpenCV image to bytes with error handling"""
    #     try:
    #         success, buffer = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    #         if not success:
    #             raise ValueError("Could not encode image")
    #         return buffer.tobytes()
    #     except Exception as e:
    #         logger.error(f"Failed to convert image to bytes: {e}")
    #         raise
    
    def image_to_bytes(self, cv_image: np.ndarray, target_height: int = 150) -> bytes:
        """Convert OpenCV image to bytes with consistent sizing and error handling"""
        try:
            if cv_image is None or cv_image.size == 0:
                raise ValueError("Empty or invalid image")
            
            # Get original dimensions
            original_height, original_width = cv_image.shape[:2]
            
            # Calculate new width maintaining aspect ratio
            aspect_ratio = original_width / original_height
            new_width = int(target_height * aspect_ratio)
            
            # Resize the image
            resized_image = cv2.resize(cv_image, (new_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Encode to JPEG bytes
            success, buffer = cv2.imencode('.jpg', resized_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise ValueError("Could not encode image")
            
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            raise

    def read_image_from_firebase(self, path: str):
        bucket = storage.bucket()
        blob = bucket.blob("compressed_"+path)

        img_bytes = blob.download_as_bytes()
        image_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        return img
    def extract_full_image_embedding(self, image_input) -> torch.Tensor:
        """Extract complete image embeddings using the existing OpenCLIP model"""
        try:
            img_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb[0]
        except Exception as e:
            logger.error(f"Failed to extract full image embedding: {e}")
            raise

    def deduplicate_faces(self, faces_with_bboxes):
        """Remove duplicate faces based on overlap ratio"""
        if len(faces_with_bboxes) <= 1:
            return faces_with_bboxes
        
        # Sort by detection score (highest first)
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
        
        logger.info(f"Deduplicated faces: {len(faces_with_bboxes)} -> {len(unique_faces)}")
        return unique_faces

    def process_image(self, image_id: int, location, yolo_model, collection_name: str) -> List[dict]:
        """Process single image with improved face-clothing association"""
        try:
            # Load original image
            img = self.read_image_from_firebase(image_id)
            if img is None:
                logger.warning(f"Failed to decode image {image_id}")
                return []
            try:
                # Convert OpenCV image to PIL for embedding extraction
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_emb = self.extract_full_image_embedding(pil_img)
                
                # Store in image collection
                image_collection_name = f"image_{collection_name}"
                self.qdrant.upsert(
                    collection_name=image_collection_name,
                    points=[
                        PointStruct(
                            id=str(image_id),  # Use image_id as point_id
                            vector=image_emb.cpu().tolist(),
                            payload={
                                "image_id": image_id,
                                "group_id": collection_name,
                            }
                        )
                    ]
                )
                logger.info(f"Stored image embedding for image {image_id}")
                
            except Exception as e:
                logger.error(f"Failed to extract/store image embedding for {image_id}: {e}")
                # Step 1: Detect all faces in the original image
                all_faces = self.face_app.get(img)
                if not all_faces:
                    logger.info(f"No faces detected in image {image_id}")
                    return []

            # Step 2: YOLO person detection
            results = yolo_model(img)[0]
            person_boxes = []
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > config.PERSON_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append([x1, y1, x2, y2])

            if not person_boxes:
                logger.info(f"No persons detected in image {image_id}")
                return []

            # Step 3: Associate faces with persons and create face-clothing pairs
            records = []
            faces_with_bboxes = [(face, face.bbox) for face in all_faces]
            
            # Deduplicate faces
            unique_faces = self.deduplicate_faces(faces_with_bboxes)
            
            for face, face_bbox in unique_faces:
                try:
                    face_x1, face_y1, face_x2, face_y2 = map(int, face_bbox)
                    face_center_x = (face_x1 + face_x2) // 2
                    face_center_y = (face_y1 + face_y2) // 2
                    
                    # Find the person box that contains this face
                    best_person_box = None
                    for person_box in person_boxes:
                        px1, py1, px2, py2 = person_box
                        if px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2:
                            best_person_box = person_box
                            break
                    
                    if best_person_box is None:
                        logger.warning(f"No person box found for face in image {image_id}")
                        continue
                    
                    # Step 4: Extract clothing embedding using expanded face region
                    clothing_bbox = expand_face_bbox_for_clothing(
                        [face_x1 - best_person_box[0], face_y1 - best_person_box[1], 
                        face_x2 - best_person_box[0], face_y2 - best_person_box[1]], 
                        best_person_box
                    )
                    
                    # Ensure clothing bbox is within image bounds
                    h, w = img.shape[:2]
                    clothing_bbox[0] = max(0, clothing_bbox[0])
                    clothing_bbox[1] = max(0, clothing_bbox[1])
                    clothing_bbox[2] = min(w, clothing_bbox[2])
                    clothing_bbox[3] = min(h, clothing_bbox[3])
                    
                    if clothing_bbox[2] <= clothing_bbox[0] or clothing_bbox[3] <= clothing_bbox[1]:
                        logger.warning(f"Invalid clothing bbox for face in image {image_id}")
                        continue
                    
                    clothing_crop = img[clothing_bbox[1]:clothing_bbox[3], clothing_bbox[0]:clothing_bbox[2]]
                    
                    if clothing_crop.size == 0:
                        logger.warning(f"Empty clothing crop for face in image {image_id}")
                        continue
                    
                    # Extract clothing embedding
                    try:
                        pil_img = Image.fromarray(cv2.cvtColor(clothing_crop, cv2.COLOR_BGR2RGB))
                        clothing_emb = self.extract_clothing_embedding(pil_img)
                    except Exception as e:
                        logger.warning(f"Failed to extract clothing embedding for image {image_id}: {e}")
                        continue
                    
                    # Extract face embedding
                    face_emb = face.normed_embedding
                    point_id = str(uuid.uuid4())
                    
                    # Extract InsightFace confidence
                    insight_face_confidence = float(face.det_score) if hasattr(face, 'det_score') and face.det_score is not None else 0.0
                    
                    # Extract face thumbnail with padding
                    face_thumb_bytes = None
                    face_crop = None
                    try:
                        pad_x = int((face_x2 - face_x1) * 0.4)
                        pad_y = int((face_y2 - face_y1) * 0.4)
                        
                        padded_x1 = max(0, face_x1 - pad_x)
                        padded_y1 = max(0, face_y1 - pad_y)
                        padded_x2 = min(w, face_x2 + pad_x)
                        padded_y2 = min(h, face_y2 + pad_y)
                        
                        if padded_x2 > padded_x1 and padded_y2 > padded_y1:
                            face_crop = img[padded_y1:padded_y2, padded_x1:padded_x2]
                            if face_crop.size > 0:
                                face_thumb_bytes = self.image_to_bytes(face_crop)
                    except Exception:
                        face_crop = None
                        face_thumb_bytes = None

                    # Compute quality score
                    # try:
                    #     quality_score = compute_face_quality(face_crop, face)
                    #     quality_score = -1
                    # except Exception as e:
                    #     logger.warning(f"Quality scoring failed for image {image_id}, face: {e}")
                    #     quality_score = 0.0
                    
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
                                    "cloth_ids": None,
                                }
                            )
                        ]
                    )
                    
                    records.append({
                        "id": point_id,
                        "image_id": image_id,
                        "person_id": None,
                        "face_thumb_bytes": face_thumb_bytes,
                        "quality_score": -1,
                        "insight_face_confidence":insight_face_confidence
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
                executor.submit(self.process_image, img_id, location, yolo_model, collection_name): img_id
                for img_id, location in images_batch
            }
            
            for future in concurrent.futures.as_completed(futures):
                img_id = futures[future]
                try:
                    result = future.result(timeout=60)
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
            unprocessed = DatabaseManager.fetch_unprocessed_images(group_id, config.BATCH_SIZE)
            
            if not unprocessed:
                logger.info(f"No more unprocessed images for group {group_id}")
                break
            
            logger.info(f"Found {len(unprocessed)} unprocessed images for group {group_id}")
            
            records = indexer.process_images_batch(unprocessed, yolo_model, group_id)
            processed_image_ids = [img_id for img_id, _ in unprocessed]
            
            if records:
                DatabaseManager.insert_faces_batch(records, group_id)
            
            DatabaseManager.mark_images_processed_batch(processed_image_ids)
            
            processed_count += len(unprocessed)
            logger.info(f"Group {group_id}: Processed {processed_count} images so far, {len(records)} faces indexed")
            
            if len(unprocessed) < config.BATCH_SIZE:
                break
            
        logger.info(f"Completed processing group {group_id}: {processed_count} total images processed")
        
    except Exception as e:
        logger.error(f"Failed to process group {group_id}: {e}")
        raise

def main():
    """Main execution function with proper error handling"""
    logger.info("Initializing YOLO model...")
    yolo_model = YOLO("yolov8x.pt")
    
    logger.info("Initializing face indexer...")
    indexer = HybridFaceIndexer()
    
    while True:
        try:
            groups = DatabaseManager.fetch_warm_groups()
            logger.info(f"Found {len(groups)} warm groups to process")
            
            if not groups or len(groups) == 0:
                break
            
            if not groups:
                logger.info("No warm groups found, exiting")
                return
            
            for group_id in groups:
                try:
                    DatabaseManager.mark_group_process_status(group_id)
                    process_group(group_id, indexer, yolo_model)
                    DatabaseManager.mark_group_processed(group_id)
                    DatabaseManager.mark_group_process_status(0)
                except Exception as e:
                    logger.error(f"Failed to process group {group_id}, continuing with next group: {e}")
                    continue
            
            logger.info("Processing completed successfully")

        except Exception as e:
            logger.error(f"Critical error in main execution: {e}")
            raise

if __name__ == "__main__":
    main()