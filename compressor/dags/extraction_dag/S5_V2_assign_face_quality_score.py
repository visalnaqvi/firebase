import os
import cv2
import json
import psycopg2
import numpy as np
import logging
import concurrent.futures
import time
from typing import List, Tuple, Optional
from psycopg2.extras import DictCursor

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- DB Connection --------------------
def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

# -------------------- Image Quality Utils --------------------
def _normalize(value, min_val, max_val):
    if max_val <= min_val:
        return 0.0
    return float(min(max((value - min_val) / (max_val - min_val), 0.0), 1.0))

def bytes_to_cv_image(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        if not image_bytes:
            return None
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"Failed to decode image bytes: {e}")
        return None

def compute_sharpness_score(face_crop: np.ndarray) -> float:
    """
    Improved sharpness detection optimized for face thumbnails
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Method 1: Laplacian variance (adjusted thresholds for small images)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Adjusted thresholds for smaller images
        laplacian_score = _normalize(laplacian_var, 20, 800)
        
        # Method 2: Sobel gradient (works well on small images)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_score = _normalize(np.mean(sobel_magnitude), 5, 25)
        
        # Method 3: Variance of image (simple but effective for small images)
        variance_score = _normalize(np.var(gray), 100, 2000)
        
        # Weighted combination
        sharpness = (
            0.4 * laplacian_score + 
            0.4 * sobel_score + 
            0.2 * variance_score
        )
        
        return min(1.0, sharpness)
        
    except Exception as e:
        logger.warning(f"Error in sharpness analysis: {e}")
        return 0.0

def compute_brightness_contrast_score(face_crop: np.ndarray) -> float:
    """
    Evaluate brightness and contrast - adjusted for thumbnails
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Brightness check (mean intensity)
        mean_brightness = np.mean(gray)
        
        # More forgiving brightness ranges
        if mean_brightness < 20:  # Very dark
            brightness_score = mean_brightness / 20 * 0.2
        elif mean_brightness > 235:  # Very bright  
            brightness_score = (255 - mean_brightness) / 20 * 0.3 + 0.7
        else:
            # Optimal range is wider
            brightness_score = _normalize(mean_brightness, 40, 200)
        
        # Contrast check (standard deviation)
        contrast = np.std(gray)
        contrast_score = _normalize(contrast, 10, 50)
        
        # Dynamic range
        min_val, max_val = np.min(gray), np.max(gray)
        dynamic_range = max_val - min_val
        range_score = _normalize(dynamic_range, 30, 120)
        
        # Combined score
        combined_score = 0.3 * brightness_score + 0.4 * contrast_score + 0.3 * range_score
        return max(0.1, combined_score)  # Minimum 10%
        
    except Exception as e:
        logger.warning(f"Error in brightness/contrast analysis: {e}")
        return 0.4

def compute_face_quality_from_file(image_path: str, insight_face_confidence: float) -> float:
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return 0.0
    with open(image_path, "rb") as f:
        face_thumb_bytes = f.read()
    face_crop = bytes_to_cv_image(face_thumb_bytes)
    if face_crop is None:
        return 0.0
    confidence_score = min(1.0, max(0.0, insight_face_confidence))
    sharpness_score = compute_sharpness_score(face_crop)
    brightness_score = compute_brightness_contrast_score(face_crop)
    final_score = (
        0.60 * confidence_score +
        0.25 * sharpness_score +
        0.15 * brightness_score
    )
    if sharpness_score < 0.1:
        final_score -= 0.2
    if brightness_score < 0.1:
        final_score -= 0.15
    if confidence_score > 0.8 and sharpness_score > 0.6:
        final_score += 0.05
    return float(min(max(final_score, 0.0), 1.0))

# -------------------- Processor --------------------
class FaceQualityProcessor:
    BATCH_SIZE = 50
    PARALLEL_WORKERS = 5

    def __init__(self, group_id: int):
        self.group_id = group_id
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.group_folder = os.path.join(self.script_dir,"warm-images", str(group_id), "faces")
        self.json_path = os.path.join(self.group_folder, "faces.json")

    def load_faces(self):
        if not os.path.exists(self.json_path):
            logger.error(f"No faces.json found for group {self.group_id}")
            return []
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_faces(self, faces):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(faces, f, indent=2)

    def process_faces_batch(self, faces_batch: List[dict]) -> List[Tuple[str, float]]:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.PARALLEL_WORKERS) as executor:
            future_to_face = {
                executor.submit(
                    compute_face_quality_from_file,
                    os.path.join(self.group_folder, f"{face['id']}.jpg"),
                    face.get("insight_face_confidence", 0.0)
                ): face for face in faces_batch
            }
            for future in concurrent.futures.as_completed(future_to_face):
                face = future_to_face[future]
                try:
                    score = future.result(timeout=30)
                    results.append((face["id"], score))
                except Exception as e:
                    logger.error(f"Error processing face {face['id']}: {e}")
                    results.append((face["id"], 0.0))
        return results

    def run(self):
        faces = self.load_faces()
        unprocessed = [f for f in faces if f.get("quality_score", -1) == -1]

        logger.info(f"Group {self.group_id}: {len(unprocessed)} unprocessed faces")

        processed_count = 0
        for i in range(0, len(unprocessed), self.BATCH_SIZE):
            batch = unprocessed[i:i+self.BATCH_SIZE]
            face_scores = self.process_faces_batch(batch)
            for fid, score in face_scores:
                for f in faces:
                    if f["id"] == fid:
                        f["quality_score"] = score
                        break
            processed_count += len(batch)
            logger.info(f"Group {self.group_id}: processed {processed_count}/{len(unprocessed)}")

        self.save_faces(faces)
        logger.info(f"Group {self.group_id}: faces.json updated")

# -------------------- Main --------------------
def main():
    try:
        with get_db_connection() as conn, conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT id 
                FROM groups 
                WHERE status = 'warming' 
                  AND last_processed_step = 'extraction'
            """)
            groups = cur.fetchall()

        if not groups:
            logger.info("No groups pending quality assignment")
            return

        for g in groups:
            group_id = g["id"]
            processor = FaceQualityProcessor(group_id)
            processor.run()

            # Update group status
            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute("""
                    UPDATE groups 
                    SET last_processed_step = 'assign_quality_score' 
                    WHERE id = %s
                """, (group_id,))
                conn.commit()
                logger.info(f"Group {group_id}: last_processed_step updated to 'assign_quality_score'")

    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    main()
