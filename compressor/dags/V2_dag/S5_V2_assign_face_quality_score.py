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
import argparse
class ProcessingError(Exception):
    def __init__(self, message, group_id=None, reason=None, retryable=True):
        super().__init__(message)
        self.group_id = group_id
        self.reason = reason
        self.retryable = retryable

    def __str__(self):
        return f"ProcessingError: {self.args[0]} (group_id={self.group_id}, reason={self.reason}, retryable={self.retryable})"
# Initialize Firebase once
# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
                    WHERE task = 'quality_assignment'
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
                        WHERE task = 'quality_assignment'
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
                        WHERE task = 'quality_assignment'
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
                        WHERE task = 'quality_assignment'
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
                        WHERE task = 'quality_assignment'
                        """,
                        (group_id,)
                    )
                    cur.execute(
                        """
                        UPDATE process_status
                        SET next_group_in_queue = %s
                        WHERE task = 'grouping' and next_group_in_queue is null 
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
        try:
            if not os.path.exists(self.json_path):
                raise ProcessingError(f"No faces.json found for group {self.group_id}")
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except ProcessingError as e:
            logger.error(f"Failed to load faces.json for group {self.group_id}: {e}")
            raise 
        except Exception as e:
            logger.error(f"Failed to load faces.json for group {self.group_id}: {e}")
            raise  ProcessingError(e)

    def save_faces(self, faces):
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(faces, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save faces.json for group {self.group_id}: {e}")
            raise

    def process_faces_batch(self, faces_batch: List[dict]) -> List[Tuple[str, float]]:
        results = []
        failedResults = []
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
                    logger.error(f"Error while assign quality: {e}")
                    failedResults.append((face["id"], None))
        return results , failedResults

    def run(self):
        try:
            faces = self.load_faces()
            unprocessed = [f for f in faces if f.get("quality_score", -1) == -1]

            logger.info(f"Group {self.group_id}: {len(unprocessed)} unprocessed faces")

            processed_count = 0
            success_count = 0
            failed_count = 0
            for i in range(0, len(unprocessed), self.BATCH_SIZE):
                batch = unprocessed[i:i+self.BATCH_SIZE]
                face_scores , failed_results = self.process_faces_batch(batch)
                for fid, score in face_scores:
                    for f in faces:
                        if f["id"] == fid:
                            f["quality_score"] = score
                            break
                processed_count += len(batch)
                success_count += len(face_scores)
                failed_count += len(failed_results)
                logger.info(f"Group {self.group_id}: processed {processed_count}/{len(unprocessed)}")

            self.save_faces(faces)
            logger.info(f"Group {self.group_id}: faces.json updated")
            return processed_count , failed_count , success_count
        except Exception as e:
            logger.error(f"Group {self.group_id} failed due to critical error: {e}")
            raise
            
# -------------------- Main --------------------
def main():
    # Update group status in DB
    try:            
        run_id  = int(time.time())
        group_id = get_or_assign_group_id()
        if not group_id:
            update_status(None , "No Group Found To Process" , True , "waiting")
            update_status_history(run_id , "assign_quality_score" , "run" , None , None , None , None , "no_group")
            return False
        update_status(group_id , "Running" , False , "healthy")
        update_status_history(run_id , "assign_quality_score" , "run" , None , None , None , group_id , "started")
        processor = FaceQualityProcessor(group_id)
        processed_count , failed_count , success_count = processor.run()
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE groups 
                SET last_processed_step = 'assign_quality_score' 
                WHERE id = %s
            """, (group_id,))
            conn.commit()
            logger.info(f"Group {group_id}: last_processed_step updated to 'assign_quality_score'")
        update_status(None , "Waiting" , True , "done")
        update_status_history(run_id , "assign_quality_score" , "run" , processed_count , failed_count , success_count , group_id , "done")
        update_last_provrssed_group_column(group_id)

        return True
    except Exception as e:
        logger.error(f"Failed to update DB for group {group_id}: {e}")
        update_status(group_id , f"Error while processing group : {e}" , True , "failed")
        update_status_history(run_id , "assign_quality_score" , "run" , None ,None , None , group_id , f"error while processing group : {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)