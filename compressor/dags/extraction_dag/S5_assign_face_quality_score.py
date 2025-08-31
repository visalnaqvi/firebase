import os
import cv2
import psycopg2
import numpy as np
import logging
import concurrent.futures
from contextlib import contextmanager
from typing import List, Tuple, Optional
from psycopg2.extras import execute_values
import time

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

def _normalize(value, min_val, max_val):
    """Normalize value to 0-1 range with proper bounds checking"""
    if max_val <= min_val:
        return 0.0
    return float(min(max((value - min_val) / (max_val - min_val), 0.0), 1.0))

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

def compute_face_quality_from_data(face_thumb_bytes: bytes, insight_face_confidence: float) -> float:
    """
    Simplified face quality computation using InsightFace confidence as visibility parameter
    Returns 0-1 where 1 is highest quality
    
    Components:
    - InsightFace confidence (0-1): indicates face visibility and detection quality
    - Brightness/contrast score (0-1): lighting quality 
    - Sharpness score (0-1): image clarity
    """
    if not face_thumb_bytes:
        return 0.0

    # Convert bytes to image
    face_crop = bytes_to_cv_image(face_thumb_bytes)
    if face_crop is None:
        return 0.0

    # Log image dimensions for debugging
    h, w = face_crop.shape[:2]
    logger.debug(f"Processing face crop: {w}x{h}, confidence: {insight_face_confidence:.3f}")

    # If InsightFace confidence is 0 or very low, face is not visible/usable
    if insight_face_confidence <= 0.0:
        logger.debug("InsightFace confidence is 0 - returning 0.0")
        return 0.0
    
    # Normalize InsightFace confidence (assuming it's already 0-1 range)
    confidence_score = min(1.0, max(0.0, insight_face_confidence))
    
    # Compute image quality metrics
    sharpness_score = compute_sharpness_score(face_crop)
    brightness_score = compute_brightness_contrast_score(face_crop)
    
    # Log individual scores for debugging
    logger.debug(f"Score breakdown - InsightFace confidence: {confidence_score:.3f}, "
                f"Sharpness: {sharpness_score:.3f}, Brightness: {brightness_score:.3f}")
    
    # Weighted combination - InsightFace confidence is most important
    final_score = (
        0.60 * confidence_score +     # Face visibility/detection quality (most important)
        0.25 * sharpness_score +      # Image clarity 
        0.15 * brightness_score       # Lighting quality
    )
    
    # Apply severe penalties only for extremely poor conditions
    penalties = 0.0
    
    # Penalty for extremely poor sharpness (very blurry)
    if sharpness_score < 0.1:
        penalties += 0.2
    
    # Penalty for extremely poor brightness (very dark or overexposed)
    if brightness_score < 0.1:
        penalties += 0.15
    
    # Apply penalties
    final_score = max(0.0, final_score - penalties)
    
    # Bonus for high-quality faces
    if confidence_score > 0.8 and sharpness_score > 0.6:
        final_score = min(1.0, final_score + 0.05)
    
    logger.debug(f"Final score: {final_score:.3f} (base: {final_score + penalties:.3f}, penalties: {penalties:.3f})")
    
    return float(min(max(final_score, 0.0), 1.0))

def bytes_to_cv_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Convert bytes to OpenCV image"""
    try:
        if not image_bytes:
            return None
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        logger.warning(f"Failed to decode image bytes: {e}")
        return None

def process_single_face(face_data: Tuple[str, bytes, float]) -> Tuple[str, float]:
    """
    Process a single face and return its ID and quality score
    """
    face_id, face_thumb_bytes, insight_face_confidence = face_data
    
    try:
        quality_score = compute_face_quality_from_data(face_thumb_bytes, insight_face_confidence)
        logger.debug(f"Face {face_id}: quality score = {quality_score:.3f}")
        return face_id, quality_score
    except Exception as e:
        logger.error(f"Error processing face {face_id}: {e}")
        return face_id, 0.0

class FaceQualityProcessor:
    """Handles face quality processing operations"""
    
    BATCH_SIZE = 50
    PARALLEL_WORKERS = 5
    
    @staticmethod
    def fetch_unprocessed_faces(batch_size: int = BATCH_SIZE) -> List[Tuple[str, bytes, float]]:
        """Fetch faces with quality_score = -1 (unprocessed) along with insight_face_confidence"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, face_thumb_bytes, COALESCE(insight_face_confidence, 0.0) as confidence
                    FROM faces 
                    WHERE quality_score = -1 
                    AND face_thumb_bytes IS NOT NULL
                    LIMIT %s
                    """, 
                    (batch_size,)
                )
                results = cur.fetchall()
                logger.info(f"Fetched {len(results)} unprocessed faces")
                return results
    
    @staticmethod
    def update_quality_scores_batch(face_scores: List[Tuple[str, float]]) -> None:
        """Update quality scores for a batch of faces"""
        if not face_scores:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                UPDATE faces 
                SET quality_score = %s
                WHERE id = %s
                """
                values = [(score, face_id) for face_id, score in face_scores]
                
                cur.executemany(query, values)
                conn.commit()
                
                logger.info(f"Updated quality scores for {len(face_scores)} faces")
    
    @staticmethod
    def get_unprocessed_count() -> int:
        """Get count of unprocessed faces"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM faces WHERE quality_score = -1 AND face_thumb_bytes IS NOT NULL"
                )
                return cur.fetchone()[0]
    
    def process_faces_batch(self, faces_batch: List[Tuple[str, bytes, float]]) -> List[Tuple[str, float]]:
        """Process a batch of faces with parallel execution"""
        logger.info(f"Processing batch of {len(faces_batch)} faces")
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.PARALLEL_WORKERS) as executor:
            # Submit all faces for processing
            future_to_face = {
                executor.submit(process_single_face, face_data): face_data[0]
                for face_data in faces_batch
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_face):
                face_id = future_to_face[future]
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout processing face {face_id}")
                    results.append((face_id, 0.0))
                except Exception as e:
                    logger.error(f"Error processing face {face_id}: {e}")
                    results.append((face_id, 0.0))
        
        return results

def main():
    """Main processing function"""
    processor = FaceQualityProcessor()
    
    # Get initial count
    total_unprocessed = processor.get_unprocessed_count()
    logger.info(f"Starting processing of {total_unprocessed} unprocessed faces")
    
    if total_unprocessed == 0:
        logger.info("No unprocessed faces found")
        return
    
    processed_count = 0
    batch_count = 0
    zero_score_count = 0
    
    while True:
        batch_count += 1
        start_time = time.time()
        
        # Fetch next batch
        faces_batch = processor.fetch_unprocessed_faces()
        
        if not faces_batch:
            logger.info("No more unprocessed faces found")
            break
        
        logger.info(f"Processing batch {batch_count} with {len(faces_batch)} faces")
        
        # Process the batch
        face_scores = processor.process_faces_batch(faces_batch)
        processed_count += len(faces_batch)
        batch_time = time.time() - start_time
        
        # Calculate statistics
        quality_scores = [score for _, score in face_scores]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        zero_count = sum(1 for score in quality_scores if score == 0.0)
        high_quality_count = sum(1 for score in quality_scores if score > 0.7)
        medium_quality_count = sum(1 for score in quality_scores if 0.4 <= score <= 0.7)
        low_quality_count = sum(1 for score in quality_scores if 0.0 < score < 0.4)
        
        zero_score_count += zero_count
        
        logger.info(f"Batch {batch_count} completed in {batch_time:.2f}s:")
        logger.info(f"  Processed: {len(faces_batch)} faces")
        logger.info(f"  Average quality: {avg_quality:.3f}")
        logger.info(f"  Zero quality (low confidence): {zero_count}")
        logger.info(f"  High quality (>0.7): {high_quality_count}")
        logger.info(f"  Medium quality (0.4-0.7): {medium_quality_count}")
        logger.info(f"  Low quality (0-0.4): {low_quality_count}")
        logger.info(f"  Total processed so far: {processed_count}/{total_unprocessed}")
        
        # Update database
        processor.update_quality_scores_batch(face_scores)
        
        # If we got fewer faces than batch size, we're probably done
        if len(faces_batch) < processor.BATCH_SIZE:
            break
    
    logger.info(f"Processing completed! Total faces processed: {processed_count}")
    logger.info(f"Total faces with 0 score (low InsightFace confidence): {zero_score_count}")

if __name__ == "__main__":
    try:
        main()
        print("Sleeping for 10 minutes...")
        time.sleep(600)  # 600 seconds = 10 minutes
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise