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

def compute_face_occlusion_score(face_crop: np.ndarray) -> float:
    """
    Detect face occlusion by checking for symmetry and completeness
    Returns 0-1 where 1 means no occlusion
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        if w < 10 or h < 10:  # Too small to analyze
            return 0.1
        
        # Check left-right symmetry (faces should be roughly symmetric)
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        if min_width < 5:
            return 0.2
            
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Compute correlation between halves
        correlation = cv2.matchTemplate(left_half.astype(np.float32), 
                                      right_half_flipped.astype(np.float32), 
                                      cv2.TM_CCOEFF_NORMED)[0, 0]
        
        symmetry_score = max(0, correlation)
        
        # Check for uniform regions (might indicate occlusion)
        std_dev = np.std(gray)
        if std_dev < 10:  # Very uniform, likely occluded
            return 0.2
        
        return min(1.0, symmetry_score)
        
    except Exception as e:
        logger.warning(f"Error in occlusion detection: {e}")
        return 0.5

def compute_brightness_contrast_score(face_crop: np.ndarray) -> float:
    """
    Evaluate if face is too dark or has poor contrast
    Returns 0-1 where 1 means optimal brightness/contrast
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Brightness check (mean intensity)
        mean_brightness = np.mean(gray)
        brightness_score = _normalize(mean_brightness, 50, 200)
        
        # Contrast check (standard deviation)
        contrast = np.std(gray)
        contrast_score = _normalize(contrast, 15, 80)
        
        # Dynamic range check
        min_val, max_val = np.min(gray), np.max(gray)
        dynamic_range = max_val - min_val
        range_score = _normalize(dynamic_range, 50, 200)
        
        # Combined score with weights
        combined_score = 0.4 * brightness_score + 0.4 * contrast_score + 0.2 * range_score
        return combined_score
        
    except Exception as e:
        logger.warning(f"Error in brightness/contrast analysis: {e}")
        return 0.0

def compute_sharpness_score(face_crop: np.ndarray) -> float:
    """
    Improved sharpness detection using multiple methods
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        laplacian_score = _normalize(laplacian_var, 100, 2000)
        
        # Method 2: Sobel gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_score = _normalize(np.mean(sobel_magnitude), 10, 50)
        
        # Weighted combination
        sharpness = 0.7 * laplacian_score + 0.3 * sobel_score
        return sharpness
        
    except Exception as e:
        logger.warning(f"Error in sharpness analysis: {e}")
        return 0.0

def compute_face_size_score(face_crop: np.ndarray) -> float:
    """
    Score based on face size - larger faces generally have more detail
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        h, w = face_crop.shape[:2]
        face_area = h * w
        
        # Optimal face size range (in pixels)
        min_good_area = 50 * 50     # 50x50 minimum
        optimal_area = 120 * 120    # 120x120 optimal
        
        if face_area < min_good_area:
            return face_area / min_good_area
        elif face_area > optimal_area:
            return 1.0
        else:
            return _normalize(face_area, min_good_area, optimal_area)
            
    except Exception:
        return 0.0

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

def compute_face_quality_from_bytes(face_thumb_bytes: bytes) -> float:
    """
    Compute face quality score from face thumbnail bytes
    Returns 0-1 where 1 is highest quality
    """
    if not face_thumb_bytes:
        return 0.0

    # Convert bytes to image
    face_crop = bytes_to_cv_image(face_thumb_bytes)
    if face_crop is None:
        return 0.0

    # Component scores
    sharpness_score = compute_sharpness_score(face_crop)
    brightness_score = compute_brightness_contrast_score(face_crop)
    occlusion_score = compute_face_occlusion_score(face_crop)
    size_score = compute_face_size_score(face_crop)
    
    # Weighted combination - adjust weights based on importance
    quality_score = (
        0.30 * sharpness_score +      # Critical for recognition
        0.25 * occlusion_score +      # Critical for completeness
        0.25 * brightness_score +     # Important for visibility
        0.20 * size_score             # Size adequacy
    )
    
    # Apply penalties for very poor conditions
    penalties = 0
    
    # Heavy penalty for very blurry faces
    if sharpness_score < 0.15:
        penalties += 0.4
    
    # Heavy penalty for heavily occluded faces
    if occlusion_score < 0.25:
        penalties += 0.4
    
    # Penalty for very dark faces
    if brightness_score < 0.15:
        penalties += 0.3
    
    # Penalty for very small faces
    if size_score < 0.2:
        penalties += 0.2
    
    # Apply penalties
    final_score = max(0.0, quality_score - penalties)
    
    return float(min(max(final_score, 0.0), 1.0))

def process_single_face(face_data: Tuple[str, bytes]) -> Tuple[str, float]:
    """
    Process a single face and return its ID and quality score
    """
    face_id, face_thumb_bytes = face_data
    
    try:
        quality_score = compute_face_quality_from_bytes(face_thumb_bytes)
        return face_id, quality_score
    except Exception as e:
        logger.error(f"Error processing face {face_id}: {e}")
        return face_id, 0.0

class FaceQualityProcessor:
    """Handles face quality processing operations"""
    
    BATCH_SIZE = 50
    PARALLEL_WORKERS = 2  
    @staticmethod
    def fetch_unprocessed_faces(batch_size: int = BATCH_SIZE) -> List[Tuple[str, bytes]]:
        """Fetch faces with quality_score = -1 (unprocessed)"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, face_thumb_bytes 
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
                # IMPORTANT: order must match (score, id)
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
    
    def process_faces_batch(self, faces_batch: List[Tuple[str, bytes]]) -> List[Tuple[str, float]]:
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
                    result = future.result(timeout=30)  # 30 second timeout per face
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout processing face {face_id}")
                    results.append((face_id, 0.0))  # Give failed faces a score of 0
                except Exception as e:
                    logger.error(f"Error processing face {face_id}: {e}")
                    results.append((face_id, 0.0))  # Give failed faces a score of 0
        
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
        high_quality_count = sum(1 for score in quality_scores if score > 0.7)
        medium_quality_count = sum(1 for score in quality_scores if 0.4 <= score <= 0.7)
        low_quality_count = sum(1 for score in quality_scores if score < 0.4)
        
        logger.info(f"Batch {batch_count} completed in {batch_time:.2f}s:")
        logger.info(f"  Processed: {len(faces_batch)} faces")
        logger.info(f"  Average quality: {avg_quality:.3f}")
        logger.info(f"  High quality (>0.7): {high_quality_count}")
        logger.info(f"  Medium quality (0.4-0.7): {medium_quality_count}")
        logger.info(f"  Low quality (<0.4): {low_quality_count}")
        logger.info(f"  Total processed so far: {processed_count}/{total_unprocessed}")
        # Update database
        processor.update_quality_scores_batch(face_scores)
        
        # If we got fewer faces than batch size, we're probably done
        if len(faces_batch) < processor.BATCH_SIZE:
            break
    
    logger.info(f"Processing completed! Total faces processed: {processed_count}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise