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

# InsightFace imports
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    print("InsightFace is available")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not available. Please install with: pip install insightface")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global face analysis app - initialize once
face_app = None

def initialize_face_analysis():
    """Initialize InsightFace face analysis app"""
    global face_app
    if not INSIGHTFACE_AVAILABLE:
        logger.error("InsightFace not available. Cannot perform face detection.")
        return False
    
    try:
        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use CPU for stability
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize InsightFace: {e}")
        return False

def detect_faces_count(image: np.ndarray) -> int:
    """
    Detect faces in image using InsightFace and return count
    Returns -1 if detection fails
    """
    global face_app
    
    if face_app is None:
        logger.error("Face analysis app not initialized")
        return -1
    
    try:
        # InsightFace expects RGB format
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Detect faces
        faces = face_app.get(image_rgb)
        face_count = len(faces)
        
        logger.debug(f"Detected {face_count} faces in image")
        return face_count
        
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return -1

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

def compute_frontal_face_score(face_crop: np.ndarray) -> float:
    """
    Detect if face is frontal vs profile/side view - ADJUSTED FOR THUMBNAILS
    Returns 0-1 where 1 means clear frontal face
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        if w < 15 or h < 15:  # Reduced minimum size for thumbnails
            return 0.2
        
        # Check left-right symmetry for frontal detection
        center_x = w // 2
        margin = max(2, w // 8)  # Reduced margin for smaller thumbnails
        
        left_region = gray[:, max(0, center_x - margin - w//4):center_x - margin]
        right_region = gray[:, center_x + margin:min(w, center_x + margin + w//4)]
        
        if left_region.size == 0 or right_region.size == 0:
            return 0.3
        
        # Resize to match
        min_width = min(left_region.shape[1], right_region.shape[1])
        min_height = min(left_region.shape[0], right_region.shape[0])
        
        if min_width < 5 or min_height < 5:  # Reduced minimum for thumbnails
            return 0.3
        
        left_resized = cv2.resize(left_region, (min_width, min_height))
        right_resized = cv2.resize(cv2.flip(right_region, 1), (min_width, min_height))
        
        # Calculate correlation
        correlation = cv2.matchTemplate(
            left_resized.astype(np.float32), 
            right_resized.astype(np.float32), 
            cv2.TM_CCOEFF_NORMED
        )[0, 0]
        
        symmetry_score = max(0, correlation)
        
        # Simplified eye detection for small thumbnails
        eye_region = gray[:h//2, :]
        eye_region_blur = cv2.GaussianBlur(eye_region, (3, 3), 0)
        
        # Use adaptive threshold for better small image handling
        binary = cv2.adaptiveThreshold(eye_region_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count potential eye regions with relaxed criteria
        eye_candidates = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < (h * w * 0.15):  # More relaxed eye size for thumbnails
                eye_candidates += 1
        
        eye_bonus = min(1.0, eye_candidates / 2.0) * 0.8  # Reduced weight
        
        frontal_score = 0.6 * symmetry_score + 0.4 * eye_bonus
        return min(1.0, frontal_score)
        
    except Exception as e:
        logger.warning(f"Error in frontal face detection: {e}")
        return 0.4  # More generous fallback

def compute_face_completeness_score(face_crop: np.ndarray) -> float:
    """
    FIXED: Completeness detection for face thumbnails
    Since thumbnails are cropped to faces, we check for obvious cut-offs rather than coverage
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # For thumbnails, we look for abrupt cut-offs at edges rather than coverage
        edge_thickness = max(2, min(h, w) // 20)  # Thinner edge for small images
        
        # Get edge regions
        top_edge = gray[:edge_thickness, :]
        bottom_edge = gray[-edge_thickness:, :]
        left_edge = gray[:, :edge_thickness]
        right_edge = gray[:, -edge_thickness:]
        
        def calculate_edge_cutoff(region, opposite_region):
            """Calculate if there's an abrupt cutoff by comparing edges"""
            if region.size == 0 or opposite_region.size == 0:
                return 0
            
            # Calculate gradient at edge vs gradient in opposite region
            edge_gradient = np.std(region)
            opposite_gradient = np.std(opposite_region)
            
            # If edge has much higher variation than opposite, likely cut-off
            if opposite_gradient > 0:
                cutoff_ratio = edge_gradient / opposite_gradient
                return min(1.0, max(0, (cutoff_ratio - 1.0) / 3.0))  # Normalize
            return 0
        
        # Check for cutoffs by comparing opposite edges
        top_cutoff = calculate_edge_cutoff(top_edge, bottom_edge)
        bottom_cutoff = calculate_edge_cutoff(bottom_edge, top_edge)
        left_cutoff = calculate_edge_cutoff(left_edge, right_edge)
        right_cutoff = calculate_edge_cutoff(right_edge, left_edge)
        
        # Average cutoff penalty (much more conservative)
        avg_cutoff = (top_cutoff + bottom_cutoff + left_cutoff + right_cutoff) / 4
        cutoff_penalty = min(0.4, avg_cutoff)  # Max 40% penalty instead of 80%
        
        # Check aspect ratio (faces that are too wide/narrow might be cropped)
        aspect_ratio = w / h
        aspect_penalty = 0.0
        
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # Very extreme ratios
            aspect_penalty = 0.3
        elif aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Moderately extreme
            aspect_penalty = 0.15
        
        # Check for reasonable face structure (eyes in upper half, etc.)
        structure_bonus = 0.0
        upper_half = gray[:h//2, :]
        lower_half = gray[h//2:, :]
        
        # Upper half should generally be different from lower half in a complete face
        if upper_half.size > 0 and lower_half.size > 0:
            upper_std = np.std(upper_half)
            lower_std = np.std(lower_half)
            if upper_std > 5 and lower_std > 5:  # Both regions have variation
                structure_bonus = 0.2
        
        # Final completeness score (much more generous)
        completeness_score = 0.8 - cutoff_penalty - aspect_penalty + structure_bonus
        
        return max(0.2, min(1.0, completeness_score))  # Minimum 20% instead of 5%
        
    except Exception as e:
        logger.warning(f"Error in completeness detection: {e}")
        return 0.6  # More generous fallback

def compute_face_size_score(face_crop: np.ndarray) -> float:
    """
    FIXED: Size score adjusted for thumbnails
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        h, w = face_crop.shape[:2]
        face_area = h * w
        
        # Adjusted thresholds for thumbnails
        tiny_threshold = 20 * 20      # 20x20 - very small
        small_threshold = 40 * 40     # 40x40 - small but usable  
        medium_threshold = 80 * 80    # 80x80 - good size
        large_threshold = 150 * 150   # 150x150 - large
        
        if face_area < tiny_threshold:
            return 0.1
        elif face_area < small_threshold:
            return 0.3 + _normalize(face_area, tiny_threshold, small_threshold) * 0.3
        elif face_area < medium_threshold:
            return 0.6 + _normalize(face_area, small_threshold, medium_threshold) * 0.25
        elif face_area < large_threshold:
            return 0.85 + _normalize(face_area, medium_threshold, large_threshold) * 0.15
        else:
            return 1.0
            
    except Exception:
        return 0.0

def compute_sharpness_score(face_crop: np.ndarray) -> float:
    """
    Improved sharpness detection optimized for small thumbnails
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
        laplacian_score = _normalize(laplacian_var, 20, 800)  # Lower thresholds
        
        # Method 2: Sobel gradient (works well on small images)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_score = _normalize(np.mean(sobel_magnitude), 5, 25)  # Lower thresholds
        
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
        contrast_score = _normalize(contrast, 10, 50)  # Adjusted for smaller images
        
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

def compute_face_visibility_score(face_crop: np.ndarray) -> float:
    """
    Comprehensive visibility check adjusted for thumbnails
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        # Get frontal score
        frontal_score = compute_frontal_face_score(face_crop)
        
        # Basic structure check
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check if image has reasonable variation (not just uniform color)
        std_dev = np.std(gray)
        variation_score = min(1.0, _normalize(std_dev, 8, 40))
        
        # Check aspect ratio reasonableness
        aspect_ratio = w / h
        if 0.5 <= aspect_ratio <= 2.0:
            aspect_score = 1.0
        else:
            aspect_score = 0.7
        
        # Combine factors (more lenient)
        visibility_score = (
            0.5 * frontal_score +      # Frontal vs profile  
            0.3 * variation_score +    # Has detail/variation
            0.2 * aspect_score         # Reasonable shape
        )
        
        return max(0.2, min(1.0, visibility_score))  # Minimum 20%
        
    except Exception as e:
        logger.warning(f"Error in visibility analysis: {e}")
        return 0.5

def compute_face_quality_from_bytes(face_thumb_bytes: bytes) -> float:
    """
    ENHANCED: Face quality computation with InsightFace detection
    Returns 0-1 where 1 is highest quality
    
    NEW LOGIC:
    - If 0 faces detected: return 0.0
    - If >1 faces detected: return 0.0  
    - If exactly 1 face detected: proceed with quality analysis
    - If face detection fails: fallback to original quality analysis (with penalty)
    """
    if not face_thumb_bytes:
        return 0.0

    # Convert bytes to image
    face_crop = bytes_to_cv_image(face_thumb_bytes)
    if face_crop is None:
        return 0.0

    # Log image dimensions for debugging
    h, w = face_crop.shape[:2]
    logger.debug(f"Processing face crop: {w}x{h}")

    # NEW: Face detection check using InsightFace
    face_count = detect_faces_count(face_crop)
    
    if face_count == 0:
        logger.debug("No faces detected - returning 0.0")
        return 0.0
    elif face_count > 1:
        logger.debug(f"Multiple faces detected ({face_count}) - returning 0.0")
        return 0.0
    elif face_count == -1:
        # Face detection failed - proceed with original analysis but with penalty
        logger.warning("Face detection failed - proceeding with penalty")
        detection_penalty = 0.3  # 30% penalty for failed detection
    else:
        # Exactly 1 face detected - no penalty
        logger.debug("Exactly 1 face detected - proceeding with full analysis")
        detection_penalty = 0.0

    # Component scores (original logic)
    sharpness_score = compute_sharpness_score(face_crop)
    brightness_score = compute_brightness_contrast_score(face_crop)
    visibility_score = compute_face_visibility_score(face_crop)
    completeness_score = compute_face_completeness_score(face_crop)
    size_score = compute_face_size_score(face_crop)
    
    # Log individual scores for debugging
    logger.debug(f"Score breakdown - Sharpness: {sharpness_score:.3f}, "
                f"Brightness: {brightness_score:.3f}, Visibility: {visibility_score:.3f}, "
                f"Completeness: {completeness_score:.3f}, Size: {size_score:.3f}")
    
    # Base quality calculation (same as before)
    base_quality_score = (
        0.25 * sharpness_score +      # Image quality
        0.25 * visibility_score +     # Frontal pose, visible features
        0.25 * completeness_score +   # Not cut off
        0.15 * brightness_score +     # Good lighting
        0.10 * size_score             # Adequate size
    )
    
    # Original penalties (same as before)
    penalties = 0.0
    
    # Only severe penalties for extremely poor conditions
    if completeness_score < 0.2:  # Only very incomplete faces
        penalties += 0.3  # Reduced from 0.7
    elif completeness_score < 0.3:
        penalties += 0.15  # Reduced from 0.4
    
    if visibility_score < 0.2:  # Only very poor visibility
        penalties += 0.2  # Reduced from 0.5
    
    if sharpness_score < 0.1:  # Only extremely blurry
        penalties += 0.2  # Reduced from 0.3
    
    if brightness_score < 0.1:  # Only extremely dark
        penalties += 0.15  # Reduced from 0.25
    
    # Apply penalties including detection penalty
    total_penalties = penalties + detection_penalty
    final_score = max(0.05, base_quality_score - total_penalties)  # Minimum 5%
    
    # Bonus for good complete faces (only if face detection succeeded)
    if detection_penalty == 0.0:  # Only if exactly 1 face was detected
        if completeness_score > 0.7 and visibility_score > 0.6:
            final_score = min(1.0, final_score + 0.1)
        
        # Special bonus for smaller but very complete faces
        if size_score > 0.3 and completeness_score > 0.8:
            final_score = min(1.0, final_score + 0.05)
    
    logger.debug(f"Final score: {final_score:.3f} (base: {base_quality_score:.3f}, "
                f"penalties: {penalties:.3f}, detection_penalty: {detection_penalty:.3f})")
    
    return float(min(max(final_score, 0.0), 1.0))  # Ensure 0-100% range

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

def process_single_face(face_data: Tuple[str, bytes]) -> Tuple[str, float]:
    """
    Process a single face and return its ID and quality score
    """
    face_id, face_thumb_bytes = face_data
    
    try:
        quality_score = compute_face_quality_from_bytes(face_thumb_bytes)
        logger.debug(f"Face {face_id}: quality score = {quality_score:.3f}")
        return face_id, quality_score
    except Exception as e:
        logger.error(f"Error processing face {face_id}: {e}")
        return face_id, 0.0  # Return 0 for failed faces

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
    # Initialize InsightFace
    if not initialize_face_analysis():
        logger.error("Failed to initialize face analysis. Exiting.")
        return
    
    processor = FaceQualityProcessor()
    
    # Get initial count
    total_unprocessed = processor.get_unprocessed_count()
    logger.info(f"Starting processing of {total_unprocessed} unprocessed faces")
    
    if total_unprocessed == 0:
        logger.info("No unprocessed faces found")
        return
    
    processed_count = 0
    batch_count = 0
    zero_score_count = 0  # Track faces with 0 score (0 or >1 faces detected)
    
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
        logger.info(f"  Zero quality (0/multiple faces): {zero_count}")
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
    logger.info(f"Total faces with 0 score (0 or multiple faces detected): {zero_score_count}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise