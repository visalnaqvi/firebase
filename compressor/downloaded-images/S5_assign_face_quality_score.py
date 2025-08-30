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

def compute_frontal_face_score(face_crop: np.ndarray) -> float:
    """
    Detect if face is frontal vs profile/side view
    Returns 0-1 where 1 means clear frontal face
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        if w < 20 or h < 20:
            return 0.1
        
        # Check left-right symmetry for frontal detection
        center_x = w // 2
        margin = w // 6  # Allow some margin for slight head turns
        
        left_region = gray[:, max(0, center_x - margin - w//3):center_x - margin]
        right_region = gray[:, center_x + margin:min(w, center_x + margin + w//3)]
        
        if left_region.size == 0 or right_region.size == 0:
            return 0.2
        
        # Resize to match
        min_width = min(left_region.shape[1], right_region.shape[1])
        min_height = min(left_region.shape[0], right_region.shape[0])
        
        if min_width < 10 or min_height < 10:
            return 0.2
        
        left_resized = cv2.resize(left_region, (min_width, min_height))
        right_resized = cv2.resize(cv2.flip(right_region, 1), (min_width, min_height))
        
        # Calculate correlation
        correlation = cv2.matchTemplate(
            left_resized.astype(np.float32), 
            right_resized.astype(np.float32), 
            cv2.TM_CCOEFF_NORMED
        )[0, 0]
        
        symmetry_score = max(0, correlation)
        
        # Check for presence of both eyes region (frontal indicator)
        # Look for dark regions in upper portion that could be eyes
        eye_region = gray[:h//2, :]  # Upper half
        eye_region_blur = cv2.GaussianBlur(eye_region, (5, 5), 0)
        
        # Find dark spots that could be eyes
        _, binary = cv2.threshold(eye_region_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count potential eye regions
        eye_candidates = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < (h * w * 0.1):  # Reasonable eye size
                eye_candidates += 1
        
        # Bonus for having 2 potential eye regions (frontal face indicator)
        eye_bonus = min(1.0, eye_candidates / 2.0)
        
        frontal_score = 0.7 * symmetry_score + 0.3 * eye_bonus
        return min(1.0, frontal_score)
        
    except Exception as e:
        logger.warning(f"Error in frontal face detection: {e}")
        return 0.3

def compute_face_completeness_score(face_crop: np.ndarray) -> float:
    """
    Check if the face is complete (not cut off at edges)
    Returns 0-1 where 1 means complete face
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check if face touches edges (indicating it might be cut off)
        edge_thickness = max(2, min(h, w) // 20)
        
        # Check all four edges for high gradient (indicating cut-off features)
        top_edge = gray[:edge_thickness, :]
        bottom_edge = gray[-edge_thickness:, :]
        left_edge = gray[:, :edge_thickness]
        right_edge = gray[:, -edge_thickness:]
        
        # Calculate edge gradients
        top_gradient = np.mean(np.abs(np.diff(top_edge, axis=0)))
        bottom_gradient = np.mean(np.abs(np.diff(bottom_edge, axis=0)))
        left_gradient = np.mean(np.abs(np.diff(left_edge, axis=1)))
        right_gradient = np.mean(np.abs(np.diff(right_edge, axis=1)))
        
        # High gradients at edges suggest cut-off features
        max_gradient = max(top_gradient, bottom_gradient, left_gradient, right_gradient)
        
        # Penalty for high edge gradients
        completeness_score = 1.0 - _normalize(max_gradient, 5, 25)
        
        # Additional check: ensure face doesn't occupy entire image (likely cropped)
        face_coverage = 1.0  # Assume face covers entire crop
        
        # If face covers >90% of image, it might be too tightly cropped
        if face_coverage > 0.9:
            completeness_score *= 0.8
        
        return completeness_score
        
    except Exception as e:
        logger.warning(f"Error in completeness detection: {e}")
        return 0.5

def compute_face_occlusion_score(face_crop: np.ndarray) -> float:
    """
    Improved occlusion detection focusing on facial feature visibility
    Returns 0-1 where 1 means no occlusion
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        if w < 10 or h < 10:
            return 0.1
        
        # Check for uniform regions (strong indicator of occlusion)
        std_dev = np.std(gray)
        if std_dev < 8:  # Very uniform, likely occluded
            return 0.1
        
        # Divide face into regions to check for occlusion patterns
        regions = []
        region_h, region_w = h // 3, w // 3
        
        for i in range(3):
            for j in range(3):
                y1, y2 = i * region_h, min((i + 1) * region_h, h)
                x1, x2 = j * region_w, min((j + 1) * region_w, w)
                region = gray[y1:y2, x1:x2]
                if region.size > 0:
                    regions.append(np.std(region))
        
        # Check for regions with very low variance (potential occlusion)
        low_variance_regions = sum(1 for std in regions if std < 10)
        occlusion_penalty = min(0.8, low_variance_regions / len(regions))
        
        # Check for large uniform areas using contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find large uniform contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = h * w
        large_uniform_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > total_area * 0.15:  # Large uniform regions
                large_uniform_area += area
        
        uniform_penalty = min(0.6, large_uniform_area / total_area)
        
        # Final occlusion score
        occlusion_score = 1.0 - max(occlusion_penalty, uniform_penalty)
        return max(0.1, occlusion_score)
        
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
        # Prefer brighter faces but not overexposed
        if mean_brightness < 30:  # Too dark
            brightness_score = mean_brightness / 30 * 0.3
        elif mean_brightness > 220:  # Too bright
            brightness_score = (255 - mean_brightness) / 35 * 0.5 + 0.5
        else:
            brightness_score = _normalize(mean_brightness, 60, 180)
        
        # Contrast check (standard deviation)
        contrast = np.std(gray)
        contrast_score = _normalize(contrast, 15, 60)
        
        # Dynamic range check
        min_val, max_val = np.min(gray), np.max(gray)
        dynamic_range = max_val - min_val
        range_score = _normalize(dynamic_range, 40, 150)
        
        # Histogram analysis for better contrast assessment
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.ravel() / hist.sum()
        
        # Penalize images with too much concentration in dark areas
        dark_concentration = np.sum(hist_normalized[:64])  # 0-63 (dark regions)
        if dark_concentration > 0.6:
            contrast_score *= 0.7
        
        # Combined score with adjusted weights
        combined_score = 0.4 * brightness_score + 0.4 * contrast_score + 0.2 * range_score
        return combined_score
        
    except Exception as e:
        logger.warning(f"Error in brightness/contrast analysis: {e}")
        return 0.0

def compute_sharpness_score(face_crop: np.ndarray) -> float:
    """
    Improved sharpness detection with bias toward frontal faces
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Method 1: Laplacian variance (good for general sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        laplacian_score = _normalize(laplacian_var, 50, 1500)
        
        # Method 2: Focus on center region (where face features should be sharpest)
        center_y, center_x = h // 2, w // 2
        center_region = gray[
            max(0, center_y - h//4):min(h, center_y + h//4),
            max(0, center_x - w//4):min(w, center_x + w//4)
        ]
        
        if center_region.size > 100:
            center_laplacian = cv2.Laplacian(center_region, cv2.CV_64F).var()
            center_score = _normalize(center_laplacian, 80, 2000)
        else:
            center_score = laplacian_score
        
        # Method 3: Sobel gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_score = _normalize(np.mean(sobel_magnitude), 8, 40)
        
        # Method 4: High frequency content (using FFT)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Calculate high frequency energy (sharpness indicator)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a mask for high frequencies (outer region)
        y, x = np.ogrid[:rows, :cols]
        mask = (x - ccol)**2 + (y - crow)**2 > (min(rows, cols) // 6)**2
        
        high_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)
        
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
            freq_score = _normalize(high_freq_ratio, 0.1, 0.4)
        else:
            freq_score = 0.0
        
        # Weighted combination favoring center region sharpness
        sharpness = (
            0.3 * laplacian_score + 
            0.4 * center_score + 
            0.2 * sobel_score + 
            0.1 * freq_score
        )
        
        return sharpness
        
    except Exception as e:
        logger.warning(f"Error in sharpness analysis: {e}")
        return 0.0

def compute_face_visibility_score(face_crop: np.ndarray) -> float:
    """
    Comprehensive check for face visibility and completeness
    Heavily penalizes profile views and occluded faces
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Frontal pose detection
        frontal_score = compute_frontal_face_score(face_crop)
        
        # 2. Feature presence detection using Haar cascades or simple heuristics
        # Check for eye regions in expected positions
        eye_region_score = 0.0
        upper_face = gray[:h//2, :]  # Upper half where eyes should be
        
        # Use simple template matching for eye-like dark regions
        eye_template_size = max(5, min(w, h) // 10)
        if eye_template_size > 0:
            # Create simple dark circle template for eyes
            template = np.ones((eye_template_size, eye_template_size), dtype=np.uint8) * 255
            cv2.circle(template, (eye_template_size//2, eye_template_size//2), 
                      eye_template_size//3, 0, -1)
            
            try:
                result = cv2.matchTemplate(upper_face, template, cv2.TM_CCOEFF_NORMED)
                max_val = np.max(result)
                eye_region_score = _normalize(max_val, 0.1, 0.6)
            except:
                eye_region_score = 0.3
        
        # 3. Edge analysis - faces touching borders are likely cropped
        edge_penalty = 0.0
        border_width = max(1, min(h, w) // 20)
        
        # Check if significant features are at edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels near borders
        top_edge_pixels = np.sum(edges[:border_width, :])
        bottom_edge_pixels = np.sum(edges[-border_width:, :])
        left_edge_pixels = np.sum(edges[:, :border_width])
        right_edge_pixels = np.sum(edges[:, -border_width:])
        
        total_edge_pixels = np.sum(edges)
        if total_edge_pixels > 0:
            border_edge_ratio = (top_edge_pixels + bottom_edge_pixels + 
                               left_edge_pixels + right_edge_pixels) / total_edge_pixels
            edge_penalty = min(0.5, border_edge_ratio)
        
        # 4. Aspect ratio check - faces should have reasonable proportions
        aspect_ratio = w / h
        if 0.6 <= aspect_ratio <= 1.4:  # Reasonable face aspect ratio
            aspect_score = 1.0
        else:
            aspect_score = 0.6
        
        # Combine all visibility factors
        visibility_score = (
            0.5 * frontal_score +        # Most important - frontal vs profile
            0.2 * eye_region_score +     # Eye visibility
            0.2 * (1.0 - edge_penalty) + # Not cut off
            0.1 * aspect_score           # Reasonable proportions
        )
        
        return max(0.0, min(1.0, visibility_score))
        
    except Exception as e:
        logger.warning(f"Error in visibility analysis: {e}")
        return 0.3

def compute_face_size_score(face_crop: np.ndarray) -> float:
    """
    Score based on face size - larger faces generally have more detail
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    
    try:
        h, w = face_crop.shape[:2]
        face_area = h * w
        
        # Updated size thresholds
        min_acceptable = 30 * 30    # 30x30 minimum acceptable
        min_good_area = 60 * 60     # 60x60 minimum good
        optimal_area = 120 * 120    # 120x120 optimal
        
        if face_area < min_acceptable:
            return 0.1  # Very low score for tiny faces
        elif face_area < min_good_area:
            return _normalize(face_area, min_acceptable, min_good_area) * 0.6
        elif face_area < optimal_area:
            return 0.6 + _normalize(face_area, min_good_area, optimal_area) * 0.4
        else:
            return 1.0
            
    except Exception:
        return 0.0

def compute_face_quality_from_bytes(face_thumb_bytes: bytes) -> float:
    """
    Improved face quality computation with better frontal face preference
    Returns 0-1 where 1 is highest quality
    """
    if not face_thumb_bytes:
        return 0.0

    # Convert bytes to image
    face_crop = bytes_to_cv_image(face_thumb_bytes)
    if face_crop is None:
        return 0.0

    # Component scores with improved weighting
    sharpness_score = compute_sharpness_score(face_crop)
    brightness_score = compute_brightness_contrast_score(face_crop)
    visibility_score = compute_face_visibility_score(face_crop)  # New comprehensive check
    size_score = compute_face_size_score(face_crop)
    
    # Log individual scores for debugging
    logger.debug(f"Score breakdown - Sharpness: {sharpness_score:.3f}, "
                f"Brightness: {brightness_score:.3f}, Visibility: {visibility_score:.3f}, "
                f"Size: {size_score:.3f}")
    
    # Revised weighted combination emphasizing visibility (frontal pose)
    base_quality_score = (
        0.25 * sharpness_score +      # Important but not everything
        0.40 * visibility_score +     # MOST CRITICAL - frontal, complete, unoccluded
        0.20 * brightness_score +     # Good lighting
        0.15 * size_score             # Adequate size
    )
    
    # Severe penalties for poor conditions
    penalties = 0.0
    
    # MAJOR penalty for non-frontal or heavily occluded faces
    if visibility_score < 0.4:
        penalties += 0.6  # Massive penalty
    elif visibility_score < 0.6:
        penalties += 0.3  # Moderate penalty
    
    # Heavy penalty for very blurry faces
    if sharpness_score < 0.2:
        penalties += 0.4
    elif sharpness_score < 0.4:
        penalties += 0.2
    
    # Penalty for very dark faces
    if brightness_score < 0.2:
        penalties += 0.3
    
    # Penalty for very small faces
    if size_score < 0.3:
        penalties += 0.25
    
    # Apply penalties
    final_score = max(0.0, base_quality_score - penalties)
    
    # Bonus for excellent frontal faces (even if slightly blurry)
    if visibility_score > 0.8 and sharpness_score > 0.3:
        final_score = min(1.0, final_score + 0.1)
    
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