import os
import cv2
import torch
import psycopg2
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple
import logging
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import concurrent.futures
import numpy as np
import open_clip
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase (only if not already initialized)
try:
    firebase_admin.get_app()
    print("Firebase app already initialized")
except ValueError:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "gallery-585ee.firebasestorage.app"
    })

@dataclass
class Config:
    BATCH_SIZE: int = 50
    PARALLEL_LIMIT: int = 2
    
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

class ImageEmbeddingGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize OpenCLIP model (same as main script)
        try:
            model_name = "ViT-B-32"
            pretrained = "laion2b_s34b_b79k"

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device
            )
            logger.info("OpenCLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenCLIP model: {e}")
            raise
            
        # Initialize Qdrant client
        try:
            self.qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            logger.info("Qdrant client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def setup_image_collection(self, collection_name: str) -> None:
        """Setup Qdrant image collection"""
        try:
            image_collection_name = f"image_{collection_name}"
            if not self.qdrant.collection_exists(image_collection_name):
                self.qdrant.create_collection(
                    collection_name=image_collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
            logger.info(f"Image collection {image_collection_name} setup completed")
        except Exception as e:
            logger.error(f"Failed to setup image collection {collection_name}: {e}")
            raise

    def extract_full_image_embedding(self, image_input) -> torch.Tensor:
        """Extract complete image embeddings using OpenCLIP model"""
        try:
            img_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb[0]
        except Exception as e:
            logger.error(f"Failed to extract full image embedding: {e}")
            raise

    def read_image_from_firebase(self, image_id: str):
        """Read image from Firebase storage"""
        try:
            bucket = storage.bucket()
            blob = bucket.blob("compressed_" + str(image_id))

            img_bytes = blob.download_as_bytes()
            image_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            return img
        except Exception as e:
            logger.error(f"Failed to read image {image_id} from Firebase: {e}")
            return None

    def update_similar_image_id(self, image_id: str) -> bool:
        """Update similar_image_id to -1 for the processed image"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE images SET similar_image_id = %s WHERE id = %s",
                        ("-", image_id)
                    )
                    conn.commit()
                    logger.debug(f"Updated similar_image_id to -1 for image {image_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to update similar_image_id for image {image_id}: {e}")
            return False

    def get_images_without_embeddings(self, group_id: str, batch_size: int) -> List[str]:
        """Get images that don't have embeddings in the image collection yet"""
        try:
            # First get all images from the group
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM images WHERE group_id = %s AND similar_image_id is null LIMIT %s", 
                        (group_id, batch_size)
                    )
                    all_images = [str(row[0]) for row in cur.fetchall()]

            if not all_images:
                return []

            # Check which ones already have embeddings in Qdrant
            image_collection_name = f"image_{group_id}"
            
            if not self.qdrant.collection_exists(image_collection_name):
                # If collection doesn't exist, all images need embeddings
                return all_images

            # Get existing points from Qdrant
            try:
                existing_points = self.qdrant.scroll(
                    collection_name=image_collection_name,
                    limit=len(all_images) * 2  # Get more than enough
                )
                existing_ids = {point.id for point in existing_points[0]}
            except Exception as e:
                logger.warning(f"Could not check existing embeddings: {e}")
                existing_ids = set()

            # Return images that don't have embeddings
            missing_embeddings = [img_id for img_id in all_images if img_id not in existing_ids]
            logger.info(f"Found {len(missing_embeddings)} images without embeddings out of {len(all_images)} total")
            
            return missing_embeddings

        except Exception as e:
            logger.error(f"Failed to get images without embeddings: {e}")
            return []

    def process_single_image(self, image_id: str, group_id: str) -> bool:
        """Process a single image to generate embedding and update database"""
        try:
            # Load image from Firebase
            img = self.read_image_from_firebase(image_id)
            if img is None:
                logger.warning(f"Failed to load image {image_id}")
                return False

            # Convert to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Extract image embedding
            image_emb = self.extract_full_image_embedding(pil_img)
            
            # Store in Qdrant
            image_collection_name = f"image_{group_id}"
            self.qdrant.upsert(
                collection_name=image_collection_name,
                points=[
                    PointStruct(
                        id=str(image_id),  # Use image_id as point_id
                        vector=image_emb.cpu().tolist(),
                        payload={
                            "image_id": image_id,
                            "group_id": group_id,
                        }
                    )
                ]
            )
            
            # Update database: set similar_image_id to -1
            db_update_success = self.update_similar_image_id(image_id)
            if not db_update_success:
                logger.warning(f"Failed to update database for image {image_id}, but embedding was stored")
            
            logger.info(f"Successfully processed image {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process image {image_id}: {e}")
            return False

    def process_images_batch(self, image_ids: List[str], group_id: str) -> int:
        """Process a batch of images with controlled parallelism"""
        logger.info(f"Processing batch of {len(image_ids)} images for group {group_id}")
        
        successful_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.PARALLEL_LIMIT) as executor:
            futures = {
                executor.submit(self.process_single_image, img_id, group_id): img_id
                for img_id in image_ids
            }
            
            for future in concurrent.futures.as_completed(futures):
                img_id = futures[future]
                try:
                    success = future.result(timeout=60)
                    if success:
                        successful_count += 1
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout processing image {img_id}")
                except Exception as e:
                    logger.error(f"Error processing image {img_id}: {e}")
        
        logger.info(f"Successfully processed {successful_count}/{len(image_ids)} images")
        return successful_count

    def process_group(self, group_id: str) -> None:
        """Process all images in a group that don't have embeddings"""
        try:
            # Setup collection first
            self.setup_image_collection(group_id)
            
            logger.info(f"Starting image embedding generation for group {group_id}")
            
            total_processed = 0
            
            while True:
                # Get next batch of images without embeddings
                image_batch = self.get_images_without_embeddings(group_id, config.BATCH_SIZE)
                
                if not image_batch:
                    logger.info(f"No more images to process for group {group_id}")
                    break
                
                # Process the batch
                successful_count = self.process_images_batch(image_batch, group_id)
                total_processed += successful_count
                
                logger.info(f"Group {group_id}: Processed {total_processed} images so far")
                
                # If we got less than batch size, we're done
                if len(image_batch) < config.BATCH_SIZE:
                    break
            
            logger.info(f"Completed group {group_id}: {total_processed} total images processed")
            
        except Exception as e:
            logger.error(f"Failed to process group {group_id}: {e}")
            raise

    def get_all_groups(self) -> List[str]:
        """Get all groups that have processed images"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT group_id FROM images WHERE group_id = 8"
                    )
                    groups = [str(row[0]) for row in cur.fetchall()]
            
            logger.info(f"Found {len(groups)} groups with processed images")
            return groups
            
        except Exception as e:
            logger.error(f"Failed to get groups: {e}")
            return []

    def process_all_groups(self) -> None:
        """Process all groups to generate missing image embeddings"""
        groups = self.get_all_groups()
        
        if not groups:
            logger.info("No groups found to process")
            return
        
        logger.info(f"Starting image embedding generation for {len(groups)} groups")
        
        for i, group_id in enumerate(groups, 1):
            try:
                logger.info(f"Processing group {group_id} ({i}/{len(groups)})")
                self.process_group(group_id)
            except Exception as e:
                logger.error(f"Failed to process group {group_id}, continuing with next: {e}")
                continue
        
        logger.info("Completed image embedding generation for all groups")

def main():
    """Main execution function"""
    logger.info("Initializing Image Embedding Generator...")
    generator = ImageEmbeddingGenerator()
    
    try:
        # Process all groups
        generator.process_all_groups()
        logger.info("Image embedding generation completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()