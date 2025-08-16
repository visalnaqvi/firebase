import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import psycopg2
from psycopg2.extras import DictCursor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Face:
    """Face data structure"""
    id: int
    person_id: int
    group_id: int
    quality_score: float
    image_id: int

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="admin"
    )

class FaceCentroidGenerator:
    """Generate and store person face centroids in Qdrant"""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """
        Initialize the centroid generator
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
    def get_faces_by_group(self , group_id) -> Dict[str, List[Face]]:
        """
        Retrieve all faces grouped by group_id from database
        
        Returns:
            Dictionary mapping group_id to list of Face objects
        """
        faces_by_group = defaultdict(list)
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=DictCursor)
                
            query = """
            SELECT id, person_id, group_id, quality_score, image_id
            FROM faces where group_id = %s
            ORDER BY quality_score DESC
            """
            
            cursor.execute(query  , group_id)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            for row in rows:
                face = Face(
                    id=row[0],
                    person_id=row[1],
                    group_id=row[2],
                    quality_score=row[3],
                    image_id=row[4]
                )
                faces_by_group[face.group_id].append(face)
                    
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
            
        logger.info(f"Retrieved {sum(len(faces) for faces in faces_by_group.values())} faces from {len(faces_by_group)} groups")
        return dict(faces_by_group)
    
    def select_top_quality_faces(self, faces: List[Face], top_k: int = 5) -> List[Face]:
        """
        Select top K quality faces for each person
        
        Args:
            faces: List of Face objects for a person
            top_k: Number of top quality faces to select
            
        Returns:
            List of top quality Face objects
        """
        # Sort faces by quality score in descending order and take top_k
        sorted_faces = sorted(faces, key=lambda f: f.quality_score, reverse=True)
        return sorted_faces[:top_k]
    
    def get_face_embeddings(self, face_ids: List[int], collection_name: str) -> Dict[int, np.ndarray]:
        embeddings = {}
        try:
            points = self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=face_ids,
                with_vectors=True
            )
            
            for point in points:
                vec = point.vector
                if vec is not None:
                    # If named vector, take the first value
                    if isinstance(vec, dict):
                        vec = next(iter(vec.values()))
                    embeddings[point.id] = np.array(vec, dtype=np.float32)
                else:
                    logger.warning(f"No vector found for face ID {point.id}")
                        
        except Exception as e:
            logger.error(f"Error retrieving embeddings from Qdrant: {e}")
            raise
                
        logger.info(f"Retrieved {len(embeddings)} embeddings from collection {collection_name}")
        return embeddings
    
    def compute_centroid(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute centroid of face embeddings
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Centroid embedding vector
        """
        if not embeddings:
            raise ValueError("Cannot compute centroid of empty embedding list")
            
        # Stack embeddings and compute mean
        embedding_matrix = np.stack(embeddings)
        centroid = np.mean(embedding_matrix, axis=0)
        
        # Normalize the centroid (optional, depends on your embedding space)
        centroid = centroid / np.linalg.norm(centroid)
        
        return centroid
    
    def create_centroid_collection(self, collection_name: str, vector_size: int):
        """
        Create Qdrant collection for person centroids
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors
        """
        try:
            # Check if collection already exists
            collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            if collection_name in existing_names:
                logger.info(f"Collection {collection_name} already exists")
                self.qdrant_client.delete_collection(collection_name=collection_name)
                return
                
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise
    
    def store_centroids(self, collection_name: str, centroids: Dict[int, np.ndarray]):
        """
        Store person centroids in Qdrant collection
        
        Args:
            collection_name: Name of the Qdrant collection
            centroids: Dictionary mapping person_id to centroid embedding
        """
        try:
            points = []
            for person_id, centroid in centroids.items():
                point = PointStruct(
                    id=person_id,
                    vector=centroid.tolist(),
                    payload={"person_id": person_id}
                )
                points.append(point)
            
            # Batch upload points
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(points)} centroids in collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Error storing centroids: {e}")
            raise
    
    def process_group(self, group_id: int, faces: List[Face], 
                     top_k: int = 5) -> Dict[int, np.ndarray]:
        """
        Process faces for a specific group and generate person centroids
        
        Args:
            group_id: Group ID to process
            faces: List of Face objects for this group
            top_k: Number of top quality faces to use for centroid
            
        Returns:
            Dictionary mapping person_id to centroid embedding
        """
        logger.info(f"Processing group {group_id} with {len(faces)} faces")
        
        # The source collection is named after the group_id
        source_collection = str(group_id)
        
        # Group faces by person_id
        faces_by_person = defaultdict(list)
        for face in faces:
            faces_by_person[face.person_id].append(face)
        
        centroids = {}
        
        for person_id, person_faces in faces_by_person.items():
            logger.info(f"Processing person {person_id} with {len(person_faces)} faces")
            
            # Select top quality faces
            top_faces = self.select_top_quality_faces(person_faces, top_k)
            face_ids = [face.id for face in top_faces]
            
            logger.info(f"Selected {len(top_faces)} top quality faces for person {person_id}")
            
            # Get embeddings from Qdrant - using group_id as collection name
            embeddings_dict = self.get_face_embeddings(face_ids, source_collection)
            
            if len(embeddings_dict) < len(face_ids):
                logger.warning(f"Only found {len(embeddings_dict)}/{len(face_ids)} embeddings for person {person_id}")
            
            if embeddings_dict:
                # Compute centroid
                embeddings_list = list(embeddings_dict.values())
                centroid = self.compute_centroid(embeddings_list)
                centroids[person_id] = centroid
                
                logger.info(f"Generated centroid for person {person_id} using {len(embeddings_list)} embeddings")
            else:
                logger.warning(f"No embeddings found for person {person_id}, skipping centroid generation")
        
        return centroids
    def mark_group_processed(self , group_id) -> None:
        """Mark group_id as processed and clear image_byte"""
        if not group_id:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                            UPDATE groups
                            SET status = 'cooling',
                    last_processed_at = NOW(),
                            last_processed_step = 'centroid'
                            WHERE id = %s AND status = 'warmed'
                        """
                cur.execute(query, (group_id,))
                conn.commit()
                print(f"Marked {group_id} group_id as processed")
    def generate_all_centroids(self, top_k: int = 5):
        """
        Generate centroids for all groups and store them in respective collections
        
        Args:
            top_k: Number of top quality faces to use for centroid
        """
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        cursor.execute("SELECT id FROM groups WHERE status = 'warmed' order by last_processed_at")
        groups = cursor.fetchall()
        cursor.close()
        conn.close()
        # Get all faces grouped by group_id
        group_ids = [row["id"] for row in groups]
        
        for group_id in group_ids:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing Group {group_id}")
            logger.info(f"{'='*50}")
            
            try:
                faces_by_group = self.get_faces_by_group(group_id)
                for group_id, faces in faces_by_group:
                    # Process faces for this group
                    centroids = self.process_group(group_id, faces, top_k)
                    
                    if not centroids:
                        logger.warning(f"No centroids generated for group {group_id}")
                        continue
                    
                    # Create collection for this group
                    collection_name = f"person_centroid_{group_id}"
                    
                    # Get vector size from first centroid
                    vector_size = len(next(iter(centroids.values())))
                    self.create_centroid_collection(collection_name, vector_size)
                    
                    # Store centroids
                    self.store_centroids(collection_name, centroids)
                    
                    logger.info(f"Successfully processed group {group_id}: {len(centroids)} person centroids stored")
                    self.mark_group_processed(group_id)
            except Exception as e:
                logger.error(f"Error processing group {group_id}: {e}")
                continue
        
        logger.info("\n" + "="*50)
        logger.info("Centroid generation completed for all groups")
        logger.info("="*50)

def main():
    """Main function to run the centroid generation process"""
    
    # Configuration
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    TOP_K_FACES = 5  # Number of top quality faces to use for centroid
    
    try:
        # Initialize the centroid generator
        generator = FaceCentroidGenerator(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT
        )
        
        # Generate centroids for all groups
        generator.generate_all_centroids(
            top_k=TOP_K_FACES
        )
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise

if __name__ == "__main__":
    main()