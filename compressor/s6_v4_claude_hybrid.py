import os
import numpy as np
from scipy.spatial.distance import cosine
import torch
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import DictCursor
import uuid
from collections import defaultdict
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingData:
    face_embedding: np.ndarray
    cloth_embedding: Optional[torch.Tensor]
    face_quality: float
    cloth_quality: float
    person_id: Optional[str]
    cloth_ids: Set[str]

@dataclass
class MatchCandidate:
    id: str
    face_score: float
    cloth_score: float
    person_id: Optional[str]
    cloth_ids: Set[str]
    confidence: float

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="admin"
    )

@contextmanager
def database_transaction():
    """Context manager for atomic database operations"""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
        logger.info("✅ Database transaction committed successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Database transaction rolled back: {e}")
        raise
    finally:
        conn.close()

class EnhancedFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        self.qdrant = QdrantClient(host=host, port=port)
        # Face match threshold - must be >= 0.7
        self.face_match_threshold = 0.7
        # Cloth match thresholds - face >= 0.4 AND cloth >= 0.83
        self.cloth_face_threshold = 0.4
        self.cloth_match_threshold = 0.85
        self.high_confidence_threshold = 0.8
        self.min_embedding_norm = 0.1
        self.max_candidates = 100

    def validate_embedding(self, embedding: np.ndarray, embedding_type: str = "face") -> Tuple[bool, float]:
        """Validate embedding quality and return quality score"""
        if embedding is None or len(embedding) == 0:
            return False, 0.0
        
        # Check for invalid values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.warning(f"⚠️ Invalid values in {embedding_type} embedding")
            return False, 0.0
        
        # Check embedding magnitude
        norm = np.linalg.norm(embedding)
        if norm < self.min_embedding_norm:
            logger.warning(f"⚠️ {embedding_type} embedding norm too small: {norm}")
            return False, 0.0
        
        # Quality score based on norm and variance
        variance = np.var(embedding)
        quality_score = min(1.0, (norm / 10.0) * (variance / 0.1))
        
        return True, max(0.1, min(1.0, quality_score))

    def adaptive_threshold(self, base_threshold: float, quality: float, context_factor: float = 1.0) -> float:
        """Compute adaptive threshold based on embedding quality"""
        # Higher quality allows stricter threshold, lower quality needs more permissive
        quality_factor = 0.7 + 0.6 * quality  # Range: 0.7 to 1.3
        adaptive = base_threshold * quality_factor * context_factor
        return max(0.2, min(0.95, adaptive))

    def get_unassigned_faces_batch(self, group_id: str, limit: int = 100) -> List[str]:
        """Get batch of unassigned faces with database lock"""
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        try:
            # Use SELECT FOR UPDATE to prevent race conditions
            cursor.execute("""
                SELECT id FROM faces 
                WHERE group_id = %s AND person_id IS NULL 
                ORDER BY id  -- Consistent ordering
                LIMIT %s 
                FOR UPDATE SKIP LOCKED
            """, (group_id, limit))
            
            rows = cursor.fetchall()
            face_ids = [row["id"] for row in rows]
            logger.info(f"🔃 Locked {len(face_ids)} unassigned faces for processing")
            return face_ids
            
        finally:
            cursor.close()
            conn.close()

    def get_batch_embeddings(self, group_id: str, face_ids: List[str]) -> Dict[str, EmbeddingData]:
        """Get embeddings for batch of faces with validation"""
        if not face_ids:
            return {}
        
        try:
            points = self.qdrant.retrieve(
                collection_name=group_id,
                ids=face_ids,
                with_payload=True,
                with_vectors=True
            )
            
            embeddings = {}
            for point in points:
                vectors = getattr(points[0], "vectors", None) or getattr(points[0], "vector", None)
            
                
                face_vec = np.array(vectors.get("face", []))
                cloth_vec = vectors.get("cloth")
                cloth_tensor = torch.tensor(cloth_vec) if cloth_vec else None
                
                # Validate embeddings
                face_valid, face_quality = self.validate_embedding(face_vec, "face")
                cloth_valid, cloth_quality = (True, 0.8) if cloth_tensor is None else self.validate_embedding(cloth_tensor.numpy(), "cloth")
                
                if face_valid:
                    embeddings[point.id] = EmbeddingData(
                        face_embedding=face_vec,
                        cloth_embedding=cloth_tensor,
                        face_quality=face_quality,
                        cloth_quality=cloth_quality,
                        person_id=point.payload.get('person_id') if point.payload else None,
                        cloth_ids=set(point.payload.get('cloth_ids', [])) if point.payload else set()
                    )
                else:
                    logger.warning(f"⚠️ Invalid embedding for face {point.id}")
            
            logger.info(f"✅ Retrieved {len(embeddings)} valid embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error retrieving batch embeddings: {e}")
            return {}

    def compute_batch_similarities(self, embeddings: Dict[str, EmbeddingData]) -> Dict[Tuple[str, str], MatchCandidate]:
        """Compute pairwise similarities with specific thresholds for face and cloth matches"""
        similarities = {}
        face_ids = list(embeddings.keys())
        
        for i, face_id1 in enumerate(face_ids):
            for j, face_id2 in enumerate(face_ids[i+1:], i+1):
                emb1 = embeddings[face_id1]
                emb2 = embeddings[face_id2]
                
                # Compute face similarity
                face_sim = 1 - cosine(emb1.face_embedding, emb2.face_embedding)
                
                # Compute cloth similarity if both have cloth embeddings
                cloth_sim = 0.0
                if emb1.cloth_embedding is not None and emb2.cloth_embedding is not None:
                    cloth_sim = float((emb1.cloth_embedding @ emb2.cloth_embedding).cpu())
                
                # Check if this pair qualifies as a match
                is_face_match = face_sim >= self.face_match_threshold  # >= 0.7
                is_cloth_match = (face_sim >= self.cloth_face_threshold and  # >= 0.4
                                cloth_sim >= self.cloth_match_threshold)     # >= 0.83
                
                # Only include if it meets one of our match criteria
                if is_face_match or is_cloth_match:
                    # Compute confidence score based on match type
                    if is_face_match and is_cloth_match:
                        # Both face and cloth match - highest confidence
                        confidence = min(1.0, 0.4 * face_sim + 0.6 * cloth_sim)
                        match_type = "face+cloth"
                    elif is_face_match:
                        # Face match only
                        confidence = face_sim * 0.9  # High confidence for face matches
                        match_type = "face"
                    else:
                        # Cloth match only (face >= 0.4, cloth >= 0.83)
                        confidence = 0.3 * face_sim + 0.7 * cloth_sim
                        match_type = "cloth"
                    
                    similarities[(face_id1, face_id2)] = MatchCandidate(
                        id=face_id2,
                        face_score=face_sim,
                        cloth_score=cloth_sim,
                        person_id=emb2.person_id,
                        cloth_ids=emb2.cloth_ids,
                        confidence=confidence
                    )
                    
                    logger.debug(f"Match found: {face_id1}-{face_id2} ({match_type}): "
                               f"face={face_sim:.3f}, cloth={cloth_sim:.3f}, conf={confidence:.3f}")
        
        logger.info(f"📊 Found {len(similarities)} valid matches using strict thresholds")
        return similarities

    def compute_match_confidence(self, face_score: float, cloth_score: float, 
                                face_quality: float, cloth_quality: float) -> float:
        """Compute comprehensive match confidence score"""
        # Base confidence from similarity scores
        face_conf = max(0, (face_score - 0.3) / 0.7)  # Normalize 0.3-1.0 to 0-1
        cloth_conf = max(0, (cloth_score - 0.5) / 0.5)  # Normalize 0.5-1.0 to 0-1
        
        # Weight by embedding quality
        weighted_face = face_conf * face_quality
        weighted_cloth = cloth_conf * cloth_quality
        
        # Combined confidence (prioritize face slightly)
        if cloth_score > 0:
            confidence = 0.7 * weighted_face + 0.3 * weighted_cloth
        else:
            confidence = weighted_face
        
        # Boost for high face similarity
        if face_score > 0.8:
            confidence *= 1.2
        
        # Boost for consistent face+cloth match
        if face_score > 0.6 and cloth_score > 0.7:
            confidence *= 1.3
        
        return min(1.0, confidence)

    def build_similarity_graph(self, embeddings: Dict[str, EmbeddingData], 
                             similarities: Dict[Tuple[str, str], MatchCandidate]) -> nx.Graph:
        """Build similarity graph for clustering using strict match criteria"""
        G = nx.Graph()
        
        # Add all faces as nodes
        for face_id, emb_data in embeddings.items():
            G.add_node(face_id, 
                      person_id=emb_data.person_id,
                      face_quality=emb_data.face_quality,
                      cloth_ids=emb_data.cloth_ids)
        
        # Add edges only for matches that meet our strict criteria
        for (face_id1, face_id2), candidate in similarities.items():
            # Determine match type for edge weight
            is_face_match = candidate.face_score >= self.face_match_threshold
            is_cloth_match = (candidate.face_score >= self.cloth_face_threshold and 
                            candidate.cloth_score >= self.cloth_match_threshold)
            
            # Use higher weight for stronger match types
            if is_face_match and is_cloth_match:
                edge_weight = candidate.confidence * 1.2  # Boost for dual match
            elif is_face_match:
                edge_weight = candidate.confidence * 1.1  # Boost for face match
            else:
                edge_weight = candidate.confidence  # Standard cloth match
            
            G.add_edge(face_id1, face_id2, 
                      weight=min(1.0, edge_weight),
                      face_score=candidate.face_score,
                      cloth_score=candidate.cloth_score,
                      match_type="face" if is_face_match else "cloth")
        
        logger.info(f"📈 Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges using strict thresholds")
        return G

    def cluster_faces_graph_based(self, G: nx.Graph) -> Dict[str, str]:
        """Use graph-based clustering to assign person IDs"""
        assignments = {}
        processed = set()
        
        # Find connected components
        components = list(nx.connected_components(G))
        logger.info(f"🔗 Found {len(components)} connected components")
        
        for component in components:
            if len(component) == 1:
                # Single face - check if already has person_id
                face_id = list(component)[0]
                existing_person_id = G.nodes[face_id].get('person_id')
                if existing_person_id:
                    assignments[face_id] = existing_person_id
                else:
                    assignments[face_id] = str(uuid.uuid4())
                processed.update(component)
                continue
            
            # Multiple faces in component - find best assignment
            component_list = list(component)
            person_id = self.resolve_component_assignment(G, component_list)
            
            for face_id in component_list:
                assignments[face_id] = person_id
            processed.update(component)
        
        # Handle any unprocessed faces
        for face_id in G.nodes():
            if face_id not in processed:
                existing_person_id = G.nodes[face_id].get('person_id')
                assignments[face_id] = existing_person_id or str(uuid.uuid4())
        
        return assignments

    def resolve_component_assignment(self, G: nx.Graph, component: List[str]) -> str:
        """Resolve person_id assignment for a connected component"""
        # Collect existing person_ids in component
        existing_persons = {}
        unassigned = []
        
        for face_id in component:
            person_id = G.nodes[face_id].get('person_id')
            if person_id:
                if person_id not in existing_persons:
                    existing_persons[person_id] = []
                existing_persons[person_id].append(face_id)
            else:
                unassigned.append(face_id)
        
        if not existing_persons:
            # All unassigned - create new person_id
            return str(uuid.uuid4())
        
        if len(existing_persons) == 1:
            # Single existing person - check confidence
            person_id = list(existing_persons.keys())[0]
            assigned_faces = existing_persons[person_id]
            
            # Find highest confidence connection between assigned and unassigned
            max_confidence = 0
            for assigned_face in assigned_faces:
                for unassigned_face in unassigned:
                    edge_data = G.get_edge_data(assigned_face, unassigned_face)
                    if edge_data:
                        max_confidence = max(max_confidence, edge_data['weight'])
            
            if max_confidence > self.high_confidence_threshold:
                return person_id
            else:
                # Low confidence - create new person but record similarity
                return str(uuid.uuid4())
        
        # Multiple existing persons - find best match
        best_person_id = None
        best_confidence = 0
        
        for person_id, assigned_faces in existing_persons.items():
            max_conf_for_person = 0
            for assigned_face in assigned_faces:
                for unassigned_face in unassigned:
                    edge_data = G.get_edge_data(assigned_face, unassigned_face)
                    if edge_data:
                        max_conf_for_person = max(max_conf_for_person, edge_data['weight'])
            
            if max_conf_for_person > best_confidence:
                best_confidence = max_conf_for_person
                best_person_id = person_id
        
        if best_confidence > self.high_confidence_threshold:
            return best_person_id
        else:
            return str(uuid.uuid4())

    def update_databases_atomic(self, group_id: str, assignments: Dict[str, str], 
                               cloth_updates: Dict[str, Set[str]]):
        """Atomically update both databases"""
        with database_transaction() as conn:
            cursor = conn.cursor()
            
            # Update PostgreSQL
            if assignments:
                update_data = [(person_id, face_id) for face_id, person_id in assignments.items()]
                cursor.executemany(
                    "UPDATE faces SET person_id = %s WHERE id = %s",
                    update_data
                )
                logger.info(f"📝 Updated {len(update_data)} faces in PostgreSQL")
            
            # Update Qdrant (this is not transactional with PostgreSQL, but we do our best)
            if cloth_updates:
                qdrant_points = []
                for face_id, cloth_ids in cloth_updates.items():
                    person_id = assignments.get(face_id)
                    p = {
                                "person_id": person_id,
                                "cloth_ids": list(cloth_ids)
                            }
                    if person_id:
                        self.qdrant.set_payload(
                            collection_name=group_id,
                            payload=p,
                            points= [face_id],
                        )
                        

    def track_similar_faces(self, group_id: str, G: nx.Graph, assignments: Dict[str, str]):
        """Track faces that are similar but assigned different person_ids"""
        similar_faces_data = []
        
        for edge in G.edges(data=True):
            face_id1, face_id2, edge_data = edge
            person_id1 = assignments.get(face_id1)
            person_id2 = assignments.get(face_id2)
            
            # If different person_ids but meet our match criteria, track it
            face_score = edge_data['face_score']
            cloth_score = edge_data['cloth_score']
            
            if person_id1 != person_id2:
                # Check if this was a valid match according to our criteria
                is_face_match = face_score >= self.face_match_threshold
                is_cloth_match = (face_score >= self.cloth_face_threshold and 
                                cloth_score >= self.cloth_match_threshold)
                
                if is_face_match or is_cloth_match:
                    similar_faces_data.append((group_id, face_id1, person_id2, edge_data['weight']))
                    similar_faces_data.append((group_id, face_id2, person_id1, edge_data['weight']))
        
        if similar_faces_data:
            with database_transaction() as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS similar_faces (
                        id SERIAL PRIMARY KEY,
                        group_id VARCHAR(255),
                        face_id VARCHAR(255),
                        similar_person_id VARCHAR(255),
                        confidence FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(group_id, face_id, similar_person_id)
                    )
                """)
                
                # Insert similar faces with confidence scores
                cursor.executemany("""
                    INSERT INTO similar_faces (group_id, face_id, similar_person_id, confidence) 
                    VALUES (%s, %s, %s, %s) 
                    ON CONFLICT (group_id, face_id, similar_person_id) DO NOTHING
                """, similar_faces_data)
                
                logger.info(f"🔗 Tracked {len(similar_faces_data)} similar face relationships with strict criteria")

    def process_face_batch(self, group_id: str, batch_size: int = 100):
        """Process batch of faces with enhanced accuracy"""
        logger.info(f"🚀 Processing enhanced batch for group {group_id}")
        start_time = time.time()
        
        # Step 1: Get batch of unassigned faces (with locking)
        unassigned_face_ids = self.get_unassigned_faces_batch(group_id, batch_size)
        if not unassigned_face_ids:
            logger.info("No unassigned faces found")
            return
        
        # Step 2: Get all embeddings with validation
        embeddings = self.get_batch_embeddings(group_id, unassigned_face_ids)
        if not embeddings:
            logger.warning("No valid embeddings found")
            return
        
        # Step 3: Compute pairwise similarities
        similarities = self.compute_batch_similarities(embeddings)
        
        # Step 4: Build similarity graph
        G = self.build_similarity_graph(embeddings, similarities)
        
        # Step 5: Cluster faces using graph-based approach
        assignments = self.cluster_faces_graph_based(G)
        
        # Step 6: Prepare cloth updates
        cloth_updates = {}
        for face_id, person_id in assignments.items():
            cloth_ids = embeddings[face_id].cloth_ids.copy()
            cloth_ids.add(person_id)
            cloth_updates[face_id] = cloth_ids
        
        # Step 7: Update databases atomically
        self.update_databases_atomic(group_id, assignments, cloth_updates)
        
        # Step 8: Track similar faces
        self.track_similar_faces(group_id, G, assignments)
        
        elapsed = time.time() - start_time
        logger.info(f"🎉 Batch complete! Processed {len(embeddings)} faces in {elapsed:.2f}s")
        
        # Performance metrics
        avg_edges_per_node = G.number_of_edges() / max(1, G.number_of_nodes())
        logger.info(f"📊 Graph density: {avg_edges_per_node:.2f} edges/node")

    def process_unassigned_faces(self, group_id: str, batch_size: int = 100):
        """Process all unassigned faces with enhanced algorithm"""
        logger.info(f"🚀 Starting enhanced face processing for group {group_id}")
        
        batch_count = 0
        while True:
            batch_count += 1
            logger.info(f"📦 Processing batch {batch_count}")
            
            # Check if there are faces to process
            unassigned_count = len(self.get_unassigned_faces_batch(group_id, 1))
            if unassigned_count == 0:
                logger.info("✅ All faces processed!")
                break
            
            try:
                self.process_face_batch(group_id, batch_size)
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_count}: {e}")
                break
            
            # Safety check to avoid infinite loops
            if batch_count > 1000:  # Adjust based on expected data size
                logger.warning("⚠️ Maximum batch count reached, stopping")
                break

    def process_all_groups(self, batch_size: int = 100):
        """Process all warmed groups with enhanced accuracy"""
        with database_transaction() as conn:
            cursor = conn.cursor(cursor_factory=DictCursor)
            cursor.execute("SELECT id FROM groups WHERE status = 'warmed'")
            groups = cursor.fetchall()
        
        group_ids = [row["id"] for row in groups]
        logger.info(f"📋 Found {len(group_ids)} groups to process")
        
        for i, group_id in enumerate(group_ids, 1):
            try:
                logger.info(f"🔄 Processing group {i}/{len(group_ids)}: {group_id}")
                self.process_unassigned_faces(group_id, batch_size)
                logger.info(f"✅ Completed group {group_id}")
            except Exception as e:
                logger.error(f"❌ Error processing group {group_id}: {e}")
                continue

# 🔧 Usage Example
if __name__ == "__main__":
    # Configure logging level
    logging.getLogger().setLevel(logging.INFO)
    
    grouper = EnhancedFaceGrouping()
    
    # Process all groups with enhanced accuracy
    grouper.process_all_groups(batch_size=100)
    
    # Or process a specific group
    # grouper.process_unassigned_faces("specific_group_id", batch_size=100)