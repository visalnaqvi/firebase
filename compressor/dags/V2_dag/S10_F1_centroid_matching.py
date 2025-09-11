import psycopg2
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest
import numpy as np
import logging
import argparse

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DB connection
def get_db_connection():
    return psycopg2.connect(
         host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )
def check_group_exists(group_id: int) -> bool:
    """Check if group exists and has warm status"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM groups WHERE id = %s AND status = 'cooling' and last_processed_step='centroid'", (group_id,))
            result = cur.fetchone()
            return result is not None
def mark_group_process_status(group_id , status) -> None:
            """Mark group_id as being processed"""
            if not group_id:
                return
                
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        UPDATE process_status
                        SET group_id = %s,
                        running = 'centroid_matching',
                        status = %s,
                        started_at = NOW()
                        WHERE id = 1
                    """
                    cur.execute(query, (group_id,status))
                    conn.commit()
# Get distinct person_id, group_id pairs
def get_unique_persons(group_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT person_id
        FROM faces
        WHERE person_id IS NOT NULL and group_id = %s
    """ , (group_id,))
    results = cur.fetchall()
    conn.close()
    logger.info(f"Retrieved {len(results)} unique (person_id, group_id) pairs")
    return [item[0] for item in results]

# Fetch vector for given person from Qdrant
def get_person_vector(qdrant_client, collection_name, person_id):
    try:
        points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[person_id],
            with_vectors=True
        )
        if points and points[0].vector is not None:
            vec = points[0].vector
            if isinstance(vec, dict):  # Named vector handling
                vec = next(iter(vec.values()))
            return np.array(vec, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error retrieving vector for person {person_id} from {collection_name}: {e}")
    return None

# Find similar candidates above threshold
def find_similar(qdrant_client, collection_name, vector, person_id, threshold, top_k=10):
    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=vector.tolist(),
            limit=top_k,
            score_threshold=threshold
        )
        similar_ids = []
        for r in results:
            if r.id != person_id and r.score >= threshold:
                similar_ids.append(r.id)
        return similar_ids
    except Exception as e:
        logger.error(f"Error searching similar persons for {person_id} in {collection_name}: {e}")
        return []
def mark_group_processed(group_id) -> None:
        """Mark group_id as processed and clear image_byte"""
        if not group_id:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                            UPDATE groups
                    set last_processed_at = NOW(),
                            last_processed_step = 'similarity'
                            WHERE id = %s AND status = 'cooling'
                        """
                cur.execute(query, (group_id,))
                conn.commit()
                print(f"Marked {group_id} group_id as processed")
# Insert into similar_faces table
def insert_similar_faces(pairs):
    if not pairs:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    execute_values(cur, """
        INSERT INTO similar_faces (person_id, similar_person_id, group_id)
        VALUES %s
    """, pairs)
    conn.commit()
    conn.close()
    logger.info(f"Inserted {len(pairs)} similar face records")

def main():
    parser = argparse.ArgumentParser(description="Process face quality for a specific group.")
    parser.add_argument("group_id", type=int, help="Group ID to process")
    args = parser.parse_args()

    group_id = args.group_id
    if not check_group_exists(group_id):
        print(f"Group {group_id} not found")
        raise
    qdrant_client = QdrantClient(host="localhost", port=6333)
    
    persons = get_unique_persons(group_id)
    all_pairs = []
    mark_group_process_status(group_id, 'healthy')
    try:
        for person_id in persons:
            collection_name = f"person_centroid_{group_id}"
            vector = get_person_vector(qdrant_client, collection_name, person_id)
            if vector is None:
                logger.warning(f"No vector found for person_id={person_id} in group_id={group_id}")
                continue

            similar_ids = find_similar(qdrant_client, collection_name, vector, person_id, threshold=0.5)
            for sim_id in similar_ids:
                all_pairs.append((person_id, sim_id,group_id ))

        insert_similar_faces(all_pairs)
        mark_group_processed(group_id)
        mark_group_process_status(group_id, 'done')
        logger.info(f"Similarity check completed for group {group_id}")
        logger.info("Similarity check completed")
    except Exception as e:
        logger.info(f"Similarity failed for group {group_id}")
        mark_group_process_status(group_id, 'failed')

if __name__ == "__main__":
    main()
