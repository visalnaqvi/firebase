import psycopg2
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DB connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="admin"
    )

# Create table if not exists
def ensure_similar_faces_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        DROP TABLE similar_faces;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS similar_faces (
                    id SERIAL PRIMARY KEY,
                    group_id VARCHAR(255),
                    person_id VARCHAR(255),
                    similar_person_id VARCHAR(255));
    """)
    conn.commit()
    conn.close()
    logger.info("Ensured similar_faces table exists")

# Get distinct person_id, group_id pairs
def get_unique_persons():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT person_id, group_id
        FROM faces
        WHERE person_id IS NOT NULL
    """)
    results = cur.fetchall()
    conn.close()
    logger.info(f"Retrieved {len(results)} unique (person_id, group_id) pairs")
    return results

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

# Insert into similar_faces table
def insert_similar_faces(pairs):
    if not pairs:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    execute_values(cur, """
        INSERT INTO similar_faces (person_id, similar_person_id)
        VALUES %s
    """, pairs)
    conn.commit()
    conn.close()
    logger.info(f"Inserted {len(pairs)} similar face records")

def main():
    ensure_similar_faces_table()

    qdrant_client = QdrantClient(host="localhost", port=6333)

    persons = get_unique_persons()
    all_pairs = []

    for person_id, group_id in persons:
        collection_name = f"person_centroid_{group_id}"
        vector = get_person_vector(qdrant_client, collection_name, person_id)
        if vector is None:
            logger.warning(f"No vector found for person_id={person_id} in group_id={group_id}")
            continue

        similar_ids = find_similar(qdrant_client, collection_name, vector, person_id, threshold=0.5)
        for sim_id in similar_ids:
            all_pairs.append((person_id, sim_id))

    insert_similar_faces(all_pairs)
    logger.info("Similarity check completed")

if __name__ == "__main__":
    main()
