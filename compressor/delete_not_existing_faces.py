import psycopg2
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="admin"
    )

# Qdrant connection
qdrant = QdrantClient(host="localhost", port=6333)

def cleanup_faces():
    conn = get_db_connection()
    cur = conn.cursor()

    # Step 1: Get all IDs from faces table
    cur.execute("SELECT id FROM faces")
    face_ids = [row[0] for row in cur.fetchall()]
    logging.info(f"Found {len(face_ids)} records in faces table.")

    # Step 2: Get all point IDs from Qdrant collection "4"
    qdrant_ids = set()
    offset = None
    while True:
        points = qdrant.scroll(collection_name="4", limit=1000, offset=offset, with_vectors=False)
        qdrant_ids.update(str(p.id) for p in points[0])  # p.id may be int or str in Qdrant
        offset = points[1]
        if offset is None:
            break
    logging.info(f"Found {len(qdrant_ids)} point IDs in Qdrant collection '4'.")

    # Step 3: Find missing IDs
    missing_ids = [fid for fid in face_ids if str(fid) not in qdrant_ids]
    logging.info(f"{len(missing_ids)} records are missing in Qdrant and will be deleted.")

    if missing_ids:
        execute_values(
            cur,
            "DELETE FROM faces WHERE id IN %s",
            (tuple(missing_ids),)
        )
        conn.commit()
        logging.info(f"Deleted {len(missing_ids)} records from faces table.")

    cur.close()
    conn.close()
    logging.info("Cleanup complete.")

if __name__ == "__main__":
    cleanup_faces()
