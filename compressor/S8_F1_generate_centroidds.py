import psycopg2
from psycopg2.extras import DictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==== CONFIG ====
PG_CONN = os.getenv("DATABASE_URL")  # PostgreSQL connection string
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # if using cloud Qdrant

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="admin"
    )


def get_warmed_groups():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT id AS group_id FROM groups WHERE status = 'warmed'")
    result = [row["group_id"] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return result


def get_best_faces_for_group(group_id):
    """Get highest quality_score face per person_id for a group"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("""
        SELECT DISTINCT ON (person_id)
            id, person_id, quality_score, face_thumb_bytes
        FROM faces
        WHERE group_id = %s
        ORDER BY person_id, quality_score DESC NULLS LAST, id
    """, (group_id,))
    res = cur.fetchall()
    cur.close()
    conn.close()
    return res



def ensure_person_centroid_collection(group_id, vector_size):
    """Ensure person_centroid_<group_id> exists"""
    collection_name = f"person_centroid_{group_id}"
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name=collection_name)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    logging.info(f"Created collection {collection_name}")
    return collection_name


def process_group(group_id):
    best_faces = get_best_faces_for_group(group_id)
    if not best_faces:
        logging.info(f"No faces found for group {group_id}")
        return

    # Update persons table with best face thumbs

    # Get embedding size from first point
    first_face_id = best_faces[0]["id"]
    face_point = qdrant.retrieve(collection_name=str(group_id), ids=[first_face_id], with_vectors=True)
    if not face_point or not face_point[0].vector:
        logging.warning(f"No embedding found for first face in group {group_id}")
        return

    vector_data = face_point[0].vector
    if isinstance(vector_data, dict):
        vector_data = vector_data.get("face")  # Or whichever named vector you want
    vector_size = len(vector_data)
    target_collection = ensure_person_centroid_collection(group_id, vector_size)

    points_to_upsert = []
    for face in best_faces:
        face_id = face["id"]
        person_id = face["person_id"]

        retrieved = qdrant.retrieve(
            collection_name=str(group_id),
            ids=[face_id],
            with_vectors=True
        )
        if not retrieved or not retrieved[0].vector:
            logging.warning(f"No embedding found for face {face_id} in group {group_id}")
            continue

        vector_data = retrieved[0].vector
        if isinstance(vector_data, dict):
            vector_data = vector_data.get("face")  # pick only "face" vector

        points_to_upsert.append(
            PointStruct(
                id=str(person_id),
                vector=vector_data
            )
        )

    if points_to_upsert:
        qdrant.upsert(collection_name=target_collection, points=points_to_upsert)
        logging.info(f"Upserted {len(points_to_upsert)} person centroids into {target_collection}")


def main():
    groups = get_warmed_groups()
    logging.info(f"Found warmed groups: {groups}")

    for group_id in groups:
        logging.info(f"Processing group {group_id}")
        process_group(group_id)


if __name__ == "__main__":
    main()
