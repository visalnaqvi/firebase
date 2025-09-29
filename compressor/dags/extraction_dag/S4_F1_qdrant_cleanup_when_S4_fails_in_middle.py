import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Postgres connection config


# Qdrant config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def get_group_ids():
    """Fetch group_ids from process_status where status = 'extraction'."""
    with psycopg2.connect(
         host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    ) as conn, conn.cursor() as cur:
        cur.execute("SELECT group_id FROM process_status WHERE status = 'extraction'")
        rows = cur.fetchall()
    return 17


def get_face_ids_for_group(group_id):
    """Fetch all face_ids from faces table for a specific group_id."""
    with psycopg2.connect(
         host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    ) as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM faces WHERE group_id = %s", (group_id,))
        rows = cur.fetchall()
    return {str(r[0]) for r in rows}  # Convert to string for Qdrant match


def get_qdrant_ids(client, collection_name):
    """Fetch all point IDs from a Qdrant collection."""
    qdrant_ids = set()
    offset = None

    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            limit=1000,  # fetch in batches
            offset=offset
        )
        qdrant_ids.update({str(p.id) for p in points})
        if next_page is None:
            break
        offset = next_page

    return qdrant_ids


def cleanup_qdrant():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    group_ids = get_group_ids()
    logging.info(f"Found group_ids with extraction status: {group_ids}")

    collections = [c.name for c in client.get_collections().collections]

    for group_id in group_ids:
        collection_name = str(group_id)

        if collection_name not in collections:
            logging.warning(f"Collection {collection_name} not found in Qdrant.")
            continue

        # Get Qdrant point IDs for this collection
        qdrant_ids = get_qdrant_ids(client, collection_name)

        # Get DB face_ids for this group only
        db_face_ids = get_face_ids_for_group(group_id)

        # Find IDs present in Qdrant but missing in DB
        extra_ids = qdrant_ids - db_face_ids
        if extra_ids:
            logging.info(f"Deleting {len(extra_ids)} orphaned points from collection {collection_name}")
            client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=list(extra_ids))
            )
        else:
            logging.info(f"No orphaned points found for collection {collection_name}")


if __name__ == "__main__":
    cleanup_qdrant()
