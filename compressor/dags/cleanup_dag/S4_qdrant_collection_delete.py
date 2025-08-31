import psycopg2
from qdrant_client import QdrantClient

def cleanup_qdrant():
    # 1️⃣ Connect to Postgres
    conn = psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM groups")
    valid_group_ids = {str(row[0]) for row in cur.fetchall()}
    cur.close()
    conn.close()
    print(f"✅ Loaded {len(valid_group_ids)} valid group IDs from Postgres.")

    # 2️⃣ Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")  # update if remote

    # 3️⃣ List all collections
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    print(f"🔍 Found {len(collection_names)} collections in Qdrant.")

    # 4️⃣ Cleanup loop
    for collection in collection_names:
        # direct group collection
        if collection in valid_group_ids:
            continue  # still valid group

        # person centroid collection (check suffix)
        if collection.startswith("person_centroid_"):
            group_id = collection.replace("person_centroid_", "")
            if group_id in valid_group_ids:
                continue  # still valid

        # If we reach here → collection is orphaned
        print(f"🗑️ Deleting orphan collection: {collection}")
        client.delete_collection(collection_name=collection)

    print("✅ Qdrant cleanup complete.")

if __name__ == "__main__":
    cleanup_qdrant()
