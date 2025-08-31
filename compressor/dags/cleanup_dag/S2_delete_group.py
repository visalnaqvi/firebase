import psycopg2
from psycopg2.extras import DictCursor
from qdrant_client import QdrantClient

def cleanup_expired_groups():
    # === Postgres Connection ===
    conn = psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )
    cur = conn.cursor(cursor_factory=DictCursor)

    # === Qdrant Connection ===
    qdrant = QdrantClient("localhost", port=6333)  # 🔹 update if remote Qdrant URL

    try:
        # 1️⃣ Get expired groups
        cur.execute("""
            SELECT id 
            FROM groups 
            WHERE delete_at IS NOT NULL 
              AND delete_at < NOW();
        """)
        expired_groups = cur.fetchall()

        if not expired_groups:
            print("✅ No expired groups found.")
            return

        expired_group_ids = [row["id"] for row in expired_groups]
        print(f"Found {len(expired_group_ids)} expired groups: {expired_group_ids}")

        # 2️⃣ Delete from faces
        cur.execute("DELETE FROM faces WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"Deleted {cur.rowcount} faces.")

        # 3️⃣ Delete from persons
        cur.execute("DELETE FROM persons WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"Deleted {cur.rowcount} persons.")

        # 4️⃣ Delete from similar_faces
        cur.execute("DELETE FROM similar_faces WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"Deleted {cur.rowcount} similar_faces.")

        # 5️⃣ Delete Qdrant collections
        for gid in expired_group_ids:
            try:
                qdrant.delete_collection(str(gid))
                print(f"Deleted Qdrant collection: {gid}")
            except Exception:
                print(f"⚠️ Collection {gid} not found in Qdrant.")
            try:
                qdrant.delete_collection(f"person_centroid_{gid}")
                print(f"Deleted Qdrant collection: person_centroid_{gid}")
            except Exception:
                print(f"⚠️ Collection person_centroid_{gid} not found in Qdrant.")

        # 6️⃣ Delete groups
        cur.execute("DELETE FROM groups WHERE id = ANY(%s);", (expired_group_ids,))
        print(f"Deleted {cur.rowcount} groups.")

        conn.commit()
        print("✅ Group cleanup complete.")

    except Exception as e:
        conn.rollback()
        print("❌ Error during cleanup:", e)

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    cleanup_expired_groups()
