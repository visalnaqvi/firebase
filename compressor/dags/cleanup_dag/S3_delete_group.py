import psycopg2
from psycopg2.extras import DictCursor
from qdrant_client import QdrantClient
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    qdrant = QdrantClient("localhost", port=6333)  # 🔹 update if remote

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

        # 2️⃣ Delete dependent records in Postgres
        cur.execute("DELETE FROM faces WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"🗑️ Deleted {cur.rowcount} faces.")

        cur.execute("DELETE FROM persons WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"🗑️ Deleted {cur.rowcount} persons.")

        cur.execute("DELETE FROM similar_faces WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"🗑️ Deleted {cur.rowcount} similar_faces.")

        cur.execute("DELETE FROM album_images WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"🗑️ Deleted {cur.rowcount} album-image links.")

        cur.execute("DELETE FROM albums WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"🗑️ Deleted {cur.rowcount} albums.")

        cur.execute("DELETE FROM drive_folders WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"🗑️ Deleted {cur.rowcount} drive_folders.")

        cur.execute("DELETE FROM images WHERE group_id = ANY(%s);", (expired_group_ids,))
        print(f"🗑️ Deleted {cur.rowcount} images.")

        # 3️⃣ Parallel delete in Qdrant
        def delete_collections(gid):
            results = []
            try:
                qdrant.delete_collection(str(gid))
                results.append(f"🗑️ Deleted Qdrant collection: {gid}")
            except Exception:
                results.append(f"⚠️ Collection {gid} not found in Qdrant.")

            try:
                qdrant.delete_collection(f"person_centroid_{gid}")
                results.append(f"🗑️ Deleted Qdrant collection: person_centroid_{gid}")
            except Exception:
                results.append(f"⚠️ Collection person_centroid_{gid} not found in Qdrant.")
            return results

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(delete_collections, gid) for gid in expired_group_ids]
            for future in as_completed(futures):
                for line in future.result():
                    print(line)

        # 4️⃣ Delete groups
        cur.execute("DELETE FROM groups WHERE id = ANY(%s);", (expired_group_ids,))
        print(f"🗑️ Deleted {cur.rowcount} groups.")

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
