import psycopg2
from psycopg2.extras import DictCursor

def cleanup_expired_images():
    conn = psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )
    cur = conn.cursor(cursor_factory=DictCursor)

    try:
        # 1️⃣ Find expired images
        cur.execute("""
            SELECT id 
            FROM images 
            WHERE delete_at IS NOT NULL 
              AND delete_at < NOW();
        """)
        expired_images = cur.fetchall()

        if not expired_images:
            print("✅ No expired images found.")
            return

        expired_ids = [row["id"] for row in expired_images]
        print(f"Found {len(expired_ids)} expired images: {expired_ids}")

        # 2️⃣ Delete faces first (foreign key safety)
        cur.execute(
            "DELETE FROM faces WHERE image_id = ANY(%s);",
            (expired_ids,)
        )
        print(f"Deleted {cur.rowcount} faces linked to expired images.")

        # 3️⃣ Delete images
        cur.execute(
            "DELETE FROM images WHERE id = ANY(%s);",
            (expired_ids,)
        )
        print(f"Deleted {cur.rowcount} expired images.")

        conn.commit()
        print("✅ Cleanup complete.")

    except Exception as e:
        conn.rollback()
        print("❌ Error during cleanup:", e)

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    cleanup_expired_images()
