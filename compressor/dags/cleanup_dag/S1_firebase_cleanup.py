import psycopg2
import firebase_admin
from firebase_admin import credentials, storage
from concurrent.futures import ThreadPoolExecutor, as_completed

def cleanup():
    # 1. Connect to Postgres
    conn = psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )
    cur = conn.cursor()

    # --- Get image IDs that are expired themselves ---
    cur.execute("""
        SELECT id 
        FROM images 
        WHERE delete_at IS NOT NULL 
          AND delete_at < NOW();
    """)
    expired_images = [row[0] for row in cur.fetchall()]

    # --- Get image IDs belonging to expired groups ---
    cur.execute("""
        SELECT i.id
        FROM images i
        JOIN groups g ON i.group_id = g.id
        WHERE g.delete_at IS NOT NULL
          AND g.delete_at < NOW();
    """)
    expired_group_images = [row[0] for row in cur.fetchall()]

    # Combine unique IDs
    ids_to_delete = list(set(expired_images + expired_group_images))
    print(f"🗑️ Found {len(ids_to_delete)} images to delete")

    if not ids_to_delete:
        print("✅ No images to delete")
        cur.close()
        conn.close()
        return

    # 2. Firebase bucket
    cred = credentials.Certificate("firebase-key.json")
    if not firebase_admin._apps:  # prevent re-initialization error
        firebase_admin.initialize_app(cred, {
            "storageBucket": "gallery-585ee.firebasestorage.app"
        })
    bucket = storage.bucket()

    prefixes = ["f_", "u_", "compressed_", "thumbnail_", "compressed_3k_", "stripped_", ""]

    def delete_blob(file_id, prefix):
        blob_name = f"{prefix}{file_id}"
        blob = bucket.blob(blob_name)
        try:
            blob.delete()
            return f"✅ Deleted: {blob_name}"
        except Exception:
            return f"⚠️ Skip missing: {blob_name}"

    # 3. Parallel deletion
    with ThreadPoolExecutor(max_workers=20) as executor:  # adjust workers for your needs
        futures = []
        for file_id in ids_to_delete:
            for prefix in prefixes:
                futures.append(executor.submit(delete_blob, file_id, prefix))

        for future in as_completed(futures):
            print(future.result())

    cur.close()
    conn.close()

if __name__ == "__main__":
    cleanup()
