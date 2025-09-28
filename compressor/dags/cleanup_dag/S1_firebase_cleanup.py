import psycopg2
import firebase_admin
from firebase_admin import credentials, storage

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

    # 3. Delete all variants for each ID
    prefixes = ["f_", "u_", "compressed_", "thumbnail_", "compressed_3k_", "stripped_", ""]
    for file_id in ids_to_delete:
        print(f"🗑️ Deleting files for Image ID: {file_id}")
        for prefix in prefixes:
            blob_name = f"{prefix}{file_id}"
            blob = bucket.blob(blob_name)
            try:
                blob.delete()
                print(f"   ✅ Deleted: {blob_name}")
            except Exception:
                print(f"   ⚠️ Skip missing: {blob_name}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    cleanup()
