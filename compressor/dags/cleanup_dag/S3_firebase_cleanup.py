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
    cur.execute("SELECT id FROM images")
    valid_ids = {row[0] for row in cur.fetchall()}

    # 2. Firebase bucket
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "gallery-585ee.firebasestorage.app"
    })
    bucket = storage.bucket()

    # 3. List files
    blobs = bucket.list_blobs()
    for blob in blobs:
        filename = blob.name

        # Normalize base id (strip known prefixes)
        base_id = (
            filename.replace("compressed_", "")
                    .replace("thumbnail_", "")
        )

        # 4. Delete orphan sets
        if base_id not in valid_ids:
            print(f"Deleting orphan files for: {base_id}")
            for prefix in ["", "compressed_", "thumbnail_"]:
                blob_to_delete = bucket.blob(f"{prefix}{base_id}")
                try:
                    blob_to_delete.delete()
                except Exception as e:
                    # Likely blob doesn’t exist — skip silently
                    # print(f"Skip missing: {prefix}{base_id} ({e})")
                    pass

    cur.close()
    conn.close()

cleanup()
