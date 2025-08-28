import psycopg2
from google.cloud import storage

def cleanup():
    # 1. Connect to Postgres
    conn = psycopg2.connect(
        host="your-db-host",
        dbname="your-db",
        user="your-user",
        password="your-password",
        port=5432
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM images")
    valid_ids = {row[0] for row in cur.fetchall()}

    # 2. Connect to Firebase Storage
    client = storage.Client()
    bucket = client.bucket("your-bucket-name")

    # 3. List files
    blobs = bucket.list_blobs()
    for blob in blobs:
        filename = blob.name
        base_id = filename.replace("compressed_", "") if filename.startswith("compressed_") else filename

        # 4. Delete if not in Postgres
        if base_id not in valid_ids:
            print(f"Deleting orphan file: {filename}")
            bucket.blob(base_id).delete(if_exists=True)
            bucket.blob(f"compressed_{base_id}").delete(if_exists=True)

    cur.close()
    conn.close()

cleanup()
