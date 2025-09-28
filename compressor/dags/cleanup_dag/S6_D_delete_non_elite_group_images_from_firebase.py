import psycopg2
from psycopg2.extras import DictCursor
import firebase_admin
from firebase_admin import credentials, storage

# === CONFIG ===
DB_CONFIG = {
    "host": "ballast.proxy.rlwy.net",
    "port": "56193",
    "dbname": "railway",
    "user": "postgres",
    "password": "AfldldzckDWtkskkAMEhMaDXnMqknaPY",
}

FIREBASE_BUCKET = "gallery-585ee.firebasestorage.app"  # 🔹 replace with your Firebase bucket name
CRED_FILE = "firebase-key.json"  # 🔹 path to your service account JSON


def cleanup_non_elite_files():
    # === Init Firebase ===
    cred = credentials.Certificate(CRED_FILE)
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
    bucket = storage.bucket()

    # === Connect Postgres ===
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=DictCursor)

    try:
        # 1️⃣ Get groups where plan_type != 'elite'
        cur.execute("SELECT id FROM groups WHERE plan_type IS NULL OR plan_type != 'elite';")
        groups = cur.fetchall()

        if not groups:
            print("✅ No non-elite groups found.")
            return

        group_ids = [row["id"] for row in groups]
        print(f"Found {len(group_ids)} non-elite groups: {group_ids}")

        # 2️⃣ For each group → get image ids
        for gid in group_ids:
            cur.execute("SELECT id FROM images WHERE group_id = %s;", (gid,))
            images = cur.fetchall()
            if not images:
                continue

            image_ids = [row["id"] for row in images]
            print(f" Group {gid} → {len(image_ids)} images")

            # 3️⃣ Delete original `{id}` files from Firebase
            for img_id in image_ids:
                blob = bucket.blob(img_id)
                if blob.exists():
                    blob.delete()
                    print(f"   🗑️ Deleted original file {img_id} from Firebase")
                else:
                    print(f"   ⚠️ File {img_id} not found in Firebase (skipped)")

        print("✅ Cleanup complete.")

    except Exception as e:
        print("❌ Error:", e)

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    cleanup_non_elite_files()
