import os
import shutil
import psycopg2
from PIL import Image
import io

# ✅ PostgreSQL Connection Config
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "admin",
    "host": "localhost",
    "port": "5432"
}

# ✅ Output directory where grouped folders will be created
OUTPUT_DIR = "grouped_people"

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def fetch_ready_to_group_records():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT person_id, image_path, face_thumb_bytes
        FROM ready_to_group
        WHERE person_id IS NOT NULL
        ORDER BY person_id;
    """
    cursor.execute(query)
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records

def organize_images_with_thumbnails(records, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    person_thumb_created = set()  # Track person_id with thumbnails
    person_thumb_cache = {}       # Store the first valid face bytes for each person_id

    for person_id, image_path, face_thumb_bytes in records:
        person_folder = os.path.join(output_dir, str(person_id))
        os.makedirs(person_folder, exist_ok=True)

        # ✅ Copy original image to person folder
        if os.path.exists(image_path):
            destination_path = os.path.join(person_folder, os.path.basename(image_path))
            shutil.copy2(image_path, destination_path)
        else:
            print(f"⚠️ Image not found: {image_path}")

        # ✅ Cache first non-null face bytes for this person_id
        if person_id not in person_thumb_cache and face_thumb_bytes:
            person_thumb_cache[person_id] = face_thumb_bytes

    # ✅ Now create thumbnails using cached values
    for person_id, thumb_bytes in person_thumb_cache.items():
        try:
            person_folder = os.path.join(output_dir, str(person_id))
            img = Image.open(io.BytesIO(thumb_bytes))  # Convert bytes to Image
            img.thumbnail((150, 150))  # Resize to thumbnail size
            thumb_path = os.path.join(person_folder, f"thumb_{person_id}.jpg")
            img.save(thumb_path, "JPEG")
            person_thumb_created.add(person_id)
            print(f"✅ Thumbnail created for person_id: {person_id}")
        except Exception as e:
            print(f"❌ Error creating thumbnail for {person_id}: {e}")

    print(f"✅ Organized {len(records)} images and generated {len(person_thumb_created)} thumbnails in '{output_dir}'")

if __name__ == "__main__":
    records = fetch_ready_to_group_records()
    organize_images_with_thumbnails(records)
