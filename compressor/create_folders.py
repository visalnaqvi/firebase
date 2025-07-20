import os
import shutil
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# === Configuration ===
COLLECTION_NAME = "faces"
IMAGE_BASE_PATH = "./images"  # Change if images are stored elsewhere
DEST_FOLDER_BASE = "./grouped_faces"

# Connect to local Qdrant
client = QdrantClient(host="localhost", port=6333)

# === Step 1: Retrieve all points ===
scroll_offset = None
all_points = []

while True:
    print("jnjnsdjcnjsdnjksdnkjdcnjkcnsdjkcnsdjknk")
    response = client.scroll(
        collection_name=COLLECTION_NAME,
        offset=scroll_offset,
        limit=10,  # Adjust batch size as needed
        with_payload=True
    )
    points, scroll_offset = response
    if not points:
        break
    all_points.extend(points)

    if scroll_offset is None:
        break  # No more pages

print(f"✅ Retrieved {len(all_points)} points from Qdrant.")

# === Step 2: Group by person_id ===
person_map = {}

for point in all_points:
    payload = point.payload
    if not payload:
        print("no payload")
        continue

    image_path = payload.get("img_path")
    person_id = payload.get("person_id")
    print(f"got {image_path} for person {person_id}")
    if person_id is None or image_path is None:
        continue

    person_map.setdefault(person_id, []).append(image_path)

# === Step 3: Copy images to folders ===
for person_id, images in person_map.items():
    person_folder = os.path.join(DEST_FOLDER_BASE, f"person_{person_id}")
    os.makedirs(person_folder, exist_ok=True)

    for img_path in images:
        src_path = os.path.join(IMAGE_BASE_PATH, os.path.basename(img_path))
        dst_path = os.path.join(person_folder, os.path.basename(img_path))

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"⚠️ Image not found: {src_path}")

print(f"✅ Grouped {len(person_map)} people into folders.")
