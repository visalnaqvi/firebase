from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(
    url="http://localhost:6333",  # Change if running remotely
    api_key=None  # Add API key if required
)

collection_name = "4"

# Step 1: Get all point IDs
scroll_filter = None
next_page = None
all_ids = []

while True:
    print("updating")
    result, next_page = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        with_payload=False,
        limit=500
    )
    print(result)
    if not result:
        break
    all_ids.extend([point.id for point in result])
    if next_page is None:
        break

print(f"Found {len(all_ids)} points")

# Step 2: Update payload for all points
if all_ids:
    client.set_payload(
        collection_name=collection_name,
        payload={
            "person_id": None,
            "cloth_ids": None
        },
        points=all_ids
    )
    print(f"✅ Updated {len(all_ids)} points with person_id=None and cloth_ids=None")
else:
    print("No points found.")
