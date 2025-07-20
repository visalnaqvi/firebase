from qdrant_client.http.models import CollectionStatus
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import HnswConfigDiff
COLLECTION_NAME = "faces"
THRESHOLD = 0.7
QDRANT_HOST = "http://localhost:6333"  # or your server IP

client = QdrantClient(QDRANT_HOST)
# Check if collection exists
collections = client.get_collections().collections
existing_names = [c.name for c in collections]

# if COLLECTION_NAME not in existing_names:
#     client.create_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(
#             size=128,
#             distance=Distance.COSINE
#         ),
#         hnsw_config=HnswConfigDiff(
#             m=16,
#             ef_construct=100
#         )
#     )
    # 30.38sec


    # Create the collection if it doesn't exist

if COLLECTION_NAME not in existing_names:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE)
    )