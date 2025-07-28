import os
import shutil
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SearchRequest
import uuid
from collections import defaultdict
from tqdm import tqdm

class HybridFaceGroupingBatch:
    def __init__(self, host="localhost", port=6333, batch_size=1000, workers=8):
        # Initialize models
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fashion_model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True).to(self.device)
        self.fashion_processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)

        # Initialize Qdrant
        self.qdrant = QdrantClient(host=host, port=port)
        self.collection_name = "hybrid_people_collection_batch"
        self._setup_collection()

        self.batch_size = batch_size
        self.workers = workers

    def _setup_collection(self):
        """Setup collection only if it doesn't exist (persistent)"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            print(f"✅ Using existing collection with {collection_info.points_count} points")
        except:
            print("🔧 Creating new collection...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "face": VectorParams(size=512, distance=Distance.COSINE),
                    "cloth": VectorParams(size=512, distance=Distance.COSINE)
                }
            )

    def extract_embeddings(self, image_path):
        """Extract both face and clothing embeddings"""
        img = cv2.imread(image_path)
        if img is None:
            return None

        faces = self.face_app.get(img)
        if not faces:
            return None

        face_emb = faces[0].normed_embedding
        img_pil = Image.open(image_path).convert('RGB')
        inputs = self.fashion_processor(images=img_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            cloth_emb = self.fashion_model.get_image_features(**inputs)[0]
            cloth_emb = cloth_emb / cloth_emb.norm()

        return {
            "path": image_path,
            "face": face_emb.astype(np.float16),
            "cloth": cloth_emb.cpu().numpy().astype(np.float16)
        }

    def process_folder_in_batches(self, input_folder, output_folder, face_th=0.7, cloth_th=0.85):
        os.makedirs(output_folder, exist_ok=True)
        image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        total_images = len(image_paths)
        print(f"📂 Found {total_images} images. Processing in batches of {self.batch_size}...")

        start_time = time.time()

        # Process images in batches
        for i in range(0, total_images, self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            print(f"\n📦 Processing batch {i//self.batch_size + 1}: {len(batch_paths)} images")

            # Extract embeddings in parallel
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                embeddings = list(tqdm(executor.map(self.extract_embeddings, batch_paths), total=len(batch_paths)))

            embeddings = [e for e in embeddings if e is not None]  # Remove None

            if not embeddings:
                continue

            # Bulk insert into Qdrant
            points = []
            for e in embeddings:
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"face": e["face"].tolist(), "clothing": e["cloth"].tolist()},
                        payload={"image_path": e["path"]}
                    )
                )

            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            print(f"✅ Inserted {len(points)} embeddings into Qdrant")

        print("\n✅ All batches indexed into Qdrant")
        print(f"⏱️ Total time: {(time.time() - start_time)/60:.2f} mins")

        # Grouping after indexing
        self.group_and_organize(output_folder, face_th, cloth_th)

    def group_and_organize(self, output_folder, face_th=0.7, cloth_th=0.85):
        print("\n📊 Starting grouping from Qdrant data...")

        offset = None
        person_id_map = {}  # image -> person_id
        current_person = 0

        while True:
            points, offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            if not points:
                break

            for point in points:
                img_path = point.payload["image_path"]
                if img_path in person_id_map:
                    continue

                # Assign a new group
                current_person += 1
                person_id_map[img_path] = current_person

                # Find similar images
                face_vector = point.vector["face"]
                results = self.qdrant.search(
                    collection_name=self.collection_name,
                    query=face_vector,
                    limit=50,
                    using="face"
                )

                for r in results:
                    if r.score >= face_th:
                        other_img = r.payload["image_path"]
                        if other_img not in person_id_map:
                            person_id_map[other_img] = current_person

        # Organize into folders
        grouped = defaultdict(list)
        for img, pid in person_id_map.items():
            grouped[pid].append(img)

        print(f"✅ Created {len(grouped)} groups")
        for pid, images in grouped.items():
            person_folder = os.path.join(output_folder, f"person_{pid:03d}")
            os.makedirs(person_folder, exist_ok=True)
            for img in images:
                shutil.copy2(img, os.path.join(person_folder, os.path.basename(img)))

        print(f"📂 Organized images into {len(grouped)} folders")


if __name__ == "__main__":
    grouper = HybridFaceGroupingBatch(batch_size=1000, workers=8)
    grouper.process_folder_in_batches(
        input_folder="cropped_people",
        output_folder="hybrid_grouped_faces_batch",
        face_th=0.7,
        cloth_th=0.85
    )
