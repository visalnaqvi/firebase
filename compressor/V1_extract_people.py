import cv2
from ultralytics import YOLO
import os
from glob import glob
import os
import shutil
import time
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from collections import deque
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct
)

class HybridFaceGrouping:
    def __init__(self, host="localhost", port=6333):
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fashion_model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True).to(self.device)
        self.fashion_processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)

        self.qdrant = QdrantClient(host=host, port=port)
        self.collection_name = "hybrid_people_collection"
        self._setup_collection()

    def _setup_collection(self):
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "face": VectorParams(size=512, distance=Distance.COSINE),
                "clothing": VectorParams(size=512, distance=Distance.COSINE)
            }
        )

    def extract_faces(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []
        return self.face_app.get(img)

    def extract_clothing_embedding(self, image_path):
        img = Image.open(image_path).convert('RGB')
        inputs = self.fashion_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.fashion_model.get_image_features(**inputs)
            return emb[0] / emb[0].norm()

    def index_images(self, folder_path):
        print("🔍 Indexing images with multiple faces...")
        items = []
        point_id = 0

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(folder_path, filename)
            faces = self.extract_faces(image_path)

            if not faces:
                print(f"⚠️  No face found in {filename}")
                continue

            clothing_emb = self.extract_clothing_embedding(image_path)

            for face_index, face in enumerate(faces):
                face_emb = face.normed_embedding

                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector={
                                "face": face_emb.tolist(),
                                "clothing": clothing_emb.cpu().tolist()
                            },
                            payload={
                                "image_path": image_path,
                                "filename": filename,
                                "face_index": face_index
                            }
                        )
                    ]
                )

                items.append({
                    'id': point_id,
                    'path': image_path,
                    'filename': filename,
                    'face_index': face_index,
                    'face': face_emb,
                    'cloth': clothing_emb,
                    'assigned': False
                })

                point_id += 1

        print(f"✅ Indexed {len(items)} total faces across all images")
        return items

model = YOLO("yolov8x.pt")  

input_dir = "cl_img"
output_dir = "cropped_people"
os.makedirs(output_dir, exist_ok=True)

image_paths = glob(os.path.join(input_dir, "*.*"))
supported_exts = (".jpg", ".jpeg", ".png", ".bmp")

for image_path in image_paths:
    if not image_path.lower().endswith(supported_exts):
        continue

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to read {image_path}")
        continue

    results = model(image)[0]

    person_count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf > 0.5: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = image[y1:y2, x1:x2]
            output_filename = f"{image_name}_person_{person_count}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, person_crop)
            person_count += 1

    print(f"{person_count} people cropped from '{image_path}'")
