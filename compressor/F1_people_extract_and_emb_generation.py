import os
import uuid
import cv2
import torch
import psycopg2
from glob import glob
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from psycopg2.extras import execute_values
import cv2
# ✅ PostgreSQL Config
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "admin",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def fetch_unprocessed_images():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, image_path FROM images_to_process")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # [(id, image_path), ...]

def insert_ready_to_group_batch(records):
    if not records:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
    INSERT INTO ready_to_group (id, image_path, filename, face_id, face_emb, clothing_emb, assigned, image_id, person_id , cropped_img_byte,face_thumb_bytes )
    VALUES %s
    """
    values = [
        (r['id'], r['image_path'], r['filename'], r['face_id'],
         r['face_emb'].tolist(), r['clothing_emb'].cpu().tolist(),
         r['assigned'], r['image_id'], r['person_id'] , r['cropped_img_byte '] , r['face_thumb_bytes'])
        for r in records
    ]
    execute_values(cur, query, values)
    conn.commit()
    cur.close()
    conn.close()

def mark_images_processed_batch(image_ids):
    if not image_ids:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    query = "UPDATE images_to_process SET is_emb_extracted = true WHERE id = ANY(%s)"
    cur.execute(query, (image_ids,))
    conn.commit()
    cur.close()
    conn.close()

class HybridFaceIndexer:
    def __init__(self, host="localhost", port=6333):
        # ✅ Face Analysis
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0)

        # ✅ FashionCLIP Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fashion_model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True).to(self.device)
        self.fashion_processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)

        # ✅ Qdrant
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

    def extract_faces(self, img):
        return self.face_app.get(img)

    def extract_clothing_embedding(self, image_input):
        """
        Accepts either a file path (str) or a NumPy array (OpenCV crop).
        Converts to PIL image internally.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            # Assuming it's a NumPy array (OpenCV image)
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

        inputs = self.fashion_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.fashion_model.get_image_features(**inputs)
            return emb[0] / emb[0].norm()
    def image_to_bytes(self,cv_image):
        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', cv_image)
        if not success:
            raise ValueError("Could not encode image")
        return buffer.tobytes()
    def process_image(self, image_id, image_path, yolo_model, cropped_dir):
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to read image: {image_path}")
            return []

        results = yolo_model(img)[0]
        records = []
        person_count = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.5:  # Person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = img[y1:y2, x1:x2]
                person_count += 1

                faces = self.extract_faces(person_crop)
                if not faces:
                    continue

                clothing_emb = self.extract_clothing_embedding(person_crop)
                cropped_img_byte  = self.image_to_bytes(person_crop)
                face_thumb_bytes = None
                if len(faces) == 1:
                    f = faces[0]
                    x1_f, y1_f, x2_f, y2_f = map(int, f.bbox)
                    face_crop = person_crop[y1_f:y2_f, x1_f:x2_f]
                    if face_crop.size > 0:
                        face_thumb_bytes = self.image_to_bytes(face_crop)
                for face in faces:
                    face_emb = face.normed_embedding
                    point_id = str(uuid.uuid4())
                    face_id = str(uuid.uuid4())
                  

                    # ✅ Insert into Qdrant
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
                                    "face_id": face_id,
                                    "person_id": -1,
                                    "image_id": image_id
                                }
                            )
                        ]
                    )

                    # ✅ Add record for DB
                    records.append({
                        "id": point_id,
                        "image_path": image_path,
                        "filename": image_path,
                        "face_id": face_id,
                        "face_emb": face_emb,
                        "clothing_emb": clothing_emb,
                        "assigned": False,
                        "image_id": image_id,
                        "person_id": -1,
                        "cropped_img_byte ":cropped_img_byte ,
                        "face_thumb_bytes":face_thumb_bytes
                    })

        return records

if __name__ == "__main__":
    yolo_model = YOLO("yolov8x.pt")
    cropped_dir = "cropped_people"
    os.makedirs(cropped_dir, exist_ok=True)

    indexer = HybridFaceIndexer()

    unprocessed = fetch_unprocessed_images()
    print(f"Found {len(unprocessed)} unprocessed images")

    all_records = []
    processed_image_ids = []

    for id, image_path in unprocessed:
        print(f"\n🔍 Processing {image_path}")
        records = indexer.process_image(id, image_path, yolo_model, cropped_dir)
        if records:
            all_records.extend(records)
            processed_image_ids.append(id)

    # ✅ Bulk insert
    if all_records:
        insert_ready_to_group_batch(all_records)
    if processed_image_ids:
        mark_images_processed_batch(processed_image_ids)

    print(f"✅ Process completed: {len(all_records)} faces indexed & stored")
