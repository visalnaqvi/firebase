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
import numpy as np
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

def fetch_unprocessed_images(group_id):
    print(f"🔃Fetching Images id and image_byte with statis as warm for Gorup {id}")
    conn = get_db_connection()
    cur = conn.cursor()
    # Use parameterized query to avoid SQL injection
    cur.execute("SELECT id , image_byte FROM images WHERE status = %s AND group_id = %s", ('warm', group_id))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # [(id, image_path), ...]

def fetch_warm_groups():
    print("🔃 Fetching Warm groups from database to extract embeddings")
    conn = get_db_connection()
    cur = conn.cursor()
    # Use parameterized query to avoid SQL injection
    cur.execute("SELECT id FROM groups WHERE status = %s", ('warm'))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows 


def insert_ready_to_group_batch(records , id):
    print(f"🔃 Inserting detected face into postgress db for group {id}")
    if not records:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
    INSERT INTO faces (id, face_emb, clothing_emb, assigned, image_id, group_id, person_id , cropped_img_byte , face_thumb_bytes )
    VALUES %s
    """
    values = [
        (r['id'],
         r['face_emb'].tolist(), r['clothing_emb'].cpu().tolist(),
         r['assigned'], r['image_id'],r[id], r['person_id'] , r['cropped_img_byte'] , r['face_thumb_bytes'])
        for r in records
    ]
    execute_values(cur, query, values)
    print(f"✅ Inserted detected face into postgress db for group {id}")
    conn.commit()
    cur.close()
    conn.close()

def mark_images_processed_batch(image_ids):
    if not image_ids:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    
    query = "UPDATE images SET status = 'warmed' WHERE id = ANY(%s)"
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
        print(f"🔃 Setting up qudrant collection")
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "face": VectorParams(size=512, distance=Distance.COSINE),
                "clothing": VectorParams(size=512, distance=Distance.COSINE)
            }
        )
        print(f"✅ Qdrant Setup done")

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
    def process_image(self, image_id,image_byte_3k, yolo_model):
        nparr = np.frombuffer(image_byte_3k, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(f"🔃Processing Image {image_id}")
        if img is None:
            print(f"❌ Failed to read image: {image_id}")
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
                print(f"🔃Extracting face from Image {image_id}")
                faces = self.extract_faces(person_crop)
                if not faces:
                    continue
                print(f"✅Extracted {len(faces)} faces from Image {image_id}")   
                print(f"🔃Extracting clothing emb from Image {image_id}") 
                clothing_emb = self.extract_clothing_embedding(person_crop)
                print(f"✅Extracted clothing emb from Image {image_id}")   
                print(f"🔃Extracting cropped image bytes from Image {image_id}") 
                cropped_img_byte = self.image_to_bytes(person_crop)
                print(f"✅Extracted cropped image bytes Image {image_id}")   
                print(f"🔃Extracting thumbnail image bytes from Image {image_id}") 
                face_thumb_bytes = None
                if len(faces) == 1:
                    f = faces[0]
                    x1_f, y1_f, x2_f, y2_f = map(int, f.bbox)
                    face_crop = person_crop[y1_f:y2_f, x1_f:x2_f]
                    if face_crop.size > 0:
                        face_thumb_bytes = self.image_to_bytes(face_crop)
                print(f"✅Extracted thumbnail image bytes Image {image_id}")   
                print(f"🔃Extracting each face embedding from Image {image_id}")
                for face in faces:
                    face_emb = face.normed_embedding
                    point_id = str(uuid.uuid4())               

                    print(f"✅Extracted each face embedding Image {image_id}")   
                    print(f"🔃 Inserting Embedding into qdrant db with point id as {point_id} for Image {image_id}")
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
                                    "person_id": -1,
                                    "image_id": image_id
                                }
                            )
                        ]
                    )

                    print(f"✅Inserted Embedding into qdrant db with point id as {point_id} for Image {image_id}")   
                    print(f"🔃 Inserting Details into records with id as {point_id} for Image {image_id}")

                    records.append({
                        "id": point_id,
                        "face_emb": face_emb,
                        "clothing_emb": clothing_emb,
                        "assigned": False,
                        "image_id": image_id,
                        "person_id": -1,
                        "cropped_img_byte":cropped_img_byte,
                        "face_thumb_bytes":face_thumb_bytes
                    })
        print(f"✅ All Face proccessing finished for image {image_id}")
        return records

if __name__ == "__main__":
    yolo_model = YOLO("yolov8x.pt")

    indexer = HybridFaceIndexer()
    groups = [row[0] for row in fetch_warm_groups()]
    print(f"ℹFound {len(groups)} warm groups")
    for id in groups:
        print(f"🔃 Processing group {id}")
        unprocessed = fetch_unprocessed_images(id)
        print(f"ℹFound {len(unprocessed)} unprocessed images")

        all_records = []
        processed_image_ids = []

        for id,image_byte in unprocessed:
            print(f"\n🔍 Processing {id}")
            records = indexer.process_image(id,image_byte, yolo_model)
            if records:
                print(f"✅ Got All Face Records for image {id} extending into all records")
                all_records.extend(records)
                # processed_image_ids.append(id)

        if all_records:
            insert_ready_to_group_batch(all_records)
        if processed_image_ids:
            mark_images_processed_batch(processed_image_ids , id)

        print(f"✅ Process completed: {len(all_records)} faces indexed & stored")
