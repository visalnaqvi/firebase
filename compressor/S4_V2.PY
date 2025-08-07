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
import concurrent.futures
from threading import Lock
import time
import traceback

# Configuration
BATCH_SIZE = 10
PARALLEL_LIMIT = 10
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

def fetch_unprocessed_images(group_id):
    print(f"🔃 Fetching Images id and image_byte with status as warm for Group {group_id}")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, image_byte FROM images WHERE status = %s AND group_id = %s ORDER BY id", ('warm', group_id))
        rows = cur.fetchall()
        print(f"✅ Fetched {len(rows)} warm images for group {group_id}")
        return rows
    except Exception as e:
        print(f"❌ Error fetching images for group {group_id}: {str(e)}")
        return []
    finally:
        cur.close()
        conn.close()

def fetch_warm_groups():
    print("🔃 Fetching warm groups from database to extract embeddings")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM groups WHERE status = %s", ('warm',))
        rows = cur.fetchall()
        print(f"✅ Found {len(rows)} warm groups")
        return rows
    except Exception as e:
        print(f"❌ Error fetching warm groups: {str(e)}")
        return []
    finally:
        cur.close()
        conn.close()

def insert_faces_batch_with_retry(records, group_id, max_retries=MAX_RETRIES):
    """Insert face records with retry mechanism and fallback to individual inserts"""
    if not records:
        return True
    
    print(f"🔃 Inserting {len(records)} detected faces into postgres db for group {group_id}")
    
    for attempt in range(max_retries):
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Try batch insert first
            query = """
            INSERT INTO faces (id, image_id, group_id, person_id, face_thumb_bytes)
            VALUES %s
            """
            values = [
                (r['id'], r['image_id'], group_id, None, r['face_thumb_bytes'])
                for r in records
            ]
            
            execute_values(cur, query, values)
            conn.commit()
            
            print(f"✅ Successfully inserted {len(records)} faces for group {group_id} (attempt {attempt + 1})")
            return True
            
        except Exception as e:
            print(f"⚠️ Batch insert attempt {attempt + 1} failed: {str(e)}")
            
            try:
                conn.rollback()
            except:
                pass
            
            if attempt == max_retries - 1:
                # Last attempt - try individual inserts as fallback
                print(f"🔃 Falling back to individual inserts for group {group_id}")
                return insert_faces_individually(records, group_id)
            else:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                
        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass
    
    return False

def insert_faces_individually(records, group_id):
    """Fallback method: insert faces one by one"""
    successful_inserts = 0
    
    for record in records:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            query = """
            INSERT INTO faces (id, image_id, group_id, person_id, face_thumb_bytes)
            VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(query, (
                record['id'], record['image_id'], group_id, 
                record['person_id'], record['face_thumb_bytes']
            ))
            conn.commit()
            successful_inserts += 1
            
        except Exception as e:
            print(f"❌ Failed to insert individual face record {record['id']}: {str(e)}")
            
        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass
    
    print(f"✅ Individual inserts completed: {successful_inserts}/{len(records)} successful")
    return successful_inserts > 0

def mark_images_processed_batch_with_retry(image_ids, max_retries=MAX_RETRIES):
    """Mark images as processed with retry mechanism"""
    if not image_ids:
        return True
    
    print(f"🔃 Marking {len(image_ids)} images as processed")
    
    for attempt in range(max_retries):
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Try batch update first
            query = "UPDATE images SET status = 'warmed', image_byte = NULL WHERE id = ANY(%s)"
            cur.execute(query, (image_ids,))
            
            rows_affected = cur.rowcount
            conn.commit()
            
            print(f"✅ Successfully marked {rows_affected} images as processed (attempt {attempt + 1})")
            return True
            
        except Exception as e:
            print(f"⚠️ Mark processed attempt {attempt + 1} failed: {str(e)}")
            
            try:
                conn.rollback()
            except:
                pass
                
            if attempt == max_retries - 1:
                # Last attempt - try individual updates
                print(f"🔃 Falling back to individual updates")
                return mark_images_individually(image_ids)
            else:
                time.sleep(RETRY_DELAY * (attempt + 1))
                
        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass
    
    return False

def mark_images_individually(image_ids):
    """Fallback method: mark images as processed one by one"""
    successful_updates = 0
    
    for image_id in image_ids:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            query = "UPDATE images SET status = 'warmed', image_byte = NULL WHERE id = %s"
            cur.execute(query, (image_id,))
            conn.commit()
            
            if cur.rowcount > 0:
                successful_updates += 1
                
        except Exception as e:
            print(f"❌ Failed to mark image {image_id} as processed: {str(e)}")
            
        finally:
            try:
                cur.close()
                conn.close()
            except:
                pass
    
    print(f"✅ Individual updates completed: {successful_updates}/{len(image_ids)} successful")
    return successful_updates > 0

class HybridFaceIndexer:
    def __init__(self, host="localhost", port=6333):
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fashion_model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True).to(self.device)
        self.fashion_processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)

        self.qdrant = QdrantClient(host=host, port=port)
        self.lock = Lock()  # Thread-safe operations

    def setup_collection(self, collection_name):
        print(f"🔃 Setting up qdrant collection: {collection_name}")
        try:
            if self.qdrant.collection_exists(collection_name):
                self.qdrant.delete_collection(collection_name)

            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "face": VectorParams(size=512, distance=Distance.COSINE),
                    "clothing": VectorParams(size=512, distance=Distance.COSINE)
                }
            )
            print(f"✅ Qdrant collection setup completed: {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to setup collection {collection_name}: {str(e)}")
            return False

    def extract_faces(self, img):
        return self.face_app.get(img)

    def extract_clothing_embedding(self, image_input):
        """Extract clothing embedding from image"""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

        inputs = self.fashion_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.fashion_model.get_image_features(**inputs)
            return emb[0] / emb[0].norm()

    def image_to_bytes(self, cv_image):
        success, buffer = cv2.imencode('.jpg', cv_image)
        if not success:
            raise ValueError("Could not encode image")
        return buffer.tobytes()

    def process_single_image(self, image_data, yolo_model, group_id):
        """Process a single image and return results"""
        image_id, image_byte_3k = image_data
        
        try:
            nparr = np.frombuffer(image_byte_3k, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"❌ Failed to decode image: {image_id}")
                return {"image_id": image_id, "success": False, "error": "Failed to decode image", "records": []}

            print(f"🔃 Processing image {image_id}")
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
                    
                    # Extract faces
                    faces = self.extract_faces(person_crop)
                    if not faces:
                        continue
                    
                    # Extract clothing embedding
                    clothing_emb = self.extract_clothing_embedding(person_crop)
                    
                    # Create face thumbnail if single face
                    face_thumb_bytes = None
                    if len(faces) == 1:
                        f = faces[0]
                        x1_f, y1_f, x2_f, y2_f = map(int, f.bbox)
                        face_crop = person_crop[y1_f:y2_f, x1_f:x2_f]
                        if face_crop.size > 0:
                            face_thumb_bytes = self.image_to_bytes(face_crop)
                    
                    # Process each face
                    for face in faces:
                        face_emb = face.normed_embedding
                        point_id = str(uuid.uuid4())
                        
                        # Insert into Qdrant (thread-safe)
                        with self.lock:
                            try:
                                self.qdrant.upsert(
                                    collection_name=group_id,
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
                            except Exception as e:
                                print(f"❌ Failed to insert into Qdrant for image {image_id}: {str(e)}")
                                continue

                        records.append({
                            "id": point_id,
                            "image_id": image_id,
                            "person_id": -1,
                            "face_thumb_bytes": face_thumb_bytes
                        })

            print(f"✅ Processed image {image_id}: {len(records)} faces detected")
            return {"image_id": image_id, "success": True, "records": records}
            
        except Exception as e:
            print(f"❌ Error processing image {image_id}: {str(e)}")
            traceback.print_exc()
            return {"image_id": image_id, "success": False, "error": str(e), "records": []}

    def process_images_batch(self, images_batch, yolo_model, group_id):
        """Process a batch of images in parallel"""
        print(f"🔃 Processing batch of {len(images_batch)} images with {PARALLEL_LIMIT} parallel workers")
        
        results = []
        
        # Process images in chunks of PARALLEL_LIMIT
        for i in range(0, len(images_batch), PARALLEL_LIMIT):
            chunk = images_batch[i:i + PARALLEL_LIMIT]
            chunk_num = i // PARALLEL_LIMIT + 1
            total_chunks = (len(images_batch) + PARALLEL_LIMIT - 1) // PARALLEL_LIMIT
            
            print(f"🔃 Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} images)")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_LIMIT) as executor:
                futures = [
                    executor.submit(self.process_single_image, image_data, yolo_model, group_id)
                    for image_data in chunk
                ]
                
                chunk_results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        chunk_results.append(result)
                    except Exception as e:
                        print(f"❌ Thread execution error: {str(e)}")
                        chunk_results.append({"success": False, "error": str(e), "records": []})
                
                results.extend(chunk_results)
            
            print(f"✅ Completed chunk {chunk_num}/{total_chunks}")
        
        return results

    def process_group(self, group_id, yolo_model):
        """Process all images for a group in batches"""
        print(f"🔃 Starting processing for group {group_id}")
        
        # Setup collection
        if not self.setup_collection(group_id):
            print(f"❌ Failed to setup collection for group {group_id}")
            return False
        
        # Fetch all images for this group
        all_images = fetch_unprocessed_images(group_id)
        if not all_images:
            print(f"✅ No images to process for group {group_id}")
            return True
        
        total_processed = 0
        total_failed = 0
        
        # Process images in batches of BATCH_SIZE
        for i in range(0, len(all_images), BATCH_SIZE):
            batch = all_images[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(all_images) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"🔃 Processing batch {batch_num}/{total_batches} ({len(batch)} images) for group {group_id}")
            
            # Process batch
            batch_results = self.process_images_batch(batch, yolo_model, group_id)
            
            # Collect successful results and processed image IDs
            successful_records = []
            processed_image_ids = []
            
            for result in batch_results:
                if result["success"]:
                    successful_records.extend(result["records"])
                    processed_image_ids.append(result["image_id"])
                    total_processed += 1
                else:
                    total_failed += 1
                    print(f"⚠️ Failed to process image {result.get('image_id', 'unknown')}: {result.get('error', 'unknown error')}")
            
            # Insert faces to database
            if successful_records:
                faces_inserted = insert_faces_batch_with_retry(successful_records, group_id)
                if not faces_inserted:
                    print(f"❌ Failed to insert faces for batch {batch_num}")
            
            # Mark images as processed
            if processed_image_ids:
                images_marked = mark_images_processed_batch_with_retry(processed_image_ids)
                if not images_marked:
                    print(f"❌ Failed to mark images as processed for batch {batch_num}")
            
            print(f"✅ Completed batch {batch_num}/{total_batches}. Batch: {len(processed_image_ids)} processed, {len(successful_records)} faces. Total: {total_processed} processed, {total_failed} failed")
        
        print(f"✅ Completed group {group_id}. Total: {total_processed}/{len(all_images)} processed successfully")
        return True

if __name__ == "__main__":
    print("🚀 Starting Face Indexing Process")
    
    try:
        yolo_model = YOLO("yolov8x.pt")
        indexer = HybridFaceIndexer()
        
        groups = [row[0] for row in fetch_warm_groups()]
        print(f"ℹ️ Found {len(groups)} warm groups")
        
        for group_id in groups:
            print(f"\n🔃 Processing group {group_id}")
            success = indexer.process_group(group_id, yolo_model)
            
            if success:
                print(f"✅ Successfully completed group {group_id}")
            else:
                print(f"❌ Failed to process group {group_id}")
        
        print("🎉 Face indexing process completed!")
        
    except Exception as e:
        print(f"❌ Fatal error in main process: {str(e)}")
        traceback.print_exc()