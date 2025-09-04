import os
import numpy as np
import torch
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import DictCursor
import uuid
from collections import defaultdict
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

class ImageSimilarityGrouper:
    def __init__(self, host="localhost", port=6333):
        self.qdrant = QdrantClient(host=host, port=port)

    def get_unprocessed_images_batch(self, group_id, limit=50):
        """Get unprocessed images from PostgreSQL - those with similar_image_id = '-'"""
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        print(f"🔃 Fetching {limit} unprocessed images for group {group_id}")
        # Query for similar_image_id = '-' (string with hyphen)
        cursor.execute(
            "SELECT id FROM images WHERE group_id = %s AND similar_image_id = %s LIMIT %s", 
            (group_id, '-', limit)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        image_ids = [str(row["id"]) for row in rows]  # Ensure string format for Qdrant
        print(f"✅ Found {len(image_ids)} unprocessed images")
        return image_ids

    def get_image_embedding(self, group_id, image_id):
        """Get single image embedding from Qdrant image collection"""
        try:
            collection_name = f"image_{group_id}"
            points = self.qdrant.retrieve(
                collection_name=collection_name,
                ids=[str(image_id)],
                with_payload=True,
                with_vectors=True
            )
            
            if not points or len(points) == 0:
                print(f"⚠️ No embedding found for image {image_id}")
                return None

            point = points[0]
            vector = getattr(point, "vector", None)
            payload = getattr(point, "payload", {}) or {}
            
            if vector:
                return {
                    "embedding": np.array(vector),
                    "image_id": payload.get("image_id"),
                    "group_id": payload.get("group_id")
                }

            print(f"⚠️ No vector found for image {image_id}")
            return None

        except Exception as e:
            print(f"⚠️ Error retrieving embedding for image {image_id}: {e}")
            return None

    def find_similar_images(self, image_embedding, group_id, current_image_id, threshold=0.95, limit=20):
        """Find similar images using high similarity threshold"""
        try:
            if image_embedding is None:
                print("⚠️ find_similar_images called with None embedding")
                return []

            collection_name = f"image_{group_id}"
            
            # Exclude current image from search
            filter_obj = Filter(
                must_not=[
                    FieldCondition(key="image_id", match=MatchValue(value=current_image_id))
                ]
            )

            candidates = self.qdrant.query_points(
                collection_name=collection_name,
                query=image_embedding.tolist(),
                score_threshold=threshold,
                limit=limit,
                with_payload=True,
                query_filter=filter_obj
            )

            points = getattr(candidates, "points", None)
            if points is None:
                if isinstance(candidates, list):
                    points = candidates
                elif isinstance(candidates, dict) and "points" in candidates:
                    points = candidates["points"]
                else:
                    print("⚠️ qdrant.query_points returned no points")
                    return []

            similar_images = []
            print(f"Found {len(points)} similar image candidate(s)")

            for candidate in points:
                payload = candidate.payload or {}
                similar_images.append({
                    "id": candidate.id,
                    "score": candidate.score,
                    "image_id": payload.get("image_id")
                })

            return similar_images

        except Exception as e:
            print(f"❌ Error finding similar images: {e}")
            return []

    def check_existing_group(self, image_ids):
        """Check if any of the similar images already belong to a similarity group"""
        if not image_ids:
            return None
            
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        try:
            # Convert to strings and create format string for IN clause
            str_ids = [str(img_id) for img_id in image_ids]
            format_strings = ','.join(['%s'] * len(str_ids))
            
            # Exclude '-' from existing groups check and look for actual UUIDs
            cursor.execute(
                f"SELECT DISTINCT similar_image_id FROM images WHERE id IN ({format_strings}) AND similar_image_id IS NOT NULL AND similar_image_id != %s",
                str_ids + ['-']
            )
            
            existing_groups = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if existing_groups:
                # Return the first existing group ID found
                return existing_groups[0]["similar_image_id"]
            
            return None
            
        except Exception as e:
            print(f"❌ Error checking existing groups: {e}")
            cursor.close()
            conn.close()
            return None

    def update_similar_image_ids(self, image_ids, similar_image_id):
        """Update similar_image_id for a group of images"""
        if not image_ids:
            return
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Convert to strings for database update
            str_ids = [str(img_id) for img_id in image_ids]
            format_strings = ','.join(['%s'] * len(str_ids))
            
            cursor.execute(
                f"UPDATE images SET similar_image_id = %s WHERE id IN ({format_strings})",
                [similar_image_id] + str_ids
            )
            
            affected_rows = cursor.rowcount
            conn.commit()
            print(f"✅ Updated {affected_rows} images with similar_image_id: {similar_image_id}")
            
            # Debug: Show what was actually updated
            if affected_rows > 0:
                cursor.execute(
                    f"SELECT id, similar_image_id FROM images WHERE id IN ({format_strings})",
                    str_ids
                )
                updated_rows = cursor.fetchall()
                print(f"   📋 Updated images: {[(row[0], row[1]) for row in updated_rows[:5]]}")  # Show first 5
            
        except Exception as e:
            print(f"❌ Error updating similar_image_ids: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def process_image_batch(self, group_id, batch_size=50):
        """Process a batch of unprocessed images for similarity grouping"""
        print(f"🚀 Processing batch of {batch_size} images for group {group_id}")
        
        # Get batch of unprocessed images
        unprocessed_image_ids = self.get_unprocessed_images_batch(group_id, batch_size)
        if not unprocessed_image_ids:
            print("No unprocessed images found")
            return False
        
        processed_images = set()  # Track processed images to avoid duplicates
        
        for image_id in unprocessed_image_ids:
            if image_id in processed_images:
                continue
                
            print(f"\n🔍 Processing image {image_id}")
            
            # Get image embedding
            embedding_data = self.get_image_embedding(group_id, image_id)
            if not embedding_data:
                print(f"   ⚠️ Could not get embedding for image {image_id}")
                # If we can't get embedding but image has similar_image_id = '-', 
                # we should still mark it as processed to avoid infinite loop
                self.update_similar_image_ids([image_id], 'no_embedding')
                continue
            
            image_emb = embedding_data['embedding']
            
            # Find similar images
            similar_images = self.find_similar_images(
                image_emb, group_id, image_id, threshold=0.95, limit=20
            )
            
            if not similar_images:
                print(f"   📋 No similar images found for {image_id}")
                # Mark as processed but not similar to anything
                self.update_similar_image_ids([image_id], 'no_similar')
                processed_images.add(image_id)
                continue
            
            print(f"   📋 Found {len(similar_images)} similar images")
            
            # Extract image IDs for grouping
            similar_image_ids = [img["image_id"] for img in similar_images if img["image_id"]]
            all_image_ids = [image_id] + similar_image_ids
            
            # Remove duplicates while preserving order
            unique_image_ids = []
            for img_id in all_image_ids:
                if img_id not in unique_image_ids:
                    unique_image_ids.append(img_id)
            
            # Check if any images already belong to a similarity group
            existing_group_id = self.check_existing_group(unique_image_ids)
            
            if existing_group_id:
                print(f"   🔗 Found existing group {existing_group_id}, adding images to it")
                similar_image_id = existing_group_id
            else:
                # Generate new UUID for this similarity group
                similar_image_id = str(uuid.uuid4())
                print(f"   🆕 Creating new similarity group: {similar_image_id}")
            
            # Update database with similarity group ID
            self.update_similar_image_ids(unique_image_ids, similar_image_id)
            
            # Mark these images as processed to avoid reprocessing in this batch
            processed_images.update(unique_image_ids)
            
            print(f"   ✅ Grouped {len(unique_image_ids)} images under ID: {similar_image_id}")
        
        print(f"🎉 Batch processing complete! Processed {len(processed_images)} unique images")
        return True

    def process_unprocessed_images(self, group_id, batch_size=50):
        """Process all unprocessed images in batches"""
        print(f"🚀 Starting image similarity processing for group {group_id}")
        
        batch_count = 0
        while True:
            batch_count += 1
            print(f"\n📦 Processing batch #{batch_count}")
            
            # Process one batch
            has_more = self.process_image_batch(group_id, batch_size)
            if not has_more:
                print("✅ All images processed!")
                break

    def mark_group_processed(self, group_id):
        """Mark group as processed"""
        if not group_id:
            return
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE groups
                SET last_processed_at = NOW(),
                    last_processed_step = 'similarity_grouping'
                WHERE id = %s
            """, (group_id,))
            conn.commit()
            print(f"Marked group {group_id} as processed for similarity grouping")
            
        except Exception as e:
            print(f"❌ Error marking group as processed: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def process_all_groups(self, batch_size=50):
        """Process all groups that have warmed status"""
        # Get all warmed groups (assuming these have image embeddings ready)
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        cursor.execute("SELECT id FROM groups WHERE id = 8")
        groups = cursor.fetchall()
        cursor.close()
        conn.close()
        
        group_ids = [row["id"] for row in groups]
        print(f"📋 Found {len(group_ids)} groups to process for image similarity")
        
        for group_id in group_ids:
            try:
                self.process_unprocessed_images(group_id, batch_size)
                self.mark_group_processed(group_id)
                print(f"✅ Completed image similarity processing for group {group_id}")
            except Exception as e:
                print(f"❌ Error processing group {group_id}: {e}")
                continue

# 🔧 Usage Example
if __name__ == "__main__":
    grouper = ImageSimilarityGrouper()
    
    # Process all groups with batch size of 50
    grouper.process_all_groups(batch_size=50)
    
    # Or process a specific group
    # grouper.process_unprocessed_images("specific_group_id", batch_size=50)