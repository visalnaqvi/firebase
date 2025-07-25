import os
import shutil
from qdrant_client import QdrantClient
from collections import defaultdict

def organize_images_by_person_id(output_folder="organized_by_person", collection_name="people_collection"):
    """
    Reads all points from Qdrant collection and organizes images into folders by person_id
    """
    # Initialize Qdrant client
    qdrant = QdrantClient(host="localhost", port=6333)
    
    # Check if collection exists
    if not qdrant.collection_exists(collection_name):
        print(f"❌ Collection '{collection_name}' does not exist!")
        return
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all points from the collection
    print("📖 Reading all points from Qdrant collection...")
    
    # Scroll through all points in the collection
    points_dict = defaultdict(list)
    offset = None
    
    while True:
        # Get batch of points
        result = qdrant.scroll(
            collection_name=collection_name,
            limit=100,  # Batch size
            offset=offset,
            with_payload=True,
            with_vectors=False  # We don't need vectors, just payload
        )
        
        points, next_offset = result
        
        if not points:
            break
            
        # Group points by person_id
        for point in points:
            person_id = point.payload.get('person_id', -1)
            image_path = point.payload.get('image_path', '')
            
            if image_path and os.path.exists(image_path):
                points_dict[person_id].append({
                    'id': point.id,
                    'image_path': image_path,
                    'face_id': point.payload.get('face_id', point.id)
                })
            else:
                print(f"⚠️  Image not found: {image_path}")
        
        offset = next_offset
        if offset is None:
            break
    
    print(f"📊 Found {sum(len(images) for images in points_dict.values())} images across {len(points_dict)} person groups")
    
    # Create folders and copy images
    copied_count = 0
    
    for person_id, images in points_dict.items():
        if person_id == -1:
            folder_name = "unassigned"
        else:
            folder_name = f"person_{person_id:03d}"
        
        person_folder = os.path.join(output_folder, folder_name)
        os.makedirs(person_folder, exist_ok=True)
        
        print(f"📁 Creating folder '{folder_name}' with {len(images)} images")
        
        for img_info in images:
            source_path = img_info['image_path']
            filename = os.path.basename(source_path)
            destination_path = os.path.join(person_folder, filename)
            
            try:
                # Handle duplicate filenames by adding face_id
                if os.path.exists(destination_path):
                    name, ext = os.path.splitext(filename)
                    destination_path = os.path.join(person_folder, f"{name}_face{img_info['face_id']}{ext}")
                
                shutil.copy2(source_path, destination_path)
                copied_count += 1
                
            except Exception as e:
                print(f"❌ Error copying {source_path}: {e}")
    
    print(f"✅ Successfully organized {copied_count} images into {len(points_dict)} folders")
    print(f"📂 Output directory: {os.path.abspath(output_folder)}")


def get_collection_stats(collection_name="people_collection"):
    """
    Get statistics about the collection
    """
    qdrant = QdrantClient(host="localhost", port=6333)
    
    if not qdrant.collection_exists(collection_name):
        print(f"❌ Collection '{collection_name}' does not exist!")
        return
    
    # Get collection info
    collection_info = qdrant.get_collection(collection_name)
    total_points = collection_info.points_count
    
    print(f"📊 Collection Statistics:")
    print(f"   Total points: {total_points}")
    
    # Count by person_id
    person_counts = defaultdict(int)
    offset = None
    
    while True:
        result = qdrant.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = result
        
        if not points:
            break
            
        for point in points:
            person_id = point.payload.get('person_id', -1)
            person_counts[person_id] += 1
        
        offset = next_offset
        if offset is None:
            break
    
    print(f"   Unique person IDs: {len(person_counts)}")
    for person_id, count in sorted(person_counts.items()):
        if person_id == -1:
            print(f"   Unassigned: {count} images")
        else:
            print(f"   Person {person_id}: {count} images")


if __name__ == "__main__":
    # Show collection statistics first
    print("=" * 50)
    get_collection_stats()
    print("=" * 50)
    
    # Organize images
    organize_images_by_person_id(
        output_folder="organized_by_person_id",
        collection_name="people_collection"
    )