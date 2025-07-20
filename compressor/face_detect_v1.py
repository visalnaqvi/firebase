from deepface import DeepFace
from scipy.spatial.distance import cosine
import shutil
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from functools import partial
import time

def extract_embedding(image_path, detector_backend="retinaface"):
    """Extract face embeddings from a single image"""
    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend=detector_backend,
            enforce_detection=True
        )
        results = []
        for obj in embedding_obj:
            results.append({
                "embedding": np.array(obj["embedding"]),  # Convert to numpy array
                "image_path": image_path
            })
        return results
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return []

def extract_embeddings_parallel(image_paths, detector_backend="retinaface", max_workers=2):
    """Extract embeddings from multiple images in parallel"""
    start_time = time.time()
    all_faces = []
    
    # Use partial to fix the detector_backend parameter
    extract_func = partial(extract_embedding, detector_backend=detector_backend)
    
    # Use timeout to prevent hanging
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(extract_func, path): path for path in image_paths}
        
        # Collect results as they complete with timeout
        for future in concurrent.futures.as_completed(future_to_path, timeout=300):  # 5 min timeout per image
            path = future_to_path[future]
            try:
                result = future.result(timeout=60)  # 1 min timeout for result
                all_faces.extend(result)
                print(f"✅ Processed {path}")
            except concurrent.futures.TimeoutError:
                print(f"⏰ Timeout processing {path}")
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")
    
    end_time = time.time()
    print(f"⏱️ Embedding extraction took: {end_time - start_time:.2f} seconds")
    return all_faces

def extract_embeddings_sequential(image_paths, detector_backend="retinaface"):
    """Extract embeddings sequentially - more reliable but slower"""
    start_time = time.time()
    all_faces = []
    
    for i, path in enumerate(image_paths, 1):
        print(f"🔄 Processing {path} ({i}/{len(image_paths)})...")
        try:
            result = extract_embedding(path, detector_backend)
            all_faces.extend(result)
            print(f"✅ Processed {path}")
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")
    
    end_time = time.time()
    print(f"⏱️ Embedding extraction took: {end_time - start_time:.2f} seconds")
    return all_faces
    """Group faces using DBSCAN clustering - much faster than nested loops"""
    if len(embeddings) == 0:
        return []
    
    start_time = time.time()
    
    # Convert to numpy array for efficient computation
    embeddings_array = np.array(embeddings)
    
    # Normalize embeddings to ensure proper cosine similarity calculation
    embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    # Calculate cosine similarity matrix efficiently
    similarity_matrix = cosine_similarity(embeddings_array)
    
    # Convert similarity to distance (DBSCAN uses distance)
    # Clip values to ensure they're in valid range [0, 1] to avoid floating-point errors
    similarity_matrix = np.clip(similarity_matrix, 0, 1)
    distance_matrix = 1 - similarity_matrix
    
    # Ensure distance matrix is symmetric and has zero diagonal
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Use DBSCAN clustering
    # eps = 1 - threshold (distance threshold)
    # min_samples = 1 (minimum faces to form a group)
    clustering = DBSCAN(eps=1-threshold, min_samples=1, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)
    
    # Group images by cluster labels
    groups = {}
    for i, label in enumerate(labels):
        if label not in groups:
            groups[label] = set()
        groups[label].add(image_paths[i])
    
    end_time = time.time()
    print(f"⏱️ Face grouping took: {end_time - start_time:.2f} seconds")
    
    return list(groups.values())

def group_faces_optimized_loop(all_faces, threshold=0.7):
    """Optimized version of original nested loop approach"""
    if len(all_faces) == 0:
        return []
    
    # Pre-compute embeddings as numpy arrays
    embeddings = np.array([face["embedding"] for face in all_faces])
    
    seen = [False] * len(all_faces)
    groups = []
    
    for i in range(len(all_faces)):
        if seen[i]:
            continue
            
        group = set()
        group.add(all_faces[i]["image_path"])
        seen[i] = True
        
        # Vectorized similarity computation for remaining faces
        if i + 1 < len(all_faces):
            # Calculate similarities for all remaining unseen faces at once
            remaining_embeddings = embeddings[i + 1:]
            current_embedding = embeddings[i]
            
            # Compute cosine similarities efficiently
            similarities = cosine_similarity([current_embedding], remaining_embeddings)[0]
            
            for j, sim in enumerate(similarities):
                actual_idx = i + 1 + j
                if not seen[actual_idx] and sim > threshold:
                    group.add(all_faces[actual_idx]["image_path"])
                    seen[actual_idx] = True
        
        groups.append(group)
    
    return groups

def create_person_folders(groups, base_dir="output"):
    """Create folders and copy images"""
    start_time = time.time()
    
    # Clean up existing output directory
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    person_id = 1
    for group in groups:
        folder_name = os.path.join(base_dir, f"person_{person_id}")
        os.makedirs(folder_name, exist_ok=True)
        
        for img_path in group:
            dest_path = os.path.join(folder_name, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
        
        person_id += 1
    
    end_time = time.time()
    print(f"⏱️ File operations took: {end_time - start_time:.2f} seconds")

# === Main execution ===
if __name__ == "__main__":
    # Start total timer
    total_start_time = time.time()
    
    image_paths = ["cp_2.jpg", "cp_4.jpg", "cp_8.jpg", "cp_1.jpg", "cp_3.jpg", "cp_6.jpg"]
    threshold = 0.7
    
    print("🔄 Extracting embeddings...")
    
    # METHOD 1: Parallel extraction + DBSCAN clustering (RECOMMENDED)
    all_faces = extract_embeddings_parallel(image_paths, max_workers=3)
    
    if all_faces:
        embeddings = [face["embedding"] for face in all_faces]
        image_paths_extracted = [face["image_path"] for face in all_faces]
        
        print("🔄 Grouping faces using clustering...")
        groups = group_faces_clustering(embeddings, image_paths_extracted, threshold)
        
        print("🔄 Creating person folders...")
        create_person_folders(groups)
        
        # Calculate total time
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"✅ Grouped faces into {len(groups)} people")
        print(f"🎉 TOTAL PROCESS TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Print group details
        for i, group in enumerate(groups, 1):
            print(f"Person {i}: {list(group)}")
    else:
        print("❌ No faces found in any images")
    
    # ALTERNATIVE METHOD: Sequential extraction + optimized loop
    # Uncomment below to test the optimized loop version
    """
    print("\n--- Alternative Method ---")
    all_faces_seq = []
    for path in image_paths:
        all_faces_seq.extend(extract_embedding(path))
    
    groups_seq = group_faces_optimized_loop(all_faces_seq, threshold)
    print(f"Sequential + Optimized Loop: {len(groups_seq)} groups")
    """