from PIL import Image, ExifTags
import face_recognition
import os
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict
import shutil

# 📌 Auto-rotate based on EXIF
def correct_image_orientation(img):
    try:
        exif = img._getexif()
        if exif is not None:
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_key = tag
                    break
            orientation = exif.get(orientation_key)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception as e:
        print("Warning: Could not auto-rotate image due to:", e)
    return img

# 📌 Compress image
def compress_image(input_path, output_path, max_width=3000, target_size_mb=1.5, quality_step=5):
    img = Image.open(input_path)
    img = correct_image_orientation(img)
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original size of {input_path}: {original_size:.2f} MB")

    if img.width > max_width:
        w_percent = max_width / float(img.width)
        h_size = int(float(img.height) * w_percent)
        img = img.resize((max_width, h_size), Image.LANCZOS)
        print(f"Resized to: {img.size}")

    quality = 95
    while quality > 10:
        img.save(output_path, "JPEG", quality=quality, optimize=True)
        compressed_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Trying quality={quality} → Size={compressed_size:.2f} MB")
        if compressed_size <= target_size_mb:
            print(f"✅ Compressed {output_path} successfully at quality={quality}")
            break
        quality -= quality_step

# 📌 Extract face embeddings with reference to original image
def extract_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)
    print(f"Found {len(encodings)} face(s) in {image_path}")
    return encodings

# 📌 Compress and process multiple images
def process_images(image_paths):
    embeddings = []
    image_labels = []

    for path in image_paths:
        compressed_path = f"compressed_{os.path.basename(path)}"
        compress_image(path, compressed_path, max_width=1000)

        faces = extract_faces(compressed_path)
        embeddings.extend(faces)
        image_labels.extend([(compressed_path, path)] * len(faces))  # Store both compressed and original

    return embeddings, image_labels

# 📌 Cluster faces and copy original image into folders
def cluster_faces(embeddings, image_labels, output_dir="output"):
    if not embeddings:
        print("❌ No faces found in any image.")
        return

    model = DBSCAN(eps=0.6, min_samples=1, metric='euclidean')
    labels = model.fit_predict(embeddings)

    grouped = defaultdict(set)  # person -> set of original image paths

    for label, (compressed_img, original_img) in zip(labels, image_labels):
        grouped[f"Person_{label + 1}"].add(original_img)

    print("\n📂 Organizing folders...")
    os.makedirs(output_dir, exist_ok=True)

    for person, images in grouped.items():
        person_folder = os.path.join(output_dir, person)
        os.makedirs(person_folder, exist_ok=True)
        print(f"{person}:")
        for img_path in images:
            dest_path = os.path.join(person_folder, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
            print(f" - Copied: {img_path} → {person_folder}")

# ✅ Main entry
if __name__ == "__main__":
    image_list = ["pic1.jpg", "pic2.jpg", "pic3.jpg", "pic4.jpg" ,"pic5.jpg", "pic6.jpg" , "pic7.jpg", "pic8.jpg" , "pic9.jpg", "pic10.jpg"]  # Replace with your DSLR image filenames
    embeddings, image_labels = process_images(image_list)
    cluster_faces(embeddings, image_labels, output_dir="people_folders")
