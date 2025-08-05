from PIL import Image, ExifTags
import os
# import psycopg2
# from psycopg2.extras import execute_values

# DB_CONFIG = {
#     "dbname": "postgres",
#     "user": "postgres",
#     "password": "admin",
#     "host": "localhost",
#     "port": "5432"
# }

# def get_db_connection():
#     return psycopg2.connect(**DB_CONFIG)

# def insert_images_batch(records):
#     if not records:
#         return
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = """
#         INSERT INTO images_to_process (image_path, is_emb_extracted, is_assigned)
#         VALUES %s
#     """
#     execute_values(cursor, query, records)
#     conn.commit()
#     cursor.close()
#     conn.close()
#     print(f"✅ Inserted {len(records)} records in batch")

def correct_image_orientation(img):
    try:
        exif = img._getexif()
        if exif is not None:
            orientation_key = None
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_key = tag
                    break
            if orientation_key and orientation_key in exif:
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

def compress_image(input_path, output_path, max_width=3000, target_size_mb=1, quality_step=5):
    img = Image.open(input_path)
    img = correct_image_orientation(img)
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"\n📷 Original size of {os.path.basename(input_path)}: {original_size:.2f} MB")

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
            print(f"✅ Compressed successfully at quality={quality}")
            break
        quality -= quality_step

def compress_all_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    supported_ext = (".jpg", ".jpeg", ".png")

    batch_records = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_ext):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            compress_image(input_path, output_path, max_width=3000)

            batch_records.append((output_path, False, False))

    # insert_images_batch(batch_records)

if __name__ == "__main__":
    input_folder = "ds"               
    output_folder = "compressed_img" 
    compress_all_images(input_folder, output_folder)
    print("\n✅ All images compressed and inserted in batch")
