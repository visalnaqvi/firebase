import io, zlib
from PIL import Image
import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'gallery-585ee.firebasestorage.app' 
})

def upload_to_firebase(compressed_data, filename):
    bucket = storage.bucket()
    blob = bucket.blob(f"compressed_images/{filename}")
    
    # Upload bytes directly
    blob.upload_from_string(compressed_data, content_type="application/octet-stream")
    
    # Optional: make URL accessible
    blob.make_public()
    print("Uploaded to:", blob.public_url)
    return blob.public_url


def compress_image_lossless(image_path):
    # Open image and convert to PNG (lossless)
    img = Image.open(image_path)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)

    # Compress with zlib
    compressed_data = zlib.compress(buffer.getvalue())

    # Store this `compressed_data` in your DB or cloud storage (as binary, not base64 ideally)
    return compressed_data


def restore_image_from_bytes(compressed_data, output_path="restored_image.png"):
    decompressed_bytes = zlib.decompress(compressed_data)

    # Load as PIL image
    image = Image.open(io.BytesIO(decompressed_bytes))
    image.save(output_path)  # or stream directly to frontend
from PIL import Image

def save_pixel_data_to_text(image_path, output_txt_path):
    # Open image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())  # List of (R, G, B) tuples

    # Write pixel values to a text file
    with open(output_txt_path, "w") as f:
        for pixel in pixels:
            f.write(f"{pixel[0]},{pixel[1]},{pixel[2]}\n")  # Each line: R,G,B

    print(f"Pixel data saved to {output_txt_path}")

save_pixel_data_to_text("yo.jpg" , "yotexct.txt")

# with open("image.jpg", "rb") as f:
#     byte_data = f.read()

# with open("bytes_hex.txt", "w") as f:
#     f.write(byte_data.hex())  # Converts to hex string
