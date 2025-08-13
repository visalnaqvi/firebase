import psycopg2
import numpy as np
import cv2
from PIL import Image
import torch
import open_clip

# ==== DB connection ====
conn = psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="admin")
cur = conn.cursor()

# ==== Fetch one image byte ====
cur.execute("SELECT image_byte FROM images WHERE image_byte IS NOT NULL LIMIT 1;")
row = cur.fetchone()
if not row:
    raise ValueError("No image found in DB")
image_bytes = row[0]  # bytea from PostgreSQL

# ==== Decode to NumPy (BGR) ====
nparr = np.frombuffer(image_bytes, np.uint8)
bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# ==== Convert to PIL (RGB) ====
pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

# ==== Load OpenCLIP model + preprocess ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
)

# ==== Preprocess and embed ====
img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

with torch.no_grad():
    emb = model.encode_image(img_tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize

print("Embedding shape:", emb.shape)
print("First 5 values:", emb[0, :5].cpu().numpy())

# ==== Cleanup ====
cur.close()
conn.close()
