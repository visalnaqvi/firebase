import os
import shutil
import time
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from collections import deque
# 🧠 Setup Face Model
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# 👗 Setup FashionCLIP Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fashion_model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True).to(device)
fashion_processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)

def extract_cloth_embedding(path):
    img = Image.open(path).convert('RGB')
    inputs = fashion_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = fashion_model.get_image_features(**inputs)
        return emb[0] / emb[0].norm()

def extract_face_embedding(path):
    img = cv2.imread(path)
    if img is None:
        return None
    faces = face_app.get(img)
    return faces[0].normed_embedding if faces else None

# 🚀 Grouping Logic
from collections import deque

def group_images(folder_in, folder_out, face_th=0.7, cloth_th=0.8):
    os.makedirs(folder_out, exist_ok=True)
    items = []

    for f in os.listdir(folder_in):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        p = os.path.join(folder_in, f)
        fe = extract_face_embedding(p)
        ce = extract_cloth_embedding(p)
        items.append({'path': p, 'face': fe, 'cloth': ce, 'assigned': False})

    groups = []

    for idx, item in enumerate(items):
        if item['assigned']:
            continue

        group = [item['path']]
        item['assigned'] = True
        queue = deque([item])

        while queue:
            current = queue.popleft()
            for other in items:
                if other['assigned']:
                    continue

                face_sim = (
                    1 - cosine(current['face'], other['face'])
                    if current['face'] is not None and other['face'] is not None
                    else 0
                )
                cloth_sim = float((current['cloth'] @ other['cloth']).cpu())
                
                print(f"Comparing {os.path.basename(current['path'])} vs {os.path.basename(other['path'])}")
                print(f"  face sim: {face_sim:.3f}, cloth sim: {cloth_sim:.3f}")

                if face_sim >= face_th or (face_sim >= 0.4 and cloth_sim >= cloth_th):
                    other['assigned'] = True
                    group.append(other['path'])
                    queue.append(other)  # 🔁 Keep checking from here

        groups.append(group)

    for idx, grp in enumerate(groups, 1):
        group_dir = os.path.join(folder_out, f'person_{idx:03d}')
        os.makedirs(group_dir, exist_ok=True)
        for img in grp:
            shutil.copy(img, os.path.join(group_dir, os.path.basename(img)))

    print(f"✅ Grouped into {len(groups)} people from {len(items)} images.")


# 🔧 Run
if __name__ == "__main__":
    start = time.time()
    group_images('cropped_people', 'grouped_faces_clothes', face_th=0.7, cloth_th=0.85)
    print(f"Time: {time.time()-start:.2f}s")
