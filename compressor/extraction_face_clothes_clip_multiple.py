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


def extract_face_and_cloth_embeddings_per_person(path):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Couldn't read image: {path}")
        return []

    faces = face_app.get(img)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results = []

    for face in faces:
        face_embedding = face.normed_embedding
        x1, y1, x2, y2 = map(int, face.bbox)

        height = y2 - y1
        y2_expanded = min(img.shape[0], y2 + int(height * 1.5))
        x1_crop, y1_crop, x2_crop, y2_crop = max(0, x1), max(0, y1), min(img.shape[1], x2), y2_expanded

        cropped = pil_img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
        inputs = fashion_processor(images=cropped, return_tensors="pt").to(device)
        with torch.no_grad():
            cloth_emb = fashion_model.get_image_features(**inputs)
            cloth_embedding = cloth_emb[0] / cloth_emb[0].norm()

        results.append({
            'face': face_embedding,
            'cloth': cloth_embedding,
            'bbox': (x1, y1, x2, y2),
        })

    return results


def group_images(folder_in, folder_out, face_th=0.7, cloth_th=0.8):
    os.makedirs(folder_out, exist_ok=True)
    items = []

    for f in os.listdir(folder_in):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        p = os.path.join(folder_in, f)
        embeddings = extract_face_and_cloth_embeddings_per_person(p)
        for i, emb in enumerate(embeddings):
            items.append({
                'path': p,
                'face': emb['face'],
                'cloth': emb['cloth'],
                'bbox': emb['bbox'],
                'assigned': False,
                'instance_id': f"{os.path.splitext(f)[0]}_{i}"
            })

    groups = []

    for idx, item in enumerate(items):
        if item['assigned']:
            continue

        group = [item]
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

                print(f"Comparing {current['instance_id']} vs {other['instance_id']}")
                print(f"  face sim: {face_sim:.3f}, cloth sim: {cloth_sim:.3f}")

                if face_sim >= face_th or cloth_sim >= cloth_th:
                    other['assigned'] = True
                    group.append(other)
                    queue.append(other)

        groups.append(group)

    for idx, grp in enumerate(groups, 1):
        group_dir = os.path.join(folder_out, f'person_{idx:03d}')
        os.makedirs(group_dir, exist_ok=True)
        for person in grp:
            base = os.path.basename(person['path'])
            name = f"{os.path.splitext(base)[0]}_{person['bbox'][0]:04d}_{person['bbox'][1]:04d}.jpg"
            shutil.copy(person['path'], os.path.join(group_dir, name))

    print(f"✅ Grouped into {len(groups)} people from {len(items)} detected persons.")


# 🔧 Run
if __name__ == "__main__":
    start = time.time()
    group_images('cl_img', 'grouped_faces_clothes', face_th=0.7, cloth_th=0.85)
    print(f"⏱️ Time: {time.time()-start:.2f}s")
