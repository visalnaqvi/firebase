import os
import shutil
import time
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import os
import torch
import shutil
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import torchreid
# Initialize InsightFace model
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'cuda' if available
face_app.prepare(ctx_id=0)  # Use 0 for CPU or CUDA device index for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Re-ID model (clothes-based)
model = torchreid.models.build_model(
    name='osnet_x1_0', num_classes=1000, pretrained=True
)
model.eval()
model.to(device)

# Image preprocessing for model
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def extract_clothes_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)
        embedding = embedding.cpu().numpy().flatten()
    return embedding

def extract_embedding(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Couldn't read image: {image_path}")
            return []

        faces = face_app.get(img)
        results = []
        cl_emb = extract_clothes_embedding(image_path)
        for face in faces:
            embedding = face.normed_embedding  # Already L2 normalized
            results.append({
                "face_embedding": embedding,
                "image_path": image_path,
                "seen": False,
                "cloth_embedding":cl_emb
            })

        return results
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return []

# === Main comparison logic ===
if __name__ == "__main__":
    start_time = time.time()
    
    IMAGE_FOLDER = "./cl_img"
    OUTPUT_FOLDER = "./grouped_faces_clothes"
    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    image_paths = [
        os.path.join(IMAGE_FOLDER, file)
        for file in os.listdir(IMAGE_FOLDER)
        if file.lower().endswith(SUPPORTED_EXTS)
    ]
    
    all_embedding = []

    # Step 1: Extract embeddings from all images
    for path in image_paths:
        all_embedding.extend(extract_embedding(path))

    threshold = 0.6  # Adjust based on your tests; InsightFace uses cosine similarity
    person_id = 1
    groups = []

    # Step 2: Group similar faces
    for i in range(len(all_embedding)):
        if all_embedding[i]["seen"]:
            continue
        group = set()
        print(f"processing for img {all_embedding[i]['image_path']}")
        group.add(all_embedding[i]["image_path"])
        all_embedding[i]["seen"] = True

        for j in range(i + 1, len(all_embedding)):
            if not all_embedding[j]["seen"]:
                sim_face = 1 - cosine(all_embedding[i]["face_embedding"], all_embedding[j]["face_embedding"])
                sim_cloth = 1 - cosine(all_embedding[i]["cloth_embedding"], all_embedding[j]["cloth_embedding"])
                sim = (0.8 * sim_face + 0.4 * sim_cloth) / (0.8 + 0.4) 
                print(f"face sim {sim_face}")
                print(f"clothe sim {sim_cloth}")
                if sim_face > 0.7 or sim_cloth > 0.9:
                    print(f"adding to group")
                    group.add(all_embedding[j]["image_path"])
                    all_embedding[j]["seen"] = True

        groups.append(group)

    # Step 3: Copy grouped images
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for group in groups:
        folder_name = os.path.join(OUTPUT_FOLDER, f"person_img_{person_id}")
        os.makedirs(folder_name, exist_ok=True)
        for img in group:
            img_name = os.path.basename(img)
            dest_path = os.path.join(folder_name, img_name)
            shutil.copy(img, dest_path)
        person_id += 1

    print(f"✅ Grouped faces into {len(groups)} people")
    print(f"✅ Total time taken: {time.time() - start_time:.2f} seconds")
