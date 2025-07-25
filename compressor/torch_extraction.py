import os
import torch
import shutil
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import torchreid

# ------------------ Setup ------------------

# Init device
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

# ------------------ Grouping Logic ------------------

def extract_clothes_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)
        embedding = embedding.cpu().numpy().flatten()
    return embedding

def group_by_clothing(image_folder, output_folder, threshold=0.7):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    groups = []  # Each group: list of (embedding, image_path)

    for img_path in image_paths:
        emb = extract_clothes_embedding(img_path)

        matched = False
        for idx, group in enumerate(groups):
            # Compare with the first image's embedding in each group
            existing_emb = group[0][0]
            sim = cosine_similarity(
                torch.tensor(emb).unsqueeze(0),
                torch.tensor(existing_emb).unsqueeze(0)
            ).item()

            if sim > threshold:
                groups[idx].append((emb, img_path))
                matched = True
                break

        if not matched:
            groups.append([(emb, img_path)])

    # Save grouped images
    for i, group in enumerate(groups, start=1):
        group_folder = os.path.join(output_folder, f'person_{i:03d}')
        os.makedirs(group_folder, exist_ok=True)

        for _, path in group:
            shutil.copy(path, os.path.join(group_folder, os.path.basename(path)))

    print(f"Done: {len(groups)} groups created.")

# ------------------ Run ------------------

# Example usage
group_by_clothing(
    image_folder='cl_img',
    output_folder='grouped_by_clothes',
    threshold=0.75  # Can adjust to 0.6–0.8 based on performance
)
