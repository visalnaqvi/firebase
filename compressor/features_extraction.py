from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
import cv2
from scipy.spatial.distance import cosine
import os

# 1. Load FastReID config + pretrained model
cfg = get_cfg()
cfg.merge_from_file("fastreid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "model_final.pth"  # downloaded from FastReID model zoo
cfg.MODEL.DEVICE = "cpu"              # or "cuda"
predictor = DefaultPredictor(cfg)

def extract_feature(img_path):
    im = cv2.imread(img_path)
    outputs = predictor(im)
    return outputs["instances"].features.cpu().numpy()[0]  # 1×512 feature




IMAGE_FOLDER = "./img"
OUTPUT_FOLDER = "./grouped_by_features"
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

images = [
    os.path.join(IMAGE_FOLDER, file)
    for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(SUPPORTED_EXTS)
]
features = [extract_feature(p) for p in images]

TH = 0.75
groups = []
used = set()

for i, f in enumerate(features):
    if i in used:
        continue
    group = [images[i]]
    used.add(i)
    for j in range(i + 1, len(features)):
        if j in used:
            continue
        if 1 - cosine(f, features[j]) > TH:
            group.append(images[j])
            used.add(j)
    groups.append(group)

print("Grouped:", groups)
