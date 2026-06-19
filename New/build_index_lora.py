"""
LoRA/Bingsu CLIP용 FAISS image index 빌드 스크립트

필요 폴더:
    data/items.json
    data/images/{atcId}.jpg
    lora_weights/best/

실행:
    python build_index_lora.py

출력:
    embeddings_lora/index.faiss
    embeddings_lora/metadata.pkl
"""

import json
import os
import pickle

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from clipback.config import CAMPUS_CATEGORIES, DEVICE
from clipback.search_engine import load_model

DATA_PATH = "data/items.json"
IMG_DIR = "data/images"
SAVE_DIR = "embeddings_lora"

os.makedirs(SAVE_DIR, exist_ok=True)


def get_image_embedding(model, processor, img: Image.Image) -> np.ndarray:
    inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            emb = model.get_image_features(**inputs)
        else:
            out = model.vision_model(**inputs)
            emb = model.visual_projection(out.pooler_output)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()[0]


def build_index():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} 파일이 없습니다.")
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"{IMG_DIR} 폴더가 없습니다.")

    print("LoRA 모델 로딩 중...")
    model, processor = load_model("lora")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    allowed_categories = set(CAMPUS_CATEGORIES) - {"전체"}
    items = [
        item for item in items
        if str(item.get("prdtClNm", "")).split(" > ")[0] in allowed_categories
    ]
    print(f"카테고리 필터링 후: {len(items)}개")

    embeddings = []
    valid_items = []

    for i, item in enumerate(items):
        atc_id = item.get("atcId")
        img_path = os.path.join(IMG_DIR, f"{atc_id}.jpg")

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            emb = get_image_embedding(model, processor, img)
        except Exception as e:
            print(f"skip {atc_id}: {e}")
            continue

        embeddings.append(emb)
        valid_items.append(item)

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(items)} 처리 중... 유효 {len(valid_items)}개")

    if not embeddings:
        raise RuntimeError("유효한 이미지 embedding이 없습니다.")

    embeddings = np.array(embeddings).astype("float32")
    # normalize는 이미 했지만 안전하게 한 번 더
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = os.path.join(SAVE_DIR, "index.faiss")
    meta_path = os.path.join(SAVE_DIR, "metadata.pkl")

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(valid_items, f)

    print("완료")
    print(f"index: {index_path}")
    print(f"metadata: {meta_path}")
    print(f"개수: {len(valid_items)}")


if __name__ == "__main__":
    build_index()
