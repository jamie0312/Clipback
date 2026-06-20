import json
import requests
import numpy as np
import faiss
import pickle
import time
import torch
import os
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoProcessor

os.makedirs("embeddings", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("모델 로딩 중...")
model = AutoModel.from_pretrained("Bingsu/clip-vit-large-patch14-ko").to(device).eval()
processor = AutoProcessor.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
print("모델 로딩 완료")

CAMPUS_CATEGORIES = {"지갑", "전자기기", "휴대폰", "가방", "카드", "의류", "도서용품", "컴퓨터", "기타물품"}

def load_image_from_url(url):
    try:
        res = requests.get(url, timeout=5)
        img = Image.open(BytesIO(res.content)).convert("RGB")
        return img
    except:
        return None

def build_index():
    with open("data/items.json", "r", encoding="utf-8") as f:
        items = json.load(f)

    # 캠퍼스 카테고리 필터링
    items = [
        item for item in items
        if item["prdtClNm"].split(" > ")[0] in CAMPUS_CATEGORIES
    ]
    print(f"캠퍼스 카테고리 필터링 후: {len(items)}개")

    embeddings = []
    valid_items = []

    for i, item in enumerate(items):
        img = load_image_from_url(item["imgUrl"])
        if img is None:
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.vision_model(**inputs)
            emb = outputs.pooler_output
            emb = model.visual_projection(emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy()[0])
        valid_items.append(item)

        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(items)} 처리 중... (유효: {len(valid_items)}개)")

        time.sleep(0.1)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "embeddings/index.faiss")
    with open("embeddings/metadata.pkl", "wb") as f:
        pickle.dump(valid_items, f)

    print(f"\n완료! {len(valid_items)}개 인덱스 저장됨")

if __name__ == "__main__":
    build_index()