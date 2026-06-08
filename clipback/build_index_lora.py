"""
LoRA fine-tuned 모델로 FAISS 인덱스 빌드

사용법:
    python build_index.py

출력:
    embeddings_lora/index.faiss
    embeddings_lora/metadata.pkl
"""

import json
import os
import numpy as np
import faiss
import pickle
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"
LORA_DIR   = "lora_weights/best"
DATA_PATH  = "data/items.json"
IMG_DIR    = "data/images"
SAVE_DIR   = "embeddings_lora"

CAMPUS_CATEGORIES = {"지갑", "전자기기", "휴대폰", "가방", "카드", "의류", "도서용품", "컴퓨터", "기타물품"}

os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ────────────────────────────────────────────────
# 모델 로드 (LoRA merged)
# ────────────────────────────────────────────────
print("모델 로딩 중...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

model.text_model = PeftModel.from_pretrained(model.text_model, LORA_DIR)
model.text_model = model.text_model.merge_and_unload()
proj_state = torch.load(f"{LORA_DIR}/text_projection.pt",
                        map_location=device, weights_only=True)
model.text_projection.load_state_dict(proj_state)
print("LoRA 가중치 로드 완료")


# ────────────────────────────────────────────────
# 인덱스 빌드
# ────────────────────────────────────────────────
def build_index():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 캠퍼스 카테고리 필터링
    items = [
        item for item in items
        if item["prdtClNm"].split(" > ")[0] in CAMPUS_CATEGORIES
    ]
    print(f"캠퍼스 카테고리 필터링 후: {len(items)}개")

    embeddings  = []
    valid_items = []

    for i, item in enumerate(items):
        img_path = f"{IMG_DIR}/{item['atcId']}.jpg"
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.vision_model(**inputs)
            emb = model.visual_projection(out.pooler_output)
            emb = F.normalize(emb, dim=-1)

        embeddings.append(emb.cpu().numpy()[0])
        valid_items.append(item)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)} 처리 중... (유효: {len(valid_items)}개)")

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{SAVE_DIR}/index.faiss")
    with open(f"{SAVE_DIR}/metadata.pkl", "wb") as f:
        pickle.dump(valid_items, f)

    print(f"\n완료! {len(valid_items)}개 → {SAVE_DIR}/")


if __name__ == "__main__":
    build_index()