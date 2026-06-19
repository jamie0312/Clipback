import os
import pickle
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoProcessor

from clipback.config import DEVICE, INDEX_PATH, LORA_DIR, META_PATH, MODEL_NAME


def load_model(mode: str = "lora"):
    """
    mode: "baseline" | "lora"

    return:
        model, processor
    """
    print(f"모델 로딩 중... mode={mode}, device={DEVICE}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    if mode == "lora":
        if not os.path.isdir(LORA_DIR):
            raise FileNotFoundError(
                f"LoRA 폴더를 찾을 수 없습니다: {LORA_DIR}\n"
                "프로젝트 루트에 lora_weights/best 폴더를 넣어주세요."
            )

        model.text_model = PeftModel.from_pretrained(model.text_model, LORA_DIR)
        model.text_model = model.text_model.merge_and_unload()

        projection_path = os.path.join(LORA_DIR, "text_projection.pt")
        if os.path.exists(projection_path):
            proj_state = torch.load(
                projection_path,
                map_location=DEVICE,
                weights_only=True,
            )
            model.text_projection.load_state_dict(proj_state)
            print("LoRA text_projection.pt 로드 완료")
        else:
            print("경고: text_projection.pt가 없습니다. text_model LoRA만 적용합니다.")

        print("LoRA 가중치 로드 완료")

    print("모델 로딩 완료")
    return model, processor


def load_index(mode: str = "lora"):
    """
    mode는 호환성을 위해 받지만, 이 clean 버전은 config.py의 INDEX_PATH/META_PATH를 사용합니다.
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index를 찾을 수 없습니다: {INDEX_PATH}\n"
            "먼저 embeddings_lora/index.faiss 파일을 넣거나 build_index_lora.py를 실행하세요."
        )
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"metadata 파일을 찾을 수 없습니다: {META_PATH}\n"
            "먼저 embeddings_lora/metadata.pkl 파일을 넣거나 build_index_lora.py를 실행하세요."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"Index 로드 완료: {INDEX_PATH}")
    print(f"Metadata 로드 완료: {META_PATH} ({len(metadata)}개)")
    print(f"FAISS metric_type: {index.metric_type}  |  ntotal: {index.ntotal}")
    return index, metadata


def get_text_embedding(model, processor, query: str) -> np.ndarray:
    """검색어를 L2 정규화된 float32 numpy vector로 변환"""
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(DEVICE)

    with torch.inference_mode():
        out = model.text_model(**inputs)
        emb = model.text_projection(out.pooler_output)
        emb = F.normalize(emb, dim=-1)

    return emb.cpu().numpy().astype("float32")


def _faiss_score_to_similarity(raw_score: float, index) -> float:
    """
    IndexFlatIP이면 raw_score가 cosine similarity에 가까움.
    IndexFlatL2이면 raw_score는 squared L2 distance라서 similarity로 변환.
    """
    if index.metric_type == faiss.METRIC_L2:
        # normalized vector 기준: ||x-y||^2 = 2 - 2cos
        return 1.0 - float(raw_score) / 2.0
    return float(raw_score)


def search(
    query: str,
    model,
    processor,
    index,
    metadata: List[Dict[str, Any]],
    top_k: int = 100,
) -> List[Dict[str, Any]]:
    """LoRA-CLIP text embedding으로 FAISS 검색"""
    text_emb = get_text_embedding(model, processor, query)
    D, I = index.search(text_emb, top_k)

    results = []
    for raw_score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        item["lora_score"] = _faiss_score_to_similarity(float(raw_score), index)
        item["raw_faiss_score"] = float(raw_score)
        item["score"] = item["lora_score"]
        results.append(item)

    return results
