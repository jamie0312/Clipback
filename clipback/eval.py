"""
Baseline vs LoRA 성능 평가
지표: R@1, R@5 (Image Retrieval Recall@k)

사용법:
    python evaluate.py --mode baseline   # 원본 모델
    python evaluate.py --mode lora       # LoRA fine-tuned 모델

출력 예시:
    [baseline] R@1: 0.312  R@5: 0.589
    [lora]     R@1: 0.421  R@5: 0.703
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"
LORA_DIR   = "lora_weights/best"
VAL_PATH   = "lora_weights/val_items.json"   # train_lora.py가 저장한 val set
device     = "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────────────────────────────
# 모델 로드
# ────────────────────────────────────────────────
def load_model(mode: str):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    if mode == "lora":
        model.text_model = PeftModel.from_pretrained(model.text_model, LORA_DIR)
        model.text_model = model.text_model.merge_and_unload()   # 추론 속도 최적화
        proj_state = torch.load(f"{LORA_DIR}/text_projection.pt", map_location=device)
        model.text_projection.load_state_dict(proj_state)
        print("LoRA 가중치 로드 완료")

    return model, processor


# ────────────────────────────────────────────────
# 임베딩 추출
# ────────────────────────────────────────────────
def embed_image(model, processor, atc_id: str, img_dir: str = "data/images") -> np.ndarray | None:
    try:
        img = Image.open(f"{img_dir}/{atc_id}.jpg").convert("RGB")
    except Exception:
        return None

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.vision_model(**inputs)
        emb = model.visual_projection(out.pooler_output)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()[0]


def embed_text(model, processor, text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt",
                       padding=True, truncation=True, max_length=77).to(device)
    with torch.inference_mode():
        out = model.text_model(**inputs)
        emb = model.text_projection(out.pooler_output)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()[0]


def make_text(item: dict) -> str:
    parts = [
        item.get("fdPrdtNm", ""),
        item.get("clrNm", ""),
        item.get("prdtClNm", "").replace(" > ", " "),
    ]
    return " ".join(p for p in parts if p).strip()


# ────────────────────────────────────────────────
# 평가
# ────────────────────────────────────────────────
def evaluate(mode: str):
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_items = json.load(f)
    print(f"Val set: {len(val_items)}개")

    model, processor = load_model(mode)

    # 1. val set 전체 이미지 임베딩 (검색 DB)
    print("이미지 임베딩 중...")
    img_embs   = []
    valid_idxs = []
    for i, item in enumerate(val_items):
        emb = embed_image(model, processor, item["atcId"])
        if emb is not None:
            img_embs.append(emb)
            valid_idxs.append(i)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(val_items)}")

    img_embs = np.array(img_embs)   # (N, D)
    print(f"유효 이미지: {len(valid_idxs)}개")

    # 2. 각 아이템의 텍스트로 쿼리 → 자기 이미지가 top-k에 있는지 확인
    print("텍스트 쿼리 평가 중...")
    r1_scores = []
    r5_scores = []

    for rank_i, orig_i in enumerate(valid_idxs):
        item  = val_items[orig_i]
        query = make_text(item)

        text_emb = embed_text(model, processor, query)                  # (D,)
        scores   = img_embs @ text_emb                                  # (N,)
        ranked   = np.argsort(-scores)                                  # 내림차순

        # ranked 안에서 자기 자신(rank_i)의 순위
        rank = int(np.where(ranked == rank_i)[0][0]) + 1               # 1-indexed

        r1_scores.append(1 if rank <= 1 else 0)
        r5_scores.append(1 if rank <= 5 else 0)

    r1 = np.mean(r1_scores)
    r5 = np.mean(r5_scores)

    print(f"\n{'='*40}")
    print(f"[{mode:8s}]  R@1 = {r1:.4f}   R@5 = {r5:.4f}")
    print(f"{'='*40}\n")

    # 결과 저장
    result = {"mode": mode, "R@1": round(r1, 4), "R@5": round(r5, 4),
              "n_queries": len(r1_scores)}
    with open(f"eval_{mode}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장 → eval_{mode}.json")

    return r1, r5


# ────────────────────────────────────────────────
# Failure case 분석
# ────────────────────────────────────────────────
def failure_analysis(mode: str, n_examples: int = 5):
    """
    R@5에서 실패한 케이스 출력 (보고서 Failure case 섹션용)
    """
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_items = json.load(f)

    model, processor = load_model(mode)

    img_embs   = []
    valid_idxs = []
    for i, item in enumerate(val_items):
        emb = embed_image(model, processor, item["atcId"])
        if emb is not None:
            img_embs.append(emb)
            valid_idxs.append(i)

    img_embs = np.array(img_embs)

    failures = []
    for rank_i, orig_i in enumerate(valid_idxs):
        item     = val_items[orig_i]
        query    = make_text(item)
        text_emb = embed_text(model, processor, query)
        scores   = img_embs @ text_emb
        ranked   = np.argsort(-scores)
        rank     = int(np.where(ranked == rank_i)[0][0]) + 1

        if rank > 5:
            top1_item = val_items[valid_idxs[ranked[0]]]
            failures.append({
                "query":        query,
                "true_rank":    rank,
                "top1_returned": make_text(top1_item),
                "category":     item.get("prdtClNm", ""),
                "top1_category": top1_item.get("prdtClNm", ""),
            })

    print(f"\n=== Failure Cases (rank > 5) ===")
    print(f"총 실패: {len(failures)}건\n")
    for f in failures[:n_examples]:
        print(f"쿼리:       {f['query']}")
        print(f"정답 순위:  {f['true_rank']}")
        print(f"Top-1 반환: {f['top1_returned']}")
        print(f"카테고리:   {f['category']} → {f['top1_category']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "lora", "both", "failure"],
                        default="both")
    parser.add_argument("--failure_mode", default="lora",
                        help="failure 분석에 사용할 모드")
    args = parser.parse_args()

    if args.mode == "both":
        r1_base, r5_base = evaluate("baseline")
        r1_lora, r5_lora = evaluate("lora")
        print(f"\n{'='*40}")
        print(f"성능 향상: R@1 {r1_base:.4f} → {r1_lora:.4f} "
              f"(+{r1_lora - r1_base:.4f})")
        print(f"          R@5 {r5_base:.4f} → {r5_lora:.4f} "
              f"(+{r5_lora - r5_base:.4f})")
    elif args.mode == "failure":
        failure_analysis(args.failure_mode)
    else:
        evaluate(args.mode)