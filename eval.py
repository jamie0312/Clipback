"""
Baseline vs LoRA vs LoRA-Aug 성능 평가
지표: R@1, R@5 (Image Retrieval Recall@k)

사용법:
    python eval.py --mode baseline
    python eval.py --mode lora
    python eval.py --mode lora_aug
    python eval.py --mode all        # 세 개 전부 비교
    python eval.py --mode failure --failure_mode lora
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
device     = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = {
    "baseline": {
        "lora_dir":  None,
        "val_path":  "lora_weights/val_items.json",
    },
    "lora": {
        "lora_dir":  "lora_weights/best",
        "val_path":  "lora_weights/val_items.json",
    },
    "lora_aug": {
        "lora_dir":  "lora_weights_aug/best",
        "val_path":  "lora_weights_aug/val_items.json",
    },
    "lora_aug2": {                              
        "lora_dir":  "lora_weights_aug2/best",
        "val_path":  "lora_weights_aug2/val_items.json",
    },
}


# ────────────────────────────────────────────────
# 모델 로드
# ────────────────────────────────────────────────
def load_model(mode: str):
    lora_dir = CONFIGS[mode]["lora_dir"]

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    if lora_dir:
        model.text_model = PeftModel.from_pretrained(model.text_model, lora_dir)
        model.text_model = model.text_model.merge_and_unload()
        proj_state = torch.load(
            f"{lora_dir}/text_projection.pt",
            map_location=device,
            weights_only=True,
        )
        model.text_projection.load_state_dict(proj_state)
        print(f"LoRA 가중치 로드 완료 ({lora_dir})")

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
        emb = F.normalize(model.visual_projection(out.pooler_output), dim=-1)
    return emb.cpu().numpy()[0]


def embed_text(model, processor, text: str) -> np.ndarray:
    inputs = processor(
        text=[text], return_tensors="pt",
        padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.inference_mode():
        out = model.text_model(**inputs)
        emb = F.normalize(model.text_projection(out.pooler_output), dim=-1)
    return emb.cpu().numpy()[0]


def make_text(item: dict) -> str:
    return " ".join(filter(None, [
        item.get("fdPrdtNm", ""),
        item.get("clrNm", ""),
        item.get("prdtClNm", "").replace(" > ", " "),
    ])).strip()


# ────────────────────────────────────────────────
# 평가
# ────────────────────────────────────────────────
def evaluate(mode: str):
    val_path = CONFIGS[mode]["val_path"]

    with open(val_path, "r", encoding="utf-8") as f:
        val_items = json.load(f)
    print(f"\nVal set: {len(val_items)}개 ({val_path})")

    model, processor = load_model(mode)

    print("이미지 임베딩 중...")
    img_embs, valid_idxs = [], []
    for i, item in enumerate(val_items):
        emb = embed_image(model, processor, item["atcId"])
        if emb is not None:
            img_embs.append(emb)
            valid_idxs.append(i)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(val_items)}")

    img_embs = np.array(img_embs)
    print(f"유효 이미지: {len(valid_idxs)}개")

    print("텍스트 쿼리 평가 중...")
    r1_scores, r5_scores = [], []
    for rank_i, orig_i in enumerate(valid_idxs):
        item     = val_items[orig_i]
        query    = make_text(item)
        text_emb = embed_text(model, processor, query)
        scores   = img_embs @ text_emb
        ranked   = np.argsort(-scores)
        rank     = int(np.where(ranked == rank_i)[0][0]) + 1

        r1_scores.append(1 if rank <= 1 else 0)
        r5_scores.append(1 if rank <= 5 else 0)

    r1 = np.mean(r1_scores)
    r5 = np.mean(r5_scores)

    print(f"\n{'='*40}")
    print(f"[{mode:10s}]  R@1 = {r1:.4f}   R@5 = {r5:.4f}")
    print(f"{'='*40}")

    result = {"mode": mode, "R@1": round(r1, 4), "R@5": round(r5, 4), "n_queries": len(r1_scores)}
    with open(f"eval_{mode}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장 → eval_{mode}.json")

    return r1, r5


# ────────────────────────────────────────────────
# Failure case 분석
# ────────────────────────────────────────────────
def failure_analysis(mode: str, n_examples: int = 5):
    val_path = CONFIGS[mode]["val_path"]
    with open(val_path, "r", encoding="utf-8") as f:
        val_items = json.load(f)

    model, processor = load_model(mode)

    img_embs, valid_idxs = [], []
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
                "query":         query,
                "true_rank":     rank,
                "top1_returned": make_text(top1_item),
                "category":      item.get("prdtClNm", ""),
                "top1_category": top1_item.get("prdtClNm", ""),
            })

    print(f"\n=== Failure Cases (rank > 5) / {mode} ===")
    print(f"총 실패: {len(failures)}건\n")
    for f in failures[:n_examples]:
        print(f"쿼리:       {f['query']}")
        print(f"정답 순위:  {f['true_rank']}")
        print(f"Top-1 반환: {f['top1_returned']}")
        print(f"카테고리:   {f['category']} → {f['top1_category']}")
        print()


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["baseline", "lora", "lora_aug", "lora_aug2", "all", "failure"],
                        default="all")
    parser.add_argument("--failure_mode",
                        choices=["baseline", "lora", "lora_aug", "lora_aug2"],
                        default="lora_aug")
    args = parser.parse_args()

    if args.mode == "all":
        r1_base, r5_base = evaluate("baseline")
        r1_lora, r5_lora = evaluate("lora")
        r1_aug,  r5_aug  = evaluate("lora_aug")
        r1_aug2, r5_aug2 = evaluate("lora_aug2")

        print(f"\n{'='*50}")
        print(f"{'모델':<12} {'R@1':>8} {'R@5':>8}")
        print(f"{'-'*50}")
        print(f"{'Baseline':<12} {r1_base:>8.4f} {r5_base:>8.4f}")
        print(f"{'LoRA':<12} {r1_lora:>8.4f} {r5_lora:>8.4f}")
        print(f"{'LoRA+Aug':<12} {r1_aug:>8.4f} {r5_aug:>8.4f}")
        print(f"{'LoRA+Aug2':<12} {r1_aug2:>8.4f} {r5_aug2:>8.4f}")
        print(f"{'='*50}")

    elif args.mode == "failure":
        failure_analysis(args.failure_mode)
    else:
        evaluate(args.mode)