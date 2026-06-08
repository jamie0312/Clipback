"""
LoRA fine-tuning for Bingsu/clip-vit-large-patch14-ko
경찰청 습득물 데이터(items.json)로 text encoder에 LoRA 적용

사용법:
    python train_lora.py

출력:
    lora_weights/  ← LoRA 가중치 저장
"""

import json
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import get_peft_model, LoraConfig

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
MODEL_NAME   = "Bingsu/clip-vit-large-patch14-ko"
DATA_PATH    = "data/items.json"
SAVE_DIR     = "lora_weights"
TRAIN_RATIO  = 0.8
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-4
TEMPERATURE  = 0.07
MAX_ITEMS    = None   # None이면 전체 사용, 디버그 시 100 등으로 제한
SEED         = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ────────────────────────────────────────────────
# 데이터셋
# ────────────────────────────────────────────────
def make_text(item: dict) -> str:
    """
    메타데이터 → 텍스트 설명 생성
    예) "갤럭시 버즈 화이트(흰)색 무선이어폰 전자기기"
    """
    parts = [
        item.get("fdPrdtNm", ""),
        item.get("clrNm", ""),
        item.get("prdtClNm", "").replace(" > ", " "),
    ]
    return " ".join(p for p in parts if p).strip()


class LostItemDataset(Dataset):
    def __init__(self, items: list, processor, img_dir: str = "data/images"):
        self.items     = items
        self.processor = processor
        self.img_dir   = img_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # 로컬 파일 읽기
        try:
            img_path = f"{self.img_dir}/{item['atcId']}.jpg"
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        text = make_text(item)

        img_inputs  = self.processor(images=img,   return_tensors="pt", padding=True)
        text_inputs = self.processor(text=[text],  return_tensors="pt", padding=True,
                                     truncation=True, max_length=77)

        return {
            "pixel_values": img_inputs["pixel_values"].squeeze(0),
            "input_ids":    text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    pixel_values  = torch.stack([b["pixel_values"]  for b in batch])
    input_ids     = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    return {
        "pixel_values":   pixel_values,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
    }


# ────────────────────────────────────────────────
# InfoNCE Loss (CLIP 원래 loss)
# ────────────────────────────────────────────────
def clip_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    image_emb, text_emb: (B, D) L2 정규화된 임베딩
    """
    logits = (image_emb @ text_emb.T) / temperature          # (B, B)
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = F.cross_entropy(logits,   labels)                # image → text
    loss_t = F.cross_entropy(logits.T, labels)                # text → image
    return (loss_i + loss_t) / 2


# ────────────────────────────────────────────────
# 임베딩 추출 헬퍼
# ────────────────────────────────────────────────
def get_image_emb(model, pixel_values):
    out = model.vision_model(pixel_values=pixel_values)
    emb = model.visual_projection(out.pooler_output)
    return F.normalize(emb, dim=-1)


def get_text_emb(model, input_ids, attention_mask):
    out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    emb = model.text_projection(out.pooler_output)
    return F.normalize(emb, dim=-1)


# ────────────────────────────────────────────────
# 학습
# ────────────────────────────────────────────────
def train():
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 데이터 로드 및 분할
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    if MAX_ITEMS:
        items = items[:MAX_ITEMS]

    random.shuffle(items)
    split = int(len(items) * TRAIN_RATIO)
    train_items = items[:split]
    val_items   = items[split:]
    print(f"Train: {len(train_items)}개 / Val: {len(val_items)}개")

    # 분할 인덱스 저장 (evaluate.py에서 val set 재사용)
    with open(f"{SAVE_DIR}/val_items.json", "w", encoding="utf-8") as f:
        json.dump(val_items, f, ensure_ascii=False, indent=2)
    print(f"Val set 저장 → {SAVE_DIR}/val_items.json")

    # 모델 & 프로세서 로드
    print("모델 로딩 중...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)

    # LoRA 설정
    # text_model이 BERT/RoBERTa 계열 → query, key, value 타겟
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # 실제 모델 weight 이름 확인 후 필요시 수정:
        # BERT계열: "query", "key", "value"
        # GPT/CLIP계열: "q_proj", "k_proj", "v_proj"
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    # text_model에만 LoRA 적용, vision_model은 freeze
    model.text_model = get_peft_model(model.text_model, lora_config)
    model.text_model.print_trainable_parameters()

    # vision_model 완전 freeze
    for param in model.vision_model.parameters():
        param.requires_grad = False

    model = model.to(device)

    # 옵티마이저: LoRA 파라미터 + text_projection만
    trainable_params = (
        list(model.text_model.parameters()) +
        list(model.text_projection.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)

    # 데이터로더
    train_dataset = LostItemDataset(train_items, processor)
    train_loader  = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=2
    )

    # 학습 루프
    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        start = time.time()

        for step, batch in enumerate(train_loader, 1):
            pixel_values  = batch["pixel_values"].to(device)
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            image_emb = get_image_emb(model, pixel_values)
            text_emb  = get_text_emb(model, input_ids, attention_mask)
            loss      = clip_loss(image_emb, text_emb, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"  Epoch {epoch} Step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - start
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}  ({elapsed:.0f}s)")

        # 체크포인트 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.text_model.save_pretrained(f"{SAVE_DIR}/best")
            torch.save(model.text_projection.state_dict(),
                       f"{SAVE_DIR}/best/text_projection.pt")
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")

    print(f"\n학습 완료! 가중치 → {SAVE_DIR}/best/")


if __name__ == "__main__":
    train()