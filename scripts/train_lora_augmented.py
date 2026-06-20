"""
Qwen2.5-VL 증강 캡션 데이터로 LoRA fine-tuning

사용법:
    python train_lora_augmented.py

출력:
    lora_weights_aug/best/
"""

import json
import os
import random
import time

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
MODEL_NAME  = "Bingsu/clip-vit-large-patch14-ko"
# DATA_PATH   = "data/items_augmented_qwen.json"
DATA_PATH = "data/items_augmented_qwen2.json"
IMG_DIR     = "data/images"
# SAVE_DIR    = "lora_weights_aug"
SAVE_DIR    = "lora_weights_aug2"
TRAIN_RATIO = 0.8
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 2e-4
TEMPERATURE = 0.07
SEED        = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
os.makedirs(SAVE_DIR, exist_ok=True)


# ────────────────────────────────────────────────
# 텍스트 생성
# ────────────────────────────────────────────────
# def make_text(item: dict) -> str:
#     """원본 메타데이터 + Qwen 첫 번째 캡션 결합"""
#     original = " ".join(filter(None, [
#         item.get("fdPrdtNm", ""),
#         item.get("clrNm", ""),
#         item.get("prdtClNm", "").replace(" > ", " "),
#     ]))
#     captions = item.get("qwen_captions", [])
#     caption  = captions[0] if captions else ""
#     return f"{original} {caption}".strip() if caption else original.strip()

def make_text(item: dict, caption: str = "") -> str:
    base = " ".join(filter(None, [
        item.get("fdPrdtNm", ""),
        item.get("clrNm", ""),
        item.get("prdtClNm", "").replace(" > ", " "),
    ]))
    if caption:
        # 캡션에서 핵심어만 (따옴표·문장 부호 제거)
        cap = caption.strip().strip('"').strip("'")
        # 문장형이면 40자로 자르기
        cap = cap[:40]
        return f"{base} {cap}".strip()
    return base


# ────────────────────────────────────────────────
# 데이터셋
# ────────────────────────────────────────────────
class LostItemDataset(Dataset):
    def __init__(self, items: list, processor):
        # 캡션마다 별도 쌍으로 펼치기
        self.pairs = []
        for item in items:
            captions = item.get("qwen_captions", [])
            # 기본 쌍 (캡션 없이 메타데이터만)
            self.pairs.append((item, ""))
            # 캡션별 추가 쌍
            for cap in captions:
                self.pairs.append((item, cap))
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item, caption = self.pairs[idx]
        try:
            img = Image.open(f"{IMG_DIR}/{item['atcId']}.jpg").convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        text        = make_text(item, caption)
        img_inputs  = self.processor(images=img, return_tensors="pt", padding=True)
        text_inputs = self.processor(
            text=[text], return_tensors="pt",
            padding=True, truncation=True, max_length=77
        )
        return {
            "pixel_values":   img_inputs["pixel_values"].squeeze(0),
            "input_ids":      text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=0
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        ),
    }


# ────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────
def clip_loss(img_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (img_emb @ text_emb.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ────────────────────────────────────────────────
# 학습
# ────────────────────────────────────────────────
def train():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    items = [item for item in items if item.get("qwen_captions")]
    print(f"Qwen 캡션 있는 항목: {len(items)}개")

    random.shuffle(items)
    split       = int(len(items) * TRAIN_RATIO)
    train_items = items[:split]
    val_items   = items[split:]
    print(f"Train: {len(train_items)}개 / Val: {len(val_items)}개")

    with open(f"{SAVE_DIR}/val_items.json", "w", encoding="utf-8") as f:
        json.dump(val_items, f, ensure_ascii=False, indent=2)

    print("모델 로딩 중...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1, bias="none",
    )
    model.text_model = get_peft_model(model.text_model, lora_config)
    model.text_model.print_trainable_parameters()

    for param in model.vision_model.parameters():
        param.requires_grad = False

    model = model.to(device)

    trainable_params = (
        list(model.text_model.parameters()) +
        list(model.text_projection.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)

    train_loader = DataLoader(
        LostItemDataset(train_items, processor),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )

    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        start      = time.time()

        for step, batch in enumerate(train_loader, 1):
            pv   = batch["pixel_values"].to(device)
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            img_emb  = F.normalize(
                model.visual_projection(model.vision_model(pixel_values=pv).pooler_output),
                dim=-1,
            )
            text_emb = F.normalize(
                model.text_projection(
                    model.text_model(input_ids=ids, attention_mask=mask).pooler_output
                ),
                dim=-1,
            )

            loss = clip_loss(img_emb, text_emb, TEMPERATURE)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"  Epoch {epoch} Step {step}/{len(train_loader)} loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f} ({time.time() - start:.0f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.text_model.save_pretrained(f"{SAVE_DIR}/best")
            torch.save(
                model.text_projection.state_dict(),
                f"{SAVE_DIR}/best/text_projection.pt",
            )
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")

    print(f"\n학습 완료! → {SAVE_DIR}/best/")


if __name__ == "__main__":
    train()