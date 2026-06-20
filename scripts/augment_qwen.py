"""
Qwen2.5-VL-7B로 분실물 이미지 캡션 생성 (v2)
카테고리/색상/품명 정보를 프롬프트에 포함해 오인식 방지

사용법:
    python augment_qwen.py

출력:
    data/items_augmented_qwen.json
"""

import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH  = "data/items.json"
IMG_DIR    = "data/images"  # download_images.py로 이미지 다운로드 후 실행
SAVE_PATH  = "data/items_augmented_qwen.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Qwen2.5-VL-7B 로딩 중...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("모델 로딩 완료")


# ────────────────────────────────────────────────
# 프롬프트
# ────────────────────────────────────────────────
def make_prompt(item: dict) -> str:
    category = item.get("prdtClNm", "").replace(" > ", " ")
    color    = item.get("clrNm", "")
    name     = item.get("fdPrdtNm", "")

    return f"""이 이미지는 '{name}' 분실물 사진입니다.
카테고리: {category} / 색상: {color}

이미지에서 눈에 보이는 구체적인 특징을 한국어로 3문장 작성하세요.

[반드시 지킬 규칙]
- 각 문장은 반드시 구체적인 내용을 포함해야 합니다
- 괄호만 있는 형식 절대 금지 예) (색상), (재질) → 이런 형식 사용 금지
- 항목명만 쓰는 것 금지 예) "브랜드 로고", "패턴" → 구체적으로 어떤 로고인지 어떤 패턴인지 써야 함
- 이미지에 없는 내용 추측 금지
- 색상, 카테고리 반복 금지

[올바른 예시]
- "지갑 중앙에 금색 PRADA 로고가 새겨져 있음"
- "표면에 다이아몬드 퀼팅 패턴이 있음"
- "왼쪽 모서리에 긁힌 흔적이 있음"

[잘못된 예시]
- "(색상)" → 금지
- "브랜드 로고" → 금지
- "패턴" → 금지

형식:
1. (첫 번째 특징 - 구체적인 문장)
2. (두 번째 특징 - 구체적인 문장)
3. (세 번째 특징 - 구체적인 문장)"""


# ────────────────────────────────────────────────
# 캡션 생성
# ────────────────────────────────────────────────
def generate_captions(img: Image.Image, item: dict) -> list[str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": make_prompt(item)},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=200)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # 번호 파싱
    captions = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            caption = line.split(".", 1)[-1].strip()
            if caption:
                captions.append(caption)

    return captions if captions else [output.strip()]


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def augment():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 이미 처리된 것 이어서 하기
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            augmented = json.load(f)
        done_ids = {item["atcId"] for item in augmented}
        print(f"이미 처리된 항목: {len(done_ids)}개, 이어서 진행")
    else:
        augmented = []
        done_ids  = set()

    for item in tqdm(items, ncols=70):
        if item["atcId"] in done_ids:
            continue

        img_path = f"{IMG_DIR}/{item['atcId']}.jpg"
        if not os.path.exists(img_path):
            item["qwen_captions"] = []
            augmented.append(item)
            continue

        try:
            img      = Image.open(img_path).convert("RGB")
            captions = generate_captions(img, item)
        except Exception as e:
            print(f"\n에러 ({item['atcId']}): {e}")
            captions = []

        item["qwen_captions"] = captions
        augmented.append(item)

        # 50개마다 중간 저장
        if len(augmented) % 50 == 0:
            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(augmented, f, ensure_ascii=False, indent=2)
            print(f"\n중간 저장 ({len(augmented)}개)")

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print(f"\n완료! {len(augmented)}개 → {SAVE_PATH}")

    # 샘플 확인
    print("\n=== 샘플 ===")
    for item in augmented[:5]:
        if item.get("qwen_captions"):
            print(f"품명: {item.get('fdPrdtNm')} / {item.get('clrNm')} / {item.get('prdtClNm')}")
            for i, cap in enumerate(item["qwen_captions"], 1):
                print(f"  {i}. {cap}")
            print()


if __name__ == "__main__":
    augment()