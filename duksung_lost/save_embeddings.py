import os
import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

print(" [PRECOMPUTE] AI 모델 및 CSV 데이터셋 로드 중...")

csv_file = "lost_and_found_dataset.csv"
model_path = "./fine_tuned_campus_clip"

# CSV 로드
df = pd.read_csv(csv_file)

# GPU(CUDA)가 사용 가능하면 GPU를, 없으면 CPU를 자동으로 할당
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"현재 연산에 사용할 디바이스: [{device.upper()}]")

# 모델을 설정된 디바이스로 이관 및 평가 모드로 전환
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)
model.eval()

# URL 및 로컬 이미지 안전 로드 함수
def load_image_safely(img_path):
    img_path = str(img_path).strip()

    if img_path.startswith("http://") or img_path.startswith("https://"):
        try:
            response = requests.get(img_path, timeout=3)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

    elif os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        try:
            return Image.open(img_path).convert("RGB")
        except Exception:
            pass

    # 이미지 로드 실패 시 대체용 회색 공백 더미 이미지 반환
    return Image.new("RGB", (224, 224), color="#F2F2F2")

print(f"총 {len(df)}개의 물품에 대한 이미지 임ベ딩 추출을 시작합니다.")

image_features_list = []

# 대용량 처리를 위한 초고속 루프 시작
for idx, row in df.iterrows():
    image = load_image_safely(row.get("img_file_path", ""))

    # no_grad보다 성능이 더 좋고 메모리가 절약되는 inference_mode 켜기
    with torch.inference_mode():
        img_inputs = processor(images=image, return_tensors="pt")
        # 이미지 전처리 데이터를 GPU(또는 CPU)로 전송
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

        # 모델 아웃풋 추출
        image_outputs = model.get_image_features(**img_inputs)
        
        # [핵심 안전장치] BaseModelOutputWithPooling 객체 반환 에러 우회 및 순수 텐서 변환
        if isinstance(image_outputs, torch.Tensor):
            img_feat = image_outputs
        elif hasattr(image_outputs, 'image_embeds'):
            img_feat = image_outputs.image_embeds
        elif hasattr(image_outputs, 'pooler_output'):
            img_feat = image_outputs.pooler_output
        else:
            img_feat = image_outputs[0] if hasattr(image_outputs, '__getitem__') else image_outputs

        # 유사도 비교 행렬곱을 위한 단위 벡터 정규화(L2 Norm)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # [VRAM 폭발 방지] 계산된 벡터를 CPU 메모리로 옮겨서 누적 (Out Of Memory 에러 차단)
        image_features_list.append(img_feat.cpu())

    # 50개 단위로 터미널에 진행 상황 출력
    if (idx + 1) % 50 == 0:
        print(f" 진행 상황: {idx + 1}/{len(df)}개 완료")

# 개별 벡터 리스트들을 하나의 거대한 행렬 텐서로 결합
all_image_features = torch.cat(image_features_list, dim=0)

# 하드디스크에 바이너리 및 백업용 CSV 파일 저장
torch.save(all_image_features, "image_features.pt")
df.to_csv("lost_and_found_dataset_precomputed.csv", index=False)

print(f"\n [SUCCESS] 이미지 임베딩 저장 완료: image_features.pt")
print(f"CSV 데이터셋 동기화 완료: lost_and_found_dataset_precomputed.csv")
print(f"최종 저장된 임베딩 Tensor 구조(Shape): {all_image_features.shape}")