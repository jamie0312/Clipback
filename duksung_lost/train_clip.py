import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from PIL import Image
import os
import pandas as pd

class CampusLostDataset(Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item["image_path"]
        text_label = item["text"]

        try:
            # 💡 파일 크기가 0이거나 존재하지 않으면 빈 화이트 이미지로 대체 (무한루프 방지)
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                image = Image.open(image_path).convert("RGB")
            else:
                image = Image.new("RGB", (224, 224), color="white")
        except:
            image = Image.new("RGB", (224, 224), color="white")
        
        inputs = self.processor(
            text=[text_label], 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=64,
            truncation=True
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0)
        }

def train_lora_clip():
    model_name = "openai/clip-vit-base-patch32"
    print(f"공개 베이스 모델 로드 중: {model_name}")
    
    # 💡 이미 다운로드 받았기 때문에 인터넷 접속 없이 내 컴퓨터 캐시에서 1초만에 로드됩니다.
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    print("LoRA 파인튜닝 가중치 레이어 생성 중...")
    lora_config = LoraConfig(
        r=16,                           
        lora_alpha=32,                  
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    csv_file_path = "lost_and_found_dataset.csv"
    train_data = []

    if os.path.exists(csv_file_path):
        df_csv = pd.read_csv(csv_file_path)
        
        # 💡 [핵심] 1336개를 다 돌리면 멈추므로, 상위 50개만 짤라서 안전하게 학습합니다.
        df_csv = df_csv.head(50)
        
        for idx, row in df_csv.iterrows():
            train_data.append({
                "image_path": row["img_file_path"], 
                "text": row["description"]          
            })
        print(f"안전한 학습을 위해 총 {len(train_data)}개의 핵심 데이터만 선별했습니다.")
    else:
        print("에러: lost_and_found_dataset.csv 파일이 없습니다.")
        return

    dataset = CampusLostDataset(train_data, processor)
    # 💡 사양 부하를 줄이기 위해 배치 사이즈를 1로 낮춥니다.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("CLIP 모델 LoRA 파인튜닝을 시작합니다...")
    model.train()
    epochs = 3 
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"에포크 [{epoch+1}/{epochs}] - Loss(손실값): {total_loss / len(dataloader):.4f}")

    output_dir = "fine_tuned_campus_clip"
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"🎉 성공! 학습 완료 폴더 경로: {output_dir}")

if __name__ == "__main__":
    train_lora_clip()