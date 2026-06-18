import faiss
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from peft import PeftModel
import os

MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"
LORA_DIR   = "./lora_weights/best"
device     = "cuda" if torch.cuda.is_available() else "cpu"

INDEX_PATHS = {
    "baseline": ("./embeddings/index.faiss",      "./embeddings/metadata.pkl"),
    "lora":     ("./embeddings_lora/index.faiss",  "./embeddings_lora/metadata.pkl"),
}


def load_model(mode: str = "lora"):
    print(f"모델 로딩 중... ({mode})")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    if mode == "lora":
        model.text_model = PeftModel.from_pretrained(model.text_model, LORA_DIR)
        model.text_model = model.text_model.merge_and_unload()
        proj_state = torch.load(f"{LORA_DIR}/text_projection.pt",
                                map_location=device, weights_only=True)
        model.text_projection.load_state_dict(proj_state)
        print("LoRA 가중치 로드 완료")

    print("모델 로딩 완료")
    return model, processor


def load_index(mode: str = "lora"):
    index_path, meta_path = INDEX_PATHS[mode]
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search(query: str, model, processor, index, metadata, top_k: int = 5):
    inputs = processor(text=[query], return_tensors="pt",
                       padding=True, truncation=True, max_length=77).to(device)
    with torch.inference_mode():
        out = model.text_model(**inputs)
        emb = model.text_projection(out.pooler_output)
        emb = F.normalize(emb, dim=-1)

    text_emb = emb.cpu().numpy().astype("float32")
    D, I     = index.search(text_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results