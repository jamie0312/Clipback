import torch

MODEL_NAME   = "Bingsu/clip-vit-large-patch14-ko"
LORA_DIR     = "lora_weights/best"
INDEX_PATH   = "embeddings/index.faiss"
META_PATH    = "embeddings/metadata.pkl"
SEARCH_MODE  = "lora"   # "baseline" | "lora" | "ensemble"
TOP_K        = 20
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CAMPUS_CATEGORIES = ["전체", "지갑", "전자기기", "휴대폰", "가방", "의류", "도서용품", "기타물품"]