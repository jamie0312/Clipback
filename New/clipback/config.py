import torch

# ──────────────────────────────────────────
# Model / Path config
# ──────────────────────────────────────────
MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"
LORA_DIR = "lora_weights/best"

# LoRA로 검색할 때 사용할 FAISS index / metadata
INDEX_PATH = "embeddings_lora/index.faiss"
META_PATH = "embeddings_lora/metadata.pkl"

# 선택값: "lora" 고정 권장
SEARCH_MODE = "lora"
TOP_K = 100
DISPLAY_K = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAMPUS_CATEGORIES = [
    "전체", "지갑", "전자기기", "휴대폰", "가방", "의류", "도서용품", "기타물품"
]
