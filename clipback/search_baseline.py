import faiss
import pickle
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

print("모델 로딩 중...")
model = AutoModel.from_pretrained("Bingsu/clip-vit-large-patch14-ko").to(device).eval()
processor = AutoProcessor.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
print("모델 로딩 완료")

index = faiss.read_index("embeddings/index.faiss")
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search(query, top_k=5):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        outputs = model.text_model(**inputs)
        emb = outputs.pooler_output
        emb = model.text_projection(emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    text_emb = emb.cpu().numpy().astype("float32")
    D, I = index.search(text_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results

# def search_by_image(img, top_k=5):
#     inputs = processor(images=img, return_tensors="pt").to(device)
#     with torch.inference_mode():
#         outputs = model.vision_model(**inputs)
#         emb = outputs.pooler_output
#         emb = model.visual_projection(emb)
#         emb = emb / emb.norm(dim=-1, keepdim=True)

#     img_emb = emb.cpu().numpy().astype("float32")
#     D, I = index.search(img_emb, top_k)
#     return results

if __name__ == "__main__":
    # query = "흰색 무선 이어폰"
    query = input("검색어를 입력하세요: ")
    print(f"검색어: {query}\n")
    results = search(query, top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['fdPrdtNm']} / {r['clrNm']} / {r['prdtClNm']}")
        print(f"       {r['imgUrl']}\n")