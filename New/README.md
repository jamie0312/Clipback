# LoRA Lost & Found Streamlit App



## 폴더 구조

```text
lora_lostfound_clean/
├─ app.py
├─ build_index_lora.py
├─ requirements.txt
├─ clipback/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ search_engine.py
│  └─ rerank.py
├─ lora_weights/
│  └─ best/
│     ├─ adapter_config.json
│     ├─ adapter_model.safetensors 또는 adapter_model.bin
│     └─ text_projection.pt
└─ embeddings_lora/
   ├─ index.faiss
   └─ metadata.pkl
```


## 검색 방식

1. LoRA가 적용된 한국어 CLIP으로 검색어 embedding 생성
2. FAISS에서 후보 100개 검색
3. 색상/카테고리/키워드 보정 점수 추가
4. 최종 점수 기준으로 재정렬
5. 상위 20개 표시
