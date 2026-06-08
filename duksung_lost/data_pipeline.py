import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 🔍 [우회책] 경찰청 API가 보낼 데이터 미리 가상으로 만들어두기
def get_mock_api_data():
    print("📡 [안내] API 승인 대기 중이므로 가상 경찰청 데이터를 로드합니다...")
    
    # 실제 경찰청 API 결과와 똑같은 구조로 샘플 5개를 만듭니다.
    mock_items = [
        {"atcId": "LOST_001", "lstPrdNm": "덕성여대 학생회관에서 주운 갈색 가죽 지갑", "fdFilePathImg": "https://images.unsplash.com/photo-1627123424574-724758594e93?w=400"},
        {"atcId": "LOST_002", "lstPrdNm": "라이언 캐릭터 키링이 달린 노란색 우산", "fdFilePathImg": "https://images.unsplash.com/photo-1544816155-12df9643f363?w=400"},
        {"atcId": "LOST_003", "lstPrdNm": "검은색 나이키 백팩 가방", "fdFilePathImg": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400"},
        {"atcId": "LOST_004", "lstPrdNm": "흰색 애플 에어팟 프로 2세대 본체", "fdFilePathImg": "https://images.unsplash.com/photo-1588444650733-d459987a67f2?w=400"},
        {"atcId": "LOST_005", "lstPrdNm": "분홍색 털실로 짠 목도리 방한용품", "fdFilePathImg": "https://images.unsplash.com/photo-1520638029751-6453dbcf9a6a?w=400"},
    ]
    return mock_items

# 2. 🤖 CLIP AI 모델 로드 및 벡터 저장 함수
def build_ai_index():
    # 가상 데이터 가져오기
    items = get_mock_api_data()
    df = pd.DataFrame(items)
    
    print("🤖 다국어 지원 CLIP 모델(Sentence-Transformer) 로드 중... (최초 1회 다운로드로 1~2분 소요)")
    # 한국어 처리가 잘 되는 경량화된 다국어 CLIP/텍스트 모델 사용
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    
    print("📝 분실물 명칭(텍스트) 분석 및 임베딩 생성 중...")
    # 분실물 이름을 AI가 이해하는 숫자 배열(벡터)로 변환
    descriptions = df['lstPrdNm'].tolist()
    embeddings = model.encode(descriptions)
    
    print("⚡ FAISS 고속 검색 인덱스 생성 중...")
    # FAISS 인덱스 만들기
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # 내적(Cosine 유사도 대용) 기반 검색
    
    # 데이터 정규화 후 FAISS에 저장
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # 💾 결과물 저장 (B님이 Streamlit에서 읽어갈 파일들)
    print("💾 AI 검색 파일 저장 중... (real_faiss.index, lost_data.csv)")
    faiss.write_index(index, "real_faiss.index")
    df.to_csv("lost_data.csv", index=False, encoding='utf-8-sig')
    print("🎉 [대성공] 가상 데이터 기반 검색 엔진 빌드 완료!")

if __name__ == "__main__":
    build_ai_index()