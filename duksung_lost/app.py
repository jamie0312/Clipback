import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import requests
import streamlit as st
import pandas as pd
import torch

from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel


# ==========================================
# 1. 기본 스타일 및 테마 설정
# ==========================================
st.set_page_config(page_title="덕성여대 분실물센터", layout="wide")

st.markdown("""
    <style>
    .main-title {
        color: #8A1538;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #8A1538;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #6A102B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. 세션 상태 초기화
# ==========================================
if "selected_item" not in st.session_state:
    st.session_state.selected_item = None

if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False

if "current_query" not in st.session_state:
    st.session_state.current_query = ""


# ==========================================
# 3. 이미지 안전 로드 함수 (데이터 캐싱 처리)
# ==========================================
@st.cache_data(show_spinner=False)
def load_image_safely(img_path):
    img_path = str(img_path).strip()

    # URL 이미지
    if img_path.startswith("http://") or img_path.startswith("https://"):
        try:
            response = requests.get(img_path, timeout=1.5)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

    # 로컬 이미지
    elif os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        try:
            return Image.open(img_path).convert("RGB")
        except Exception:
            pass

    # 실패 시 더미 이미지
    return Image.new("RGB", (224, 224), color="#F2F2F2")


# ==========================================
# 4. 사전 계산 리소스 로드
# ==========================================
@st.cache_resource
def load_precomputed_resources():
    csv_file = "lost_and_found_dataset_precomputed.csv"
    pt_file = "image_features.pt"
    model_path = "./fine_tuned_campus_clip"

    if not os.path.exists(csv_file):
        st.error(f"오류: '{csv_file}' 파일을 찾을 수 없습니다.")
        return None, None, None, None, None

    if not os.path.exists(pt_file):
        st.error(f"오류: '{pt_file}' 파일을 찾을 수 없습니다.")
        return None, None, None, None, None

    if not os.path.exists(model_path):
        st.error(f"오류: '{model_path}' 폴더를 찾을 수 없습니다.")
        return None, None, None, None, None

    df = pd.read_csv(csv_file)

    # 상품명 컬럼 보정
    if "lstPrdNm" not in df.columns:
        if "description" in df.columns:
            df["lstPrdNm"] = df["description"]
        else:
            df["lstPrdNm"] = "분실물 아이템"

    # 이미지 경로 컬럼 보정
    if "img_file_path" not in df.columns:
        df["img_file_path"] = ""

    # 버튼 key 및 상세 페이지 연결용 row id
    df["_row_id"] = range(len(df))

    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP 모델 로드
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()

    # 사전 계산된 이미지 임베딩 로드
    try:
        all_image_features = torch.load(pt_file, map_location=device, weights_only=True)
    except TypeError:
        all_image_features = torch.load(pt_file, map_location=device)

    # CSV 행 개수와 임베딩 개수 검증
    if len(df) != all_image_features.shape[0]:
        st.error(
            f"CSV 행 개수({len(df)})와 이미지 임베딩 개수({all_image_features.shape[0]})가 일치하지 않습니다. "
            "임베딩 전처리 코드를 다시 실행하세요."
        )
        return None, None, None, None, None

    # 카테고리 자동 분류
    def assign_category(title):
        title = str(title)
        if "지갑" in title:
            return "지갑"
        elif "에어팟" in title or "폰" in title or "버즈" in title or "갤럭시" in title:
            return "전자기기"
        elif "가방" in title or "백팩" in title:
            return "가방"
        elif "목도리" in title or "의류" in title or "옷" in title:
            return "의류"
        else:
            return "기타"

    df["category"] = df["lstPrdNm"].apply(assign_category)

    # 코사인 유사도 계산을 위한 안전 정규화
    all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    return df, model, processor, all_image_features, device


df, model, processor, all_image_features, device = load_precomputed_resources()


# ==========================================
# 5. 화면 분기 1: 물품 상세 페이지
# ==========================================
if st.session_state.selected_item is not None:
    item = st.session_state.selected_item

    st.markdown(
        '<div class="main-title">덕성여대 분실물센터 - 물품 상세정보</div>',
        unsafe_allow_html=True
    )

    if st.button("⬅️ 검색 목록으로 돌아가기"):
        st.session_state.selected_item = None
        st.rerun()

    st.write("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        img = load_image_safely(item.get("img_file_path", ""))
        st.image(img, width="stretch")

    with col2:
        st.header(item.get("lstPrdNm", "분실물 아이템"))
        st.write(f"**카테고리:** {item.get('category', '기타')}")
        st.write("**보관 상태:** 학내 통합분실물센터 보관 중")

        score = item.get("score", "-")
        if score != "-":
            st.success(f"🎯 내 검색어와의 AI 유사도 점수: {score}점")

        st.write("---")
        st.subheader("🙋‍♀️ 주인 신청서 작성")

        with st.form(key="detail_claim_form"):
            student_id = st.text_input("학번", placeholder="예: 20240001")
            name = st.text_input("이름", placeholder="예: 덕성이")
            phone = st.text_input("연락처", placeholder="예: 010-1234-5678")

            submit_claim = st.form_submit_button("주인 확인 신청서 제출")

            if submit_claim:
                if student_id and name and phone:
                    st.success("신청서가 성공적으로 접수되었습니다. 담당자가 곧 연락드리겠습니다!")
                else:
                    st.warning("학번, 이름, 연락처를 빠짐없이 모두 입력해 주세요.")


# ==========================================
# 6. 화면 분기 2: 메인 검색 페이지
# ==========================================
else:
    st.markdown(
        '<div class="main-title">덕성여대 CLIP 분실물센터</div>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("검색 필터")

        category = st.selectbox(
            "물품 카테고리",
            ["전체", "지갑", "전자기기", "가방", "의류", "기타"]
        )

        min_score = st.slider(
            "AI 유사도 최소 커트라인 점수",
            0,
            100,
            15
        )

    if df is None or model is None or processor is None or all_image_features is None:
        st.info("핵심 백엔드 파일 리소스가 누락되어 서비스를 로드할 수 없습니다.")

    else:
        # ==========================================
        # 6-1. 검색 입력 폼 (검색어 고정 연동)
        # ==========================================
        with st.form("search_form"):
            col1, col2 = st.columns([4, 1])

            with col1:
                search_query = st.text_input(
                    "잃어버린 물건의 특징을 상세히 적어주세요",
                    value=st.session_state.current_query,
                    placeholder="예: 분홍색 키링이 달린 아이보리색 백팩 가방"
                )

            with col2:
                st.write("")
                st.write("")
                submitted = st.form_submit_button("AI 분실물 검색")

        if submitted and search_query.strip():
            st.session_state.search_clicked = True
            st.session_state.current_query = search_query.strip()
            st.rerun()

        elif submitted and not search_query.strip():
            st.session_state.search_clicked = False
            st.session_state.current_query = ""
            st.rerun()

        results_to_show = []
        is_initial_view = True

        # ==========================================
        # 6-2. AI 검색 실행 (행렬곱 연산)
        # ==========================================
        if st.session_state.search_clicked and st.session_state.current_query:
            is_initial_view = False

            with torch.inference_mode():
                inputs = processor(
                    text=[st.session_state.current_query],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 💡 [핵심 버그 수정 패치] 텍스트 반환값도 안전 상자 처리를 복구하여 순수 Tensor 추출
                text_outputs = model.get_text_features(**inputs)
                
                if isinstance(text_outputs, torch.Tensor):
                    text_features = text_outputs
                elif hasattr(text_outputs, 'text_embeds'):
                    text_features = text_outputs.text_embeds
                elif hasattr(text_outputs, 'pooler_output'):
                    text_features = text_outputs.pooler_output
                else:
                    text_features = text_outputs[0] if hasattr(text_outputs, '__getitem__') else text_outputs

                # 추출된 순수 Tensor를 정규화
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

                # 코사인 유사도 계산
                similarities = torch.matmul(text_features, all_image_features.T).squeeze(0)

                # 점수 범위 변환 및 음수 보정
                scores = (similarities * 100).clamp(0, 100).int().tolist()

            for idx, row in df.iterrows():
                score = scores[idx]

                if score >= min_score:
                    item = row.to_dict()
                    item["score"] = score

                    if category == "전체" or item["category"] == category:
                        results_to_show.append(item)

            results_to_show = sorted(
                results_to_show,
                key=lambda x: x["score"],
                reverse=True
            )

        # ==========================================
        # 6-3. 초기 화면 목록
        # ==========================================
        else:
            is_initial_view = True

            for _, row in df.iterrows():
                item = row.to_dict()
                item["score"] = "-"

                if category == "전체" or item["category"] == category:
                    results_to_show.append(item)

            results_to_show = results_to_show[:20]

        # ==========================================
        # 6-4. 결과 출력 그리드 시스템
        # ==========================================
        st.write("---")

        if not is_initial_view:
            st.write(f"### 🎯 AI 유사도순 검색 결과 ({len(results_to_show)}건)")
        else:
            st.write(f"### 📦 통합 분실물 습득 현황 (최근 {len(results_to_show)}개 목록 표시 중)")

        if not results_to_show:
            st.info("조건에 일치하는 분실물이 없습니다. 필터 점수를 낮추거나 검색어 키워드를 더 자세히 바꾸어 보세요.")

        else:
            cols = st.columns(3)

            for i, item in enumerate(results_to_show):
                with cols[i % 3]:
                    title = str(item.get("lstPrdNm", "분실물 아이템"))

                    if len(title) > 18:
                        st.subheader(title[:18] + "...")
                    else:
                        st.subheader(title)

                    img = load_image_safely(item.get("img_file_path", ""))
                    st.image(img, width="stretch")

                    st.write(f"**분류:** {item.get('category', '기타')}")

                    if item.get("score", "-") != "-":
                        st.success(f"🎯 AI 유사도 점수: {item['score']}점")
                    else:
                        st.info("📝 매칭 대기 중")

                    row_id = item.get("_row_id", i)

                    if st.button("🔍 상세보기 및 주인 신청", key=f"main_item_btn_{row_id}_{i}"):
                        st.session_state.selected_item = item
                        st.rerun()