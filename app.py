import warnings
warnings.filterwarnings("ignore")

import requests
import streamlit as st
from PIL import Image
from io import BytesIO

from search_engine import load_model, load_index, search

SEARCH_MODE = "lora"   # "baseline" | "lora"
CAMPUS_CATEGORIES = ["전체", "지갑", "전자기기", "휴대폰", "가방", "의류", "도서용품", "기타물품"]

# ──────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────
st.set_page_config(page_title="덕성여대 분실물센터", layout="wide")

st.markdown("""
<style>
.main-title { color: #8A1538; font-size: 32px; font-weight: bold; margin-bottom: 20px; }
.stButton>button { background-color: #8A1538; color: white; border-radius: 8px; font-weight: bold; }
.stButton>button:hover { background-color: #6A102B; color: white; }
            
/* 카드 높이 고정 */
[data-testid="stVerticalBlockBorderWrapper"] {
    min-height: 380px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* 제목 높이 고정 */
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    height: 60px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# 세션 상태
# ──────────────────────────────────────────
if "selected_item"  not in st.session_state: st.session_state.selected_item  = None
if "search_clicked" not in st.session_state: st.session_state.search_clicked = False
if "current_query"  not in st.session_state: st.session_state.current_query  = ""


# ──────────────────────────────────────────
# 리소스 로드 (캐싱 → 최초 1회만 실행)
# ──────────────────────────────────────────
@st.cache_resource(show_spinner="모델 로딩 중...")
def get_resources():
    model, processor = load_model(SEARCH_MODE)
    index, metadata  = load_index(SEARCH_MODE)
    return model, processor, index, metadata


@st.cache_data(show_spinner=False)
def load_image(img_url: str):
    try:
        res = requests.get(img_url, timeout=3)
        if res.status_code == 200:
            return Image.open(BytesIO(res.content)).convert("RGB")
    except Exception:
        pass
    return Image.new("RGB", (224, 224), color="#F2F2F2")


model, processor, index, metadata = get_resources()


# ──────────────────────────────────────────
# 상세 페이지
# ──────────────────────────────────────────
if st.session_state.selected_item is not None:
    item = st.session_state.selected_item

    st.markdown('<div class="main-title">덕성여대 분실물센터 - 물품 상세정보</div>',
                unsafe_allow_html=True)

    if st.button("⬅️ 검색 목록으로 돌아가기"):
        st.session_state.selected_item = None
        st.rerun()

    st.write("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(load_image(item.get("imgUrl", "")), use_container_width=True)

    with col2:
        st.header(item.get("fdPrdtNm", "분실물 아이템"))
        st.write(f"**카테고리:** {item.get('prdtClNm', '-')}")
        st.write(f"**색상:** {item.get('clrNm', '-')}")
        st.write(f"**습득 일자:** {item.get('fdYmd', '-')}")
        st.write(f"**보관 장소:** {item.get('depPlace', '-')}")
        st.success(f"🎯 AI 유사도: {item.get('score', '-')}")

        st.write("---")
        st.subheader("🙋‍♀️ 주인 신청서 작성")

        student_id = st.text_input("학번", placeholder="예: 20240001")
        name       = st.text_input("이름", placeholder="예: 덕성이")
        phone      = st.text_input("연락처", placeholder="예: 010-1234-5678")

        if st.button("주인 확인 신청서 제출"):
            if student_id and name and phone:
                st.success("신청서가 접수되었습니다!")
            else:
                st.warning("학번, 이름, 연락처를 모두 입력해 주세요.")


# ──────────────────────────────────────────
# 메인 검색 페이지
# ──────────────────────────────────────────
else:
    st.markdown('<div class="main-title">덕성여대 CLIP 분실물센터</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("검색 필터")
        category = st.selectbox("물품 카테고리", CAMPUS_CATEGORIES)

    with st.form("search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input(
                "잃어버린 물건의 특징을 상세히 적어주세요",
                value=st.session_state.current_query,
                placeholder="예: 분홍색 키링이 달린 아이보리색 백팩"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("검색", use_container_width=True)

    if submitted:
        st.session_state.search_clicked = bool(search_query.strip())
        st.session_state.current_query  = search_query.strip()
        st.rerun()

    # 검색 실행
    results    = []
    is_initial = True

    if st.session_state.search_clicked and st.session_state.current_query:
        is_initial = False
        raw = search(st.session_state.current_query, model, processor,
                     index, metadata, top_k=9)

        # 카테고리 필터
        results = [
            item for item in raw
            if category == "전체" or item.get("prdtClNm", "").split(" > ")[0] == category
        ]
    else:
        filtered = [
            item for item in metadata
            if category == "전체" or item.get("prdtClNm", "").split(" > ")[0] == category
        ]
        results = filtered[:9]

    # 결과 그리드
    st.write("---")
    if is_initial:
        st.write(f"### 📦 습득물 현황 (최근 {len(results)}개)")
    else:
        st.write(f"### 🎯 AI 검색 결과 ({len(results)}건)")

    if not results:
        st.info("조건에 맞는 분실물이 없습니다. 카테고리를 바꾸거나 검색어를 수정해 보세요.")
    else:
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                with st.container(border=True):
                    title = item.get("fdPrdtNm", "분실물")
                    # st.subheader(title[:12] + ("..." if len(title) > 18 else ""))
                    st.markdown(
                        f'<div style="height:60px; overflow:hidden; font-size:18px; font-weight:bold;">{title}</div>',
                        unsafe_allow_html=True
                    )
                    st.image(load_image(item.get("imgUrl", "")), use_container_width=True)
                    st.write(f"**분류:** {item.get('prdtClNm', '-')}")
                    st.write(f"**색상:** {item.get('clrNm', '-')}")

                    score = item.get("score")
                    if score is not None:
                        st.success(f"🎯 유사도: {score:.3f}")

                    if st.button("🔍 상세보기", key=f"btn_{i}_{item.get('atcId', i)}"):
                        st.session_state.selected_item = item
                        st.rerun()