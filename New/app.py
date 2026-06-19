import html
import warnings
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

from clipback.config import CAMPUS_CATEGORIES, DISPLAY_K, SEARCH_MODE, TOP_K
from clipback.rerank import rerank_results
from clipback.search_engine import load_index, load_model, search

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# Page config
# ──────────────────────────────────────────
st.set_page_config(page_title="덕성여대 분실물센터", layout="wide")

st.markdown(
    """
<style>
.main-title {
    color: #8A1538;
    font-size: 32px;
    font-weight: bold;app
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
[data-testid="stVerticalBlockBorderWrapper"] {
    min-height: 410px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────
# Session state
# ──────────────────────────────────────────
if "selected_item" not in st.session_state:
    st.session_state.selected_item = None
if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "show_debug" not in st.session_state:
    st.session_state.show_debug = True

# ──────────────────────────────────────────
# Resource loading
# ──────────────────────────────────────────
@st.cache_resource(show_spinner="LoRA 모델과 FAISS 인덱스 로딩 중...")
def get_resources():
    model, processor = load_model(SEARCH_MODE)
    index, metadata = load_index(SEARCH_MODE)
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


try:
    model, processor, index, metadata = get_resources()
except Exception as e:
    st.error("리소스 로딩 실패")
    st.exception(e)
    st.stop()


def match_category(item: dict, category: str) -> bool:
    if category == "전체":
        return True
    return str(item.get("prdtClNm", "")).split(" > ")[0].strip() == category


# ──────────────────────────────────────────
# Detail page
# ──────────────────────────────────────────
if st.session_state.selected_item is not None:
    item = st.session_state.selected_item

    st.markdown(
        '<div class="main-title">덕성여대 분실물센터 - 물품 상세정보</div>',
        unsafe_allow_html=True,
    )

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
        st.success(f"🎯 최종 유사도: {float(item.get('score', 0)):.3f}")

        if "lora_score" in item:
            st.caption(
                f"LoRA: {item.get('lora_score', 0):.3f} / "
                f"분류: {item.get('category_score', 0):.1f} / "
                f"색상: {item.get('color_score', 0):.1f} / "
                f"키워드: {item.get('keyword_score', 0):.2f}"
            )

        st.write("---")
        st.subheader("🙋‍♀️ 주인 신청서 작성")

        student_id = st.text_input("학번", placeholder="예: 20240001")
        name = st.text_input("이름", placeholder="예: 덕성이")
        phone = st.text_input("연락처", placeholder="예: 010-1234-5678")

        if st.button("주인 확인 신청서 제출"):
            if student_id and name and phone:
                st.success("신청서가 접수되었습니다!")
            else:
                st.warning("학번, 이름, 연락처를 모두 입력해 주세요.")

# ──────────────────────────────────────────
# Main search page
# ──────────────────────────────────────────
else:
    st.markdown(
        '<div class="main-title">덕성여대 LoRA 분실물센터</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("검색 필터")
        category = st.selectbox("물품 카테고리", CAMPUS_CATEGORIES)
        st.session_state.show_debug = st.checkbox("디버그 점수 표시", value=st.session_state.show_debug)
        st.caption("LoRA 검색 결과에 색상/분류/키워드 보정 점수를 더해 재정렬합니다.")

    with st.form("search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input(
                "잃어버린 물건의 특징을 상세히 적어주세요",
                value=st.session_state.current_query,
                placeholder="예: 분홍색 키링이 달린 아이보리색 백팩",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("검색", use_container_width=True)

    if submitted:
        st.session_state.search_clicked = bool(search_query.strip())
        st.session_state.current_query = search_query.strip()
        st.rerun()

    results = []
    is_initial = True

    if st.session_state.search_clicked and st.session_state.current_query:
        is_initial = False

        # 1) LoRA로 후보를 넉넉히 검색
        raw = search(
            st.session_state.current_query,
            model,
            processor,
            index,
            metadata,
            top_k=TOP_K,
        )

        # 2) 색상/카테고리/키워드 보정으로 재정렬
        raw = rerank_results(st.session_state.current_query, raw)

        # 3) 선택 카테고리 필터
        results = [item for item in raw if match_category(item, category)]
        results = results[:DISPLAY_K]

    else:
        # 초기 화면: metadata 앞쪽 DISPLAY_K개 표시
        filtered = [item for item in metadata if match_category(item, category)]
        results = filtered[:DISPLAY_K]

    st.write("---")
    if is_initial:
        st.write(f"### 📦 습득물 현황 (최근 {len(results)}개)")
    else:
        st.write(f"### 🎯 AI 검색 결과 ({len(results)}건)")
        st.caption(f"검색어: {st.session_state.current_query}")

    if not results:
        st.info("조건에 맞는 분실물이 없습니다. 카테고리를 바꾸거나 검색어를 수정해 보세요.")
    else:
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                with st.container(border=True):
                    title = html.escape(str(item.get("fdPrdtNm", "분실물")))
                    st.markdown(
                        f'<div style="height:60px; overflow:hidden; font-size:18px; font-weight:bold;">{title}</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(load_image(item.get("imgUrl", "")), use_container_width=True)
                    st.write(f"**분류:** {item.get('prdtClNm', '-')}")
                    st.write(f"**색상:** {item.get('clrNm', '-')}")

                    score = item.get("score")
                    if score is not None:
                        st.success(f"🎯 유사도: {float(score):.3f}")

                    if st.session_state.show_debug and "lora_score" in item:
                        st.caption(
                            f"LoRA {item.get('lora_score', 0):.3f} | "
                            f"분류 {item.get('category_score', 0):.1f} | "
                            f"색상 {item.get('color_score', 0):.1f} | "
                            f"키워드 {item.get('keyword_score', 0):.2f}"
                        )

                    button_key = f"btn_{i}_{item.get('atcId', i)}"
                    if st.button("🔍 상세보기", key=button_key):
                        st.session_state.selected_item = item
                        st.rerun()
