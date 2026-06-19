from typing import Dict, List, Optional


COLOR_KEYWORDS: Dict[str, List[str]] = {
    "검정": ["검정", "검은", "검은색", "블랙", "black"],
    "흰색": ["흰색", "하얀", "하양", "화이트", "white"],
    "빨강": ["빨강", "빨간", "레드", "red"],
    "파랑": ["파랑", "파란", "블루", "blue", "남색", "네이비", "navy"],
    "초록": ["초록", "녹색", "그린", "green"],
    "노랑": ["노랑", "노란", "옐로", "yellow"],
    "분홍": ["분홍", "핑크", "pink"],
    "보라": ["보라", "퍼플", "purple"],
    "갈색": ["갈색", "브라운", "brown"],
    "회색": ["회색", "그레이", "gray", "grey"],
    "베이지": ["베이지", "beige"],
    "아이보리": ["아이보리", "ivory"],
    "은색": ["은색", "실버", "silver"],
    "금색": ["금색", "골드", "gold"],
}

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "지갑": ["지갑", "카드지갑", "반지갑", "장지갑", "카드케이스", "카드 케이스"],
    "전자기기": ["에어팟", "버즈", "이어폰", "헤드폰", "충전기", "마우스", "태블릿", "아이패드", "노트북", "전자기기"],
    "휴대폰": ["핸드폰", "휴대폰", "스마트폰", "폰", "아이폰", "갤럭시"],
    "가방": ["가방", "백팩", "배낭", "에코백", "파우치", "크로스백", "숄더백", "토트백"],
    "의류": ["옷", "의류", "상의", "하의", "자켓", "재킷", "후드", "후드티", "모자", "목도리", "장갑"],
    "도서용품": ["책", "교재", "노트", "공책", "필통", "펜", "파일", "프린트", "도서"],
    "기타물품": ["키링", "열쇠", "우산", "텀블러", "인형", "화장품", "안경", "시계"],
}


def _norm(text: str) -> str:
    return str(text or "").lower().replace(" ", "")


def contains_any(text: str, words: List[str]) -> bool:
    t = _norm(text)
    return any(_norm(w) in t for w in words)


def find_color(text: str) -> Optional[str]:
    for color, words in COLOR_KEYWORDS.items():
        if contains_any(text, words):
            return color
    return None


def find_category(text: str) -> Optional[str]:
    for category, words in CATEGORY_KEYWORDS.items():
        if contains_any(text, words):
            return category
    return None


def get_main_category(item: dict) -> str:
    return str(item.get("prdtClNm", "")).split(" > ")[0].strip()


def keyword_match_score(query: str, item_text: str) -> float:
    """검색어 단어 중 물품명/색상/분류에 직접 포함되는 것이 있으면 가산"""
    words = [w.strip() for w in query.split() if len(w.strip()) >= 2]
    if not words:
        return 0.0
    matched = sum(1 for w in words if _norm(w) in _norm(item_text))
    return min(1.0, matched / max(1, len(words)))


def rerank_results(query: str, results: List[dict]) -> List[dict]:
    """
    LoRA 검색 결과를 색상/카테고리/키워드 기반으로 재정렬.

    final score = 0.60 * lora + 0.20 * category + 0.15 * color + 0.05 * keyword
    """
    query_color = find_color(query)
    query_category = find_category(query)

    reranked = []
    for item in results:
        item = item.copy()

        lora_score = float(item.get("lora_score", item.get("score", 0.0)))
        # 비정상값 방어. 일반적으로 IP cosine은 -1~1, 좋은 결과는 0~1 부근.
        lora_score = max(0.0, min(1.0, lora_score))

        item_name = str(item.get("fdPrdtNm", ""))
        item_color = str(item.get("clrNm", ""))
        item_category = get_main_category(item)
        item_text = " ".join([item_name, item_color, item_category, str(item.get("prdtClNm", ""))])

        color_score = 0.0
        if query_color is not None:
            color_score = 1.0 if contains_any(item_text, COLOR_KEYWORDS[query_color]) else 0.0

        category_score = 0.0
        if query_category is not None:
            # 정확히 같은 대분류면 1점, 아니면 item_text에 관련 키워드가 있으면 0.6점
            if item_category == query_category:
                category_score = 1.0
            elif contains_any(item_text, CATEGORY_KEYWORDS[query_category]):
                category_score = 0.6

        key_score = keyword_match_score(query, item_text)

        final_score = (
            0.60 * lora_score
            + 0.20 * category_score
            + 0.15 * color_score
            + 0.05 * key_score
        )

        item["score"] = final_score
        item["lora_score"] = lora_score
        item["category_score"] = category_score
        item["color_score"] = color_score
        item["keyword_score"] = key_score

        reranked.append(item)

    reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return reranked
