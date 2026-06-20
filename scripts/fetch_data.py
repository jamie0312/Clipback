import requests
import xml.etree.ElementTree as ET
import json
import time

SERVICE_KEY = API_KEY
URL = "http://apis.data.go.kr/1320000/LosfundInfoInqireService/getLosfundInfoAccToClAreaPd"
NO_IMG = "img02_no_img.gif"

def fetch_page(page_no, num_of_rows=100):
    params = {
        "serviceKey": SERVICE_KEY,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
    }
    res = requests.get(URL, params=params)
    root = ET.fromstring(res.text)
    items = root.findall(".//item")
    
    results = []
    for item in items:
        img_url = item.findtext("fdFilePathImg", "")
        if NO_IMG in img_url:  # 이미지 없는 것 스킵
            continue
        results.append({
            "atcId":     item.findtext("atcId"),
            "prdtClNm":  item.findtext("prdtClNm"),
            "clrNm":     item.findtext("clrNm"),
            "fdPrdtNm":  item.findtext("fdPrdtNm"),
            "fdSbjt":    item.findtext("fdSbjt"),
            "imgUrl":    img_url,
            "depPlace":  item.findtext("depPlace"),
            "fdYmd":     item.findtext("fdYmd"),
        })
    return results

def fetch_all(max_pages=50):
    all_data = []
    for page in range(1, max_pages + 1):
        print(f"Fetching page {page}...")
        items = fetch_page(page)
        all_data.extend(items)
        print(f"  → 이미지 있는 항목: {len(items)}개 (누적: {len(all_data)}개)")
        time.sleep(0.3)  # API 부하 방지
    
    with open("data/items.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n완료! 총 {len(all_data)}개 저장 → data/items.json")

if __name__ == "__main__":
    fetch_all(max_pages=50)