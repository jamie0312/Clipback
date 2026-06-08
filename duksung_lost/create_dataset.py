import json
import pandas as pd
import os

def json_to_dataset_csv():
    json_file_path = "items.json"
    csv_filename = "lost_and_found_dataset.csv"
    
    if not os.path.exists(json_file_path):
        print(f"오류: 폴더에 {json_file_path} 파일이 없습니다. 팀원에게 받은 파일을 이 폴더로 옮겨주세요.")
        return

    print(f"'{json_file_path}' 파일을 읽어서 변환을 시작합니다...")
    
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        items_list = []
        
        # JSON 데이터가 리스트 형태인지, 딕셔너리 형태인지에 따라 대응
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict) and "items" in data:
            raw_items = data["items"]
        else:
            # 단일 객체인 경우
            raw_items = [data]

        for item in raw_items:
            img_url = item.get("imgUrl", "https://via.placeholder.com/224")
            # 💡 CLIP AI가 학습할 텍스트 문장으로 fdSbjt(상세 설명)를 사용합니다.
            description = item.get("fdSbjt", item.get("fdPrdtNm", "상세 설명 없음"))
            
            items_list.append({
                "img_file_path": img_url,
                "description": description
            })
            
        df = pd.DataFrame(items_list)
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"성공: 총 {len(df)}개의 팀원 데이터를 기반으로 '{csv_filename}' 파일을 만들었습니다!")
        print(df.head(2)) # 변환된 데이터 2개 미리보기
        
    except Exception as e:
        print(f"JSON 파싱 중 에러 발생: {e}")

if __name__ == "__main__":
    json_to_dataset_csv()