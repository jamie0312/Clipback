# download_images.py
import json, requests, os, signal, sys
from pathlib import Path
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

with open("data/items.json", encoding="utf-8") as f:
    items = json.load(f)

Path("data/images").mkdir(parents=True, exist_ok=True)

def download(item):
    out_path = f"data/images/{item['atcId']}.jpg"
    if os.path.exists(out_path):
        return "skip"
    try:
        res = requests.get(item["imgUrl"], timeout=5)
        img = Image.open(BytesIO(res.content)).convert("RGB")
        img.save(out_path)
        return "ok"
    except:
        return "fail"

ok = skip = fail = 0

try:
    with ThreadPoolExecutor(max_workers=16) as exe:
        futures = {exe.submit(download, item): item for item in items}
        with tqdm(total=len(items), ncols=70) as pbar:
            for f in as_completed(futures):
                result = f.result()
                if result == "ok":   ok   += 1
                elif result == "skip": skip += 1
                else:                fail += 1
                pbar.set_postfix(ok=ok, skip=skip, fail=fail)
                pbar.update(1)
except KeyboardInterrupt:
    print(f"\n중단됨. ok={ok} skip={skip} fail={fail}")
    sys.exit(0)

print(f"\n완료! ok={ok} skip={skip} fail={fail}")
print(f"저장된 이미지: {len(os.listdir('data/images'))}개")