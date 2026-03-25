"""
從全國法規資料庫抓取台灣刑法條文
來源：https://law.moj.gov.tw

執行方式：
    python data/scripts/fetch_laws.py
輸出：
    data/rag/laws/criminal_code.json
"""
import json
import re
import time
import logging
import urllib3
from pathlib import Path
import requests
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 設定 ──────────────────────────────────────────────
OUTPUT_PATH = Path(__file__).parent.parent / "rag/laws/criminal_code.json"
BASE_URL = "https://law.moj.gov.tw/LawClass/LawAll.aspx"

TARGET_LAWS = {
    "C0000001": "中華民國刑法",
    "C0010001": "中華民國刑事訴訟法",
}

# 只保留與 UCF-Crime 相關的條號（條號 → 類別）
TARGET_ARTICLES = {
    "150": "Fighting",   "151": "Fighting",
    "173": "Arson",      "174": "Arson",      "175": "Arson",   "176": "Arson",
    "178": "Explosion",  "183": "RoadAccidents", "184": "RoadAccidents",
    "185": "RoadAccidents", "185-3": "RoadAccidents",
    "271": "Shooting",   "272": "Shooting",
    "277": "Assault",    "278": "Assault",    "279": "Assault", "284": "Assault",
    "294": "Abuse",      "295": "Abuse",
    "302": "Arrest",     "304": "Arrest",
    "306": "Burglary",   "307": "Burglary",
    "320": "Stealing",   "321": "Stealing",
    "325": "Robbery",    "326": "Robbery",    "328": "Robbery", "329": "Robbery",
    "354": "Vandalism",  "355": "Vandalism",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)"}


def fetch_law_html(pcode: str) -> str:
    try:
        resp = requests.get(
            BASE_URL, params={"pcode": pcode},
            headers=HEADERS, timeout=20, verify=False,
        )
        resp.encoding = "utf-8"
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"抓取失敗 pcode={pcode}: {e}")
        return ""


def parse_articles(html: str, law_name: str) -> list:
    """
    解析法規資料庫 HTML 條文結構：
      <div class="col-no"><a name="277">第 277 條</a></div>
      <div class="col-data"><div class="line-0000">傷害人之身體...</div></div>
    """
    soup = BeautifulSoup(html, "lxml")
    results = []
    current_chapter = ""

    for row in soup.find_all("div", class_="row"):
        # 更新章節資訊
        no_div = row.find("div", class_="col-no")
        if no_div:
            text = no_div.get_text(strip=True)
            if "章" in text and "條" not in text:
                current_chapter = text
                continue

        # 找條號（<a name="277">）
        anchor = row.find("a", attrs={"name": re.compile(r"^\d+(-\d+)?$")})
        if not anchor:
            continue

        article_id = anchor["name"].strip()
        crime_category = TARGET_ARTICLES.get(article_id)
        if crime_category is None:
            continue

        # 取條文內容
        data_div = row.find("div", class_="col-data")
        if not data_div:
            continue

        content = data_div.get_text(separator="\n", strip=True)
        content = re.sub(r"\n{3,}", "\n\n", content)

        if not content:
            continue

        results.append({
            "article_id": article_id,
            "law_name": law_name,
            "crime_category": crime_category,
            "chapter": current_chapter,
            "content": f"第 {article_id} 條　{content}",
        })

    return results


def fetch_all() -> list:
    all_articles = []
    for pcode, law_name in TARGET_LAWS.items():
        logger.info(f"抓取 {law_name} (pcode={pcode})...")
        html = fetch_law_html(pcode)
        if not html:
            continue
        articles = parse_articles(html, law_name)
        logger.info(f"  取得 {len(articles)} 條相關條文")
        all_articles.extend(articles)
        time.sleep(1)
    return all_articles


def save(articles: list):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"儲存：{OUTPUT_PATH}（{len(articles)} 條）")


def print_summary(articles: list):
    from collections import Counter
    counter = Counter(a["crime_category"] for a in articles)
    print("\n=== 法條統計 ===")
    for cat, count in sorted(counter.items()):
        print(f"  {cat:<20} {count} 條")
    print(f"  {'合計':<20} {len(articles)} 條")


if __name__ == "__main__":
    logger.info("開始抓取台灣刑事法條...")
    articles = fetch_all()
    if articles:
        save(articles)
        print_summary(articles)
        print("\n=== 第 277 條範例 ===")
        sample = next((a for a in articles if a["article_id"] == "277"), None)
        if sample:
            print(json.dumps(sample, ensure_ascii=False, indent=2))
    else:
        logger.error("未取得任何條文")
