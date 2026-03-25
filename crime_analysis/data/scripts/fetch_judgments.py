"""
從司法院裁判書查詢系統抓取相關裁判書
來源：https://judgment.judicial.gov.tw

執行方式：
    python data/scripts/fetch_judgments.py
    python data/scripts/fetch_judgments.py --category Assault --max 50

輸出：
    data/rag/judgments/{category}_cases.json

注意：司法院有請求限速，腳本已加入 sleep 與重試機制
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 設定 ──────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "rag/judgments"

# 司法院裁判書 Open Data API
# 文件：https://data.judicial.gov.tw/jdg/api/swagger/index.html
JUDI_API_BASE = "https://data.judicial.gov.tw/jdg/api"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (academic research)",
    "Accept": "application/json",
}

# 每個犯罪類別的搜尋關鍵字（對應 TARGET_ARTICLES 的罪名）
SEARCH_QUERIES = {
    "Assault": ["傷害", "重傷害", "傷害致死"],
    "Robbery": ["強盜", "搶奪", "準強盜"],
    "Stealing": ["竊盜", "加重竊盜"],
    "Shoplifting": ["竊盜", "商店"],
    "Burglary": ["侵入住居竊盜", "毀越門扇"],
    "Fighting": ["聚眾鬥毆", "互毆", "群聚鬥毆"],
    "Arson": ["放火", "縱火"],
    "Explosion": ["爆炸", "危害公共安全"],
    "RoadAccidents": ["肇事逃逸", "過失傷害", "不能安全駕駛"],
    "Vandalism": ["毀損", "毀棄"],
    "Abuse": ["遺棄", "虐待"],
    "Shooting": ["殺人", "殺人未遂", "槍擊"],
    "Arrest": ["妨害自由", "強制", "私行拘禁"],
}

# 每類別預設抓取數量
DEFAULT_MAX_PER_CATEGORY = 30

# 目標：只抓「事實認定有爭議」的案件（有詳細推理過程）
# 法院類型：最高法院（TPS）、高等法院（TPH）
TARGET_COURT_TYPES = ["最高法院", "臺灣高等法院", "臺灣高等法院臺中分院"]


def search_judgments(keyword: str, page: int = 1, per_page: int = 20) -> dict:
    """
    呼叫司法院 Open Data API 搜尋裁判書
    GET /api/JudgDoc?kw={keyword}&page={page}&size={per_page}
    """
    url = f"{JUDI_API_BASE}/JudgDoc"
    params = {
        "kw": keyword,
        "page": page,
        "size": per_page,
        "jtype": "判決",       # 只要判決，不要裁定
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            logger.warning("請求過於頻繁，等待 10 秒後重試...")
            time.sleep(10)
            return search_judgments(keyword, page, per_page)
        logger.error(f"HTTP 錯誤: {e}")
        return {}
    except Exception as e:
        logger.error(f"搜尋失敗 keyword={keyword}: {e}")
        return {}


def fetch_judgment_detail(jid: str) -> Optional[str]:
    """
    抓取單份裁判書全文
    GET /api/JudgDoc/{jid}
    """
    url = f"{JUDI_API_BASE}/JudgDoc/{jid}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("jfull", "")  # 全文欄位
    except Exception as e:
        logger.error(f"取得全文失敗 jid={jid}: {e}")
        return None


def parse_judgment(raw: dict, full_text: str, crime_category: str) -> Optional[dict]:
    """
    將 API 回傳的裁判書資料轉為標準格式
    提取：事實段落、法律理由段落
    """
    case_id = raw.get("jid", "")
    court = raw.get("court", "")
    verdict_date = raw.get("jdate", "")

    if not full_text or len(full_text) < 200:
        return None

    # 提取引用條號
    related_articles = extract_article_references(full_text)

    # 提取事實段落（「事實」到「理由」之間的段落）
    facts = extract_segment(full_text, start_pattern=r"(事\s*實|犯罪事實)", end_pattern=r"(理\s*由|論罪科刑)")
    reasoning = extract_segment(full_text, start_pattern=r"(理\s*由|論罪科刑)", end_pattern=r"(主\s*文|判決主文|據上論斷)")

    if not facts and not reasoning:
        # 若無法分段，取前 1500 字
        facts = full_text[:1500]

    return {
        "case_id": case_id,
        "court": court,
        "verdict_date": verdict_date,
        "crime_category": crime_category,
        "related_articles": related_articles,
        "facts": facts.strip(),
        "legal_reasoning": reasoning.strip(),
        "content": full_text[:3000],   # 完整前 3000 字備用
    }


def fetch_category(category: str, max_count: int = DEFAULT_MAX_PER_CATEGORY) -> list:
    """抓取單一犯罪類別的裁判書"""
    queries = SEARCH_QUERIES.get(category, [category])
    results = []
    seen_ids = set()

    for keyword in queries:
        if len(results) >= max_count:
            break

        logger.info(f"  搜尋關鍵字：「{keyword}」")
        page = 1

        while len(results) < max_count:
            data = search_judgments(keyword, page=page, per_page=20)
            items = data.get("data", [])

            if not items:
                break

            for item in items:
                if len(results) >= max_count:
                    break

                jid = item.get("jid", "")
                if not jid or jid in seen_ids:
                    continue

                # 只取目標法院的判決
                court = item.get("court", "")
                if not any(t in court for t in TARGET_COURT_TYPES):
                    continue

                # 抓全文
                full_text = fetch_judgment_detail(jid)
                if not full_text:
                    continue

                parsed = parse_judgment(item, full_text, category)
                if parsed:
                    results.append(parsed)
                    seen_ids.add(jid)
                    logger.debug(f"    ✓ {jid} ({court})")

                time.sleep(0.8)  # 避免請求過於頻繁

            page += 1
            total_pages = data.get("totalPage", 1)
            if page > total_pages:
                break

            time.sleep(1)

    logger.info(f"  {category}：取得 {len(results)} 份裁判書")
    return results


def save_category(category: str, judgments: list):
    output_path = OUTPUT_DIR / f"{category.lower()}_cases.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judgments, f, ensure_ascii=False, indent=2)
    logger.info(f"  儲存：{output_path}")


# ── 文字處理工具 ──────────────────────────────────────

def extract_article_references(text: str) -> list:
    """從裁判書文字中提取引用的刑法條號"""
    pattern = r"刑法第\s*(\d+(?:-\d+)?)\s*條"
    return list(set(re.findall(pattern, text)))


def extract_segment(text: str, start_pattern: str, end_pattern: str) -> str:
    """
    提取兩個標題之間的段落
    例：「事實」到「理由」之間的文字
    """
    start_match = re.search(start_pattern, text)
    end_match = re.search(end_pattern, text)

    if not start_match:
        return ""

    start_pos = start_match.end()
    end_pos = end_match.start() if end_match and end_match.start() > start_pos else start_pos + 1000

    segment = text[start_pos:end_pos].strip()
    return segment[:1500]  # 最多取 1500 字


# ── 主流程 ────────────────────────────────────────────

def main(target_categories: list = None, max_per_category: int = DEFAULT_MAX_PER_CATEGORY):
    categories = target_categories or list(SEARCH_QUERIES.keys())

    logger.info(f"目標類別：{categories}")
    logger.info(f"每類別最多：{max_per_category} 份")

    total = 0
    for category in categories:
        logger.info(f"\n=== {category} ===")
        judgments = fetch_category(category, max_count=max_per_category)

        if judgments:
            save_category(category, judgments)
            total += len(judgments)
        else:
            logger.warning(f"  {category} 無資料")

    logger.info(f"\n完成！共取得 {total} 份裁判書")
    print_summary()


def print_summary():
    """印出各類別取得的數量摘要"""
    print("\n=== 裁判書統計 ===")
    total = 0
    for json_file in sorted(OUTPUT_DIR.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        count = len(data)
        total += count
        status = "✓" if count >= 20 else "⚠ 數量不足（建議 ≥20）"
        print(f"  {json_file.stem:<30} {count:>3} 份  {status}")
    print(f"  {'合計':<30} {total:>3} 份")

    if total < 100:
        print("\n⚠  裁判書總量偏少，建議補充或考慮 LoRA 微調 Semantic Agent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="抓取司法院裁判書")
    parser.add_argument(
        "--category", type=str, default=None,
        help="指定犯罪類別（預設抓全部），例：--category Assault"
    )
    parser.add_argument(
        "--max", type=int, default=DEFAULT_MAX_PER_CATEGORY,
        help=f"每類別最多抓幾份（預設 {DEFAULT_MAX_PER_CATEGORY}）"
    )
    args = parser.parse_args()

    categories = [args.category] if args.category else None
    main(target_categories=categories, max_per_category=args.max)
