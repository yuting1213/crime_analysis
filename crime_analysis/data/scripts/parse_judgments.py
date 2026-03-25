"""
將手動下載的裁判書文字檔轉換為標準 JSON 格式

使用方式：
    1. 從司法院手動下載裁判書全文（.txt 或複製貼上）
       放到 data/rag/judgments/raw/{category}/ 資料夾
    2. 執行此腳本：
           python data/scripts/parse_judgments.py
    3. 輸出：data/rag/judgments/{category}_cases.json

資料夾結構範例：
    data/rag/judgments/raw/
    ├── Assault/
    │   ├── 最高法院112年台上字第1234號.txt
    │   └── 高等法院111年上訴字第5678號.txt
    ├── Robbery/
    │   └── ...
    └── Fighting/
        └── ...
"""
import json
import re
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(encoding="utf-8")

# ── 設定 ──────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent / "rag/judgments/raw"
OUTPUT_DIR = Path(__file__).parent.parent / "rag/judgments"

UCF_CATEGORIES = [
    "Assault", "Robbery", "Stealing", "Shoplifting", "Burglary",
    "Fighting", "Arson", "Explosion", "RoadAccidents", "Vandalism",
    "Abuse", "Shooting", "Arrest",
]

# 條號 → 類別對照
ARTICLE_TO_CATEGORY = {
    "271": "Shooting", "272": "Shooting",
    "277": "Assault", "278": "Assault", "279": "Assault", "284": "Assault",
    "294": "Abuse", "295": "Abuse",
    "302": "Arrest", "304": "Arrest",
    "306": "Burglary", "307": "Burglary",
    "320": "Stealing", "321": "Stealing",
    "325": "Robbery", "326": "Robbery", "328": "Robbery", "329": "Robbery",
    "354": "Vandalism",
    "150": "Fighting",
    "173": "Arson", "174": "Arson",
    "178": "Explosion",
    "183": "RoadAccidents", "185-3": "RoadAccidents",
}


def parse_judgment_text(text: str, filename: str, category: str) -> dict:
    """
    解析單份裁判書文字，提取關鍵段落

    司法院裁判書典型結構：
      【主文】...
      【事實】...（或「犯罪事實」）
      【理由】...（或「論罪科刑之理由」）
      【據上論斷】...
    """
    # 清理多餘空白
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)

    # 提取案號（從檔名或文字內推斷）
    case_id = extract_case_id(text, filename)

    # 提取法院名稱
    court = extract_court(text)

    # 提取引用條號
    related_articles = extract_articles(text)

    # 提取各段落
    facts = extract_section(text, [
        r"犯\s*罪\s*事\s*實", r"事\s*實\s*及\s*理\s*由", r"一[、，]\s*事\s*實"
    ], [
        r"理\s*由", r"論\s*罪\s*科\s*刑", r"二[、，]"
    ])

    reasoning = extract_section(text, [
        r"理\s*由", r"論\s*罪\s*科\s*刑\s*之\s*理\s*由", r"二[、，]\s*理\s*由"
    ], [
        r"據\s*上\s*論\s*斷", r"主\s*文", r"三[、，]"
    ])

    verdict = extract_section(text, [r"主\s*文"], [r"事\s*實", r"理\s*由"])

    # 若段落提取失敗，取全文前後段
    if not facts and not reasoning:
        mid = len(text) // 2
        facts = text[:mid][:1500]
        reasoning = text[mid:][:1500]

    return {
        "case_id": case_id,
        "court": court,
        "crime_category": category,
        "related_articles": related_articles,
        "verdict": verdict[:500] if verdict else "",
        "facts": facts[:1500] if facts else "",
        "legal_reasoning": reasoning[:1500] if reasoning else "",
        "content": text[:3000],  # 完整前 3000 字備用
        "source_file": filename,
    }


def extract_case_id(text: str, filename: str) -> str:
    """從文字或檔名提取案號"""
    # 標準案號格式：最高法院112年台上字第1234號
    pattern = r"(?:最高法院|臺灣高等法院|高等法院|地方法院)[^\n]{0,20}(?:\d{2,3})年[^\n]{0,10}第\d+號"
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    # 從檔名推斷
    return Path(filename).stem


def extract_court(text: str) -> str:
    """提取裁判法院名稱"""
    pattern = r"(最高法院|臺灣高等法院(?:臺[中南東]分院)?|臺灣[^，\n]{2,6}(?:地方)?法院)"
    match = re.search(pattern, text[:500])
    return match.group(0) if match else "未知法院"


def extract_articles(text: str) -> list:
    """提取引用的刑法條號"""
    pattern = r"刑法第\s*(\d+(?:-\d+)?)\s*條"
    return list(set(re.findall(pattern, text)))


def extract_section(text: str, start_patterns: list, end_patterns: list) -> str:
    """
    提取特定段落（從 start_pattern 到 end_pattern 之間）
    start_patterns / end_patterns 為備選正規表達式清單，依序嘗試
    """
    start_pos = None
    for p in start_patterns:
        m = re.search(p, text)
        if m:
            start_pos = m.end()
            break

    if start_pos is None:
        return ""

    end_pos = None
    for p in end_patterns:
        m = re.search(p, text[start_pos:])
        if m:
            end_pos = start_pos + m.start()
            break

    if end_pos is None:
        end_pos = start_pos + 2000  # 找不到結尾就取 2000 字

    segment = text[start_pos:end_pos].strip()
    # 移除標題殘留
    segment = re.sub(r"^[：:]\s*", "", segment)
    return segment


def process_category(category: str) -> list:
    """處理單一犯罪類別的所有原始文字檔"""
    raw_path = RAW_DIR / category
    if not raw_path.exists():
        logger.warning(f"  資料夾不存在：{raw_path}")
        return []

    txt_files = list(raw_path.glob("*.txt")) + list(raw_path.glob("*.TXT"))
    if not txt_files:
        logger.warning(f"  {category}：無 .txt 檔案")
        return []

    results = []
    for f in txt_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            parsed = parse_judgment_text(text, f.name, category)
            if parsed["facts"] or parsed["legal_reasoning"]:
                results.append(parsed)
                logger.debug(f"  ✓ {f.name}")
            else:
                logger.warning(f"  ✗ {f.name}（無法解析段落）")
        except Exception as e:
            logger.error(f"  錯誤 {f.name}: {e}")

    return results


def save(category: str, judgments: list):
    output_path = OUTPUT_DIR / f"{category.lower()}_cases.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judgments, f, ensure_ascii=False, indent=2)
    logger.info(f"  儲存：{output_path}（{len(judgments)} 份）")


def create_raw_dirs():
    """建立原始資料暫存資料夾"""
    for cat in UCF_CATEGORIES:
        (RAW_DIR / cat).mkdir(parents=True, exist_ok=True)


def print_summary():
    """印出各類別數量與 LoRA 建議"""
    print("\n=== 裁判書統計 ===")
    total = 0
    need_lora = []
    for cat in UCF_CATEGORIES:
        json_path = OUTPUT_DIR / f"{cat.lower()}_cases.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                count = len(json.load(f))
        else:
            count = 0
        total += count
        status = "✓" if count >= 20 else f"⚠  不足（{count}/20）"
        print(f"  {cat:<20} {count:>3} 份  {status}")
        if count < 20:
            need_lora.append(cat)

    print(f"\n  合計：{total} 份")

    if need_lora:
        print(f"\n⚠  以下類別不足 20 份，建議補充或對 Semantic Agent 進行 LoRA 微調：")
        for cat in need_lora:
            print(f"   - {cat}")
    else:
        print("\n✓  所有類別均達標，可使用純 RAG，不需 LoRA 微調")


if __name__ == "__main__":
    # 建立資料夾結構
    create_raw_dirs()

    raw_exists = any((RAW_DIR / cat).glob("*.txt") for cat in UCF_CATEGORIES)

    if not raw_exists:
        print(f"""
尚未放入原始裁判書文字檔。

請按以下步驟操作：
1. 前往 https://judgment.judicial.gov.tw
2. 搜尋關鍵字（例如「刑法第277條」），篩選「最高法院」+「判決」
3. 將裁判書全文複製貼上，儲存為 .txt 檔
4. 依犯罪類別放入對應資料夾：
   {RAW_DIR}/
   ├── Assault/   ← 傷害罪（刑法277條）
   ├── Robbery/   ← 強盜罪（刑法328條）
   ├── Fighting/  ← 聚眾鬥毆（刑法150條）
   └── ...（其他類別）
5. 再次執行此腳本

建議每類別至少 20 份，若不足則需考慮 LoRA 微調。
""")
    else:
        logger.info("開始解析裁判書...")
        for category in UCF_CATEGORIES:
            logger.info(f"{category}")
            judgments = process_category(category)
            if judgments:
                save(category, judgments)

        print_summary()
