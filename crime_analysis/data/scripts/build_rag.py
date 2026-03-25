"""
建立 H-RAG 知識庫索引

執行前請先安裝依賴：
    pip install rank-bm25 jieba chromadb sentence-transformers

執行步驟：
    Step 1（如尚未抓取法條）：
        python data/scripts/fetch_laws.py

    Step 2（建立 RAG 索引）：
        cd crime_analysis
        python data/scripts/build_rag.py

索引儲存位置（ChromaDB）：  rag_db/chroma/
BM25 索引（pickle）：        rag_db/bm25_law.pkl / bm25_judgment.pkl
"""
import json
import logging
import pickle
import sys
from pathlib import Path

# ── 路徑設定：從 crime_analysis/ 執行 ──────────────────
# __file__ = crime_analysis/data/scripts/build_rag.py
CRIME_ANALYSIS_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(CRIME_ANALYSIS_DIR))

from config import cfg
from rag.preprocessor import (
    LawPreprocessor,
    JudgmentPreprocessor,
    LegalArticlePreprocessor,
)
from rag.h_rag import HierarchicalRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── 資料路徑 ────────────────────────────────────────────
LAW_DIR      = CRIME_ANALYSIS_DIR / cfg.rag.law_data_dir
JUDGMENT_DIR = CRIME_ANALYSIS_DIR / cfg.rag.judgment_data_dir
MANUAL_DIR   = CRIME_ANALYSIS_DIR / cfg.rag.manual_data_dir
BM25_DIR     = CRIME_ANALYSIS_DIR / "rag_db"


def load_law_chunks():
    """
    載入法條 chunks。

    優先順序：
    1. laws/criminal_code.json（由 fetch_laws.py 抓取）
    2. laws/*.json（其他法條 JSON）
    3. laws/*.txt（刑法全文文字檔，用 process_text() 解析）
    """
    preprocessor = LawPreprocessor()
    chunks = []

    law_dir = Path(LAW_DIR)
    if not law_dir.exists():
        logger.warning(f"法條資料夾不存在：{law_dir}")
        logger.warning("請先執行：python data/scripts/fetch_laws.py")
        return []

    # JSON 格式（fetch_laws.py 的輸出）
    for json_file in sorted(law_dir.glob("*.json")):
        logger.info(f"載入法條 JSON：{json_file.name}")
        new_chunks = preprocessor.process_file(str(json_file))
        chunks.extend(new_chunks)

    # 純文字格式（手動放入的刑法全文）
    for txt_file in sorted(law_dir.glob("*.txt")):
        logger.info(f"載入法條文字檔：{txt_file.name}")
        new_chunks = preprocessor.process_text(str(txt_file))
        chunks.extend(new_chunks)

    logger.info(f"法條 chunks 合計：{len(chunks)}")
    return chunks


def load_judgment_chunks():
    """
    載入裁判書 + 法學文章 chunks（統一放入 judgment 層）。

    資料夾結構（建議）：
        data/rag/judgments/
            *.json          ← 最高法院裁判書（parse_judgments.py 的輸出）
        data/rag/manuals/
            *.txt / *.json  ← 司法週刊、律點通等法學文章
    """
    chunks = []

    # 裁判書 JSON
    judgment_preprocessor = JudgmentPreprocessor()
    judgment_dir = Path(JUDGMENT_DIR)
    if judgment_dir.exists():
        for json_file in sorted(judgment_dir.glob("*.json")):
            logger.info(f"載入裁判書：{json_file.name}")
            chunks.extend(judgment_preprocessor.process_file(str(json_file)))
    else:
        logger.warning(f"裁判書資料夾不存在：{judgment_dir}（可跳過）")

    # 法學文章（司法週刊 / 律點通）
    article_preprocessor = LegalArticlePreprocessor()
    manual_dir = Path(MANUAL_DIR)
    if manual_dir.exists():
        for f in sorted(manual_dir.glob("*.txt")) + sorted(manual_dir.glob("*.json")):
            logger.info(f"載入法學文章：{f.name}")
            chunks.extend(article_preprocessor.process_file(str(f)))
    else:
        logger.warning(f"法學文章資料夾不存在：{manual_dir}（可跳過）")

    logger.info(f"裁判書/文章 chunks 合計：{len(chunks)}")
    return chunks


def save_bm25(rag: HierarchicalRAG):
    """將 BM25 模型 pickle 存檔（ChromaDB 已自動持久化，BM25 需手動）"""
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    for attr, filename in [
        ("_bm25_law", "bm25_law.pkl"),
        ("_bm25_judgment", "bm25_judgment.pkl"),
        ("_bm25_law_docs", "bm25_law_docs.pkl"),
        ("_bm25_judgment_docs", "bm25_judgment_docs.pkl"),
    ]:
        obj = getattr(rag, attr, None)
        if obj is not None:
            out_path = BM25_DIR / filename
            with open(out_path, "wb") as f:
                pickle.dump(obj, f)
            logger.info(f"BM25 儲存：{out_path}")


def print_summary(law_chunks, judgment_chunks):
    from collections import Counter
    print("\n=== RAG 知識庫統計 ===")

    law_cats = Counter(c.get("crime_category", "?") for c in law_chunks)
    print(f"\n【法條 chunks】共 {len(law_chunks)} 條")
    for cat, n in sorted(law_cats.items()):
        print(f"  {cat:<20} {n} 條")

    if judgment_chunks:
        seg_types = Counter(
            c.get("segment_type") or c.get("source", "?")
            for c in judgment_chunks
        )
        print(f"\n【裁判書/文章 chunks】共 {len(judgment_chunks)} 段落")
        for t, n in sorted(seg_types.items()):
            print(f"  {t:<20} {n} 段落")


def build():
    logger.info("=== 開始建立 H-RAG 索引 ===")

    # Step 1: 前處理
    law_chunks      = load_law_chunks()
    judgment_chunks = load_judgment_chunks()

    if not law_chunks and not judgment_chunks:
        logger.error("無任何資料可建索引，請確認資料路徑。")
        sys.exit(1)

    # to_dict() 統一序列化（LawChunk / JudgmentChunk / ArticleChunk 都有 to_dict）
    law_dicts      = [c.to_dict() if hasattr(c, "to_dict") else c for c in law_chunks]
    judgment_dicts = [c.to_dict() if hasattr(c, "to_dict") else c for c in judgment_chunks]

    # Step 2: 建索引
    rag = HierarchicalRAG()
    rag.build_index(law_dicts, judgment_dicts)

    # Step 3: 儲存 BM25（ChromaDB 已自動持久化到 rag_db/chroma/）
    save_bm25(rag)

    # Step 4: 驗證
    logger.info("\n=== 驗證查詢 ===")
    test_query = "依中華民國刑法，持刀傷害他人身體，涉及傷害罪構成要件。"
    result = rag.query(test_query)
    logger.info(f"查詢：{test_query[:40]}...")
    logger.info(f"  返回法條：{len(result['laws'])} 條")
    logger.info(f"  返回裁判書/文章：{len(result['judgments'])} 筆")
    if result["laws"]:
        top = result["laws"][0]
        logger.info(f"  Top-1 法條：第 {top.get('article_id', '?')} 條 ({top.get('crime_category', '?')})")

    print_summary(law_dicts, judgment_dicts)
    logger.info("\n=== H-RAG 索引建立完成 ===")
    logger.info(f"ChromaDB 位置：{Path(cfg.rag.chroma_persist_dir).resolve()}")
    logger.info(f"BM25 位置：   {BM25_DIR.resolve()}")


if __name__ == "__main__":
    build()
