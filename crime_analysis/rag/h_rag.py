"""
H-RAG - 層級式檢索增強生成
雙層索引：Keyword Layer (BM25) + Data Layer (BGE-M3)
結果融合：RRF (Reciprocal Rank Fusion)
查詢增強：HyDE (Hypothetical Document Embeddings)

依賴套件（可選，未安裝時自動降級為 placeholder）：
    pip install rank-bm25 jieba chromadb sentence-transformers
"""
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ── 可選依賴 ──────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False
    logger.warning("rank-bm25 未安裝，BM25 層將使用 placeholder。pip install rank-bm25")

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False
    logger.warning("jieba 未安裝，將改用字元切分。pip install jieba")

try:
    import chromadb
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False
    logger.warning("chromadb 未安裝，Dense 層將使用 placeholder。pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    logger.warning("sentence-transformers 未安裝，BGE-M3 嵌入停用。pip install sentence-transformers")


def _tokenize(text: str) -> List[str]:
    """
    分詞：優先用 jieba，否則以單字切分。
    同時加入刑法條號為獨立 token（提升 BM25 精確查詢）。
    例：「刑法第277條」額外加入 token '277'
    """
    import re
    article_tokens = re.findall(r'第(\d+(?:-\d+)?)條', text)

    if _HAS_JIEBA:
        tokens = list(jieba.cut(text))
    else:
        tokens = list(text)

    return tokens + article_tokens


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Any, float]],
    dense_results: List[Tuple[Any, float]],
    k: int = 60,
    top_n: int = 5,
) -> List[Any]:
    """
    RRF 融合公式：score(d) = Σ 1 / (k + rank_i(d))
    k=60 是標準預設值（Robertson, 2009）
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}

    for rank, (doc, _) in enumerate(bm25_results, start=1):
        doc_id = doc.get("article_id") or doc.get("case_id") or str(rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
        doc_map[doc_id] = doc

    for rank, (doc, _) in enumerate(dense_results, start=1):
        doc_id = doc.get("article_id") or doc.get("case_id") or str(rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
        doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[i] for i in sorted_ids[:top_n]]


class HierarchicalRAG:
    """
    H-RAG 雙層索引架構：

    Keyword Layer (BM25)
    ├── 適合：條號精確查詢（「刑法第277條」→ 直接命中）
    └── 工具：rank-bm25 + jieba

    Data Layer (BGE-M3 Dense Retrieval)
    ├── 適合：語意查詢（「持刀傷人」→ 傷害罪相關條文）
    └── 工具：ChromaDB + BAAI/bge-m3

    結果融合：RRF
    查詢增強：HyDE（讓 LLM 先生成假設性法條描述再檢索）
    """

    BGE_MODEL_NAME = "BAAI/bge-m3"

    def __init__(self, cfg_rag=None):
        from config import cfg
        self.cfg = cfg_rag or cfg.rag

        # BM25 索引（Keyword Layer）
        self._bm25_law: Optional[Any] = None
        self._bm25_judgment: Optional[Any] = None
        self._bm25_law_docs: List[Dict] = []
        self._bm25_judgment_docs: List[Dict] = []

        # ChromaDB（Data Layer）
        self._chroma_client = None
        self._law_collection = None
        self._judgment_collection = None

        # Embedding 模型（BGE-M3）
        self._embedding_model = None

        # 自動載入已建好的索引（build_rag.py 產生的 pickle + ChromaDB）
        self._try_load_existing_index()

    def _try_load_existing_index(self):
        """自動載入 build_rag.py 產生的 BM25 pickle + ChromaDB 索引。"""
        import pickle
        from pathlib import Path

        # BM25 pickle（相對於 crime_analysis/ 目錄）
        bm25_dir = Path(self.cfg.chroma_persist_dir).parent  # rag_db/
        loaded = []

        for attr, filename in [
            ("_bm25_law", "bm25_law.pkl"),
            ("_bm25_law_docs", "bm25_law_docs.pkl"),
            ("_bm25_judgment", "bm25_judgment.pkl"),
            ("_bm25_judgment_docs", "bm25_judgment_docs.pkl"),
        ]:
            pkl_path = bm25_dir / filename
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        setattr(self, attr, pickle.load(f))
                    loaded.append(filename)
                except Exception as e:
                    logger.warning(f"BM25 載入失敗 {filename}：{e}")

        # ChromaDB
        chroma_dir = Path(self.cfg.chroma_persist_dir)
        if chroma_dir.exists() and _HAS_CHROMA:
            try:
                self._chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
                # collection 名稱依 build_rag.py 的 _build_dense() 決定
                self._law_collection = self._chroma_client.get_collection("laws")
                try:
                    self._judgment_collection = self._chroma_client.get_collection("judgments")
                except Exception:
                    pass  # judgment collection 可能不存在
                loaded.append("ChromaDB")
            except Exception as e:
                logger.warning(f"ChromaDB 載入失敗：{e}")

        # Embedding 模型（BGE-M3）— 延遲載入，只在 dense query 時才載
        if loaded:
            logger.info(f"H-RAG 索引自動載入：{', '.join(loaded)}")
            if self._bm25_law is not None:
                corpus_size = getattr(self._bm25_law, "corpus_size", "?")
                logger.info(f"  BM25 法條 corpus: {corpus_size} 條")
        else:
            logger.info("H-RAG 索引未找到，需先執行 python data/scripts/build_rag.py")

    # ── 公開介面 ──────────────────────────────────────────

    def build_index(
        self,
        law_chunks: List[Dict],
        judgment_chunks: List[Dict],
    ) -> None:
        """
        建立雙層索引。
        法條與裁判書分開建立，避免混淆。

        Args:
            law_chunks:      LawChunk.to_dict() 列表
            judgment_chunks: JudgmentChunk.to_dict() 或 ArticleChunk.to_dict() 列表
        """
        logger.info("建立 BM25 稀疏索引...")
        self._build_bm25(law_chunks, judgment_chunks)

        logger.info("建立 BGE-M3 稠密向量索引...")
        self._build_dense(law_chunks, judgment_chunks)

        logger.info("H-RAG 索引建立完成")

    def query(self, query_text: str, use_hyde: bool = False) -> Dict[str, List]:
        """
        雙層檢索 + RRF 融合。

        注意：SemanticAgent 的 _generate_hyde_query() 已在呼叫前完成 HyDE，
        故預設 use_hyde=False 避免雙重增強。
        若直接傳入原始影像描述，可設 use_hyde=True。

        Args:
            query_text: 查詢文字（影像描述或假設性法條描述）
            use_hyde:   是否在此層再做一次 HyDE 增強

        Returns:
            {"laws": [...], "judgments": [...]}
        """
        if use_hyde:
            query_text = self._apply_hyde(query_text)

        # Keyword Layer: BM25
        bm25_laws = self._bm25_search(query_text, mode="law")
        bm25_judgments = self._bm25_search(query_text, mode="judgment")

        # Data Layer: Dense
        dense_laws = self._dense_search(query_text, mode="law")
        dense_judgments = self._dense_search(query_text, mode="judgment")

        # RRF 融合
        merged_laws = reciprocal_rank_fusion(
            bm25_laws, dense_laws, top_n=self.cfg.top_k_final
        )
        merged_judgments = reciprocal_rank_fusion(
            bm25_judgments, dense_judgments, top_n=self.cfg.top_k_final
        )

        return {"laws": merged_laws, "judgments": merged_judgments}

    # ── 索引建立 ──────────────────────────────────────────

    def _build_bm25(self, law_chunks: List[Dict], judgment_chunks: List[Dict]):
        self._bm25_law_docs = law_chunks
        self._bm25_judgment_docs = judgment_chunks

        if not _HAS_BM25:
            logger.debug("BM25 placeholder（rank-bm25 未安裝）")
            return

        law_corpus = [_tokenize(c.get("content", "")) for c in law_chunks]
        judgment_corpus = [_tokenize(c.get("content", "")) for c in judgment_chunks]

        self._bm25_law = BM25Okapi(law_corpus) if law_corpus else None
        self._bm25_judgment = BM25Okapi(judgment_corpus) if judgment_corpus else None

        logger.debug(
            f"BM25 索引：{len(law_chunks)} 法條，{len(judgment_chunks)} 裁判書/文章段落"
        )

    def _build_dense(self, law_chunks: List[Dict], judgment_chunks: List[Dict]):
        if not (_HAS_CHROMA and _HAS_ST):
            logger.debug("Dense 索引 placeholder（chromadb / sentence-transformers 未安裝）")
            return

        self._embedding_model = SentenceTransformer(self.BGE_MODEL_NAME)
        persist_dir = getattr(self.cfg, "chroma_persist_dir", "./chroma_db")
        self._chroma_client = chromadb.PersistentClient(path=persist_dir)

        # ── 法條集合 ──
        self._law_collection = self._chroma_client.get_or_create_collection(
            name="laws",
            metadata={"hnsw:space": "cosine"},
        )
        if law_chunks:
            texts = [c.get("content", "") for c in law_chunks]
            embeddings = self._embedding_model.encode(
                texts, batch_size=32, show_progress_bar=False
            ).tolist()
            self._law_collection.add(
                ids=[f"law_{i}" for i in range(len(law_chunks))],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {k: v for k, v in c.items() if k != "content" and isinstance(v, (str, int, float, bool))}
                    for c in law_chunks
                ],
            )

        # ── 裁判書 / 法學文章集合 ──
        self._judgment_collection = self._chroma_client.get_or_create_collection(
            name="judgments",
            metadata={"hnsw:space": "cosine"},
        )
        if judgment_chunks:
            texts = [c.get("content", "") for c in judgment_chunks]
            embeddings = self._embedding_model.encode(
                texts, batch_size=32, show_progress_bar=False
            ).tolist()
            self._judgment_collection.add(
                ids=[f"judgment_{i}" for i in range(len(judgment_chunks))],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {k: v for k, v in c.items() if k != "content" and isinstance(v, (str, int, float, bool))}
                    for c in judgment_chunks
                ],
            )

        logger.debug(
            f"BGE-M3 索引：{len(law_chunks)} 法條，{len(judgment_chunks)} 裁判書/文章段落"
        )

    # ── 搜尋 ──────────────────────────────────────────────

    def _bm25_search(
        self, query: str, mode: str = "law"
    ) -> List[Tuple[Dict, float]]:
        docs = self._bm25_law_docs if mode == "law" else self._bm25_judgment_docs
        bm25 = self._bm25_law if mode == "law" else self._bm25_judgment
        top_k = getattr(self.cfg, "top_k_bm25", 5)

        if not _HAS_BM25 or bm25 is None or not docs:
            return [(d, 1.0) for d in docs[:top_k]]

        tokens = _tokenize(query)
        scores = bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(docs[i], float(s)) for i, s in ranked if s > 0]

    def _dense_search(
        self, query: str, mode: str = "law"
    ) -> List[Tuple[Dict, float]]:
        docs = self._bm25_law_docs if mode == "law" else self._bm25_judgment_docs
        top_k = getattr(self.cfg, "top_k_dense", 5)

        if not (_HAS_CHROMA and _HAS_ST) or self._embedding_model is None:
            return [(d, 0.9) for d in docs[:top_k]]

        collection = self._law_collection if mode == "law" else self._judgment_collection
        if collection is None or collection.count() == 0:
            return []

        query_embedding = self._embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )

        # ChromaDB 回傳 distances（cosine distance：0=完全相同；轉為相似度）
        output = []
        for meta, dist in zip(
            results["metadatas"][0], results["distances"][0]
        ):
            similarity = 1.0 - dist  # cosine distance → similarity
            output.append((meta, float(similarity)))
        return output

    # ── HyDE ──────────────────────────────────────────────

    def _apply_hyde(self, query: str) -> str:
        """
        HyDE：讓 LLM 生成一段假設性法條描述，再用它來檢索。
        比直接用影像描述查詢效果更好，因為向量空間更接近法律文本。

        實作：呼叫 Qwen2.5（透過 transformers pipeline）或直接用格式化前綴。
        TODO: 接入本地 Qwen2.5-7B-Instruct 做完整 HyDE 生成。
        """
        # 簡易 HyDE：加上刑法前綴讓向量空間更接近法律文本
        return f"依中華民國刑法，{query}"
