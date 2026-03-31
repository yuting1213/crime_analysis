"""
RAG 知識庫前處理
- LawPreprocessor: 台灣刑事法條結構化切割（JSON 或全文文字）
- JudgmentPreprocessor: 最高法院裁判書段落切割
- LegalArticlePreprocessor: 司法週刊 / 律點通等法學文章段落切割
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class LawChunk:
    """法條切割單元"""
    article_id: str          # 例：'277'（刑法第277條）
    law_name: str            # 例：'中華民國刑法'
    crime_category: str      # 例：'Assault'
    content: str             # 條文全文
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "article_id": self.article_id,
            "law_name": self.law_name,
            "crime_category": self.crime_category,
            "content": self.content,
            **self.metadata,
        }


@dataclass
class JudgmentChunk:
    """裁判書段落切割單元"""
    case_id: str             # 裁判書字號
    segment_type: str        # 'facts' / 'legal_reasoning' / 'verdict'
    content: str
    related_articles: List[str] = field(default_factory=list)
    crime_category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "segment_type": self.segment_type,
            "content": self.content,
            "related_articles": self.related_articles,
            "crime_category": self.crime_category,
            **self.metadata,
        }


@dataclass
class ArticleChunk:
    """法學文章段落切割單元（司法週刊 / 律點通）"""
    source: str              # 來源名稱，例：'司法週刊' / '律點通'
    title: str               # 文章標題
    paragraph_idx: int       # 段落序號（0-based）
    content: str             # 段落文字
    related_articles: List[str] = field(default_factory=list)
    crime_category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "title": self.title,
            "paragraph_idx": self.paragraph_idx,
            "content": self.content,
            "related_articles": self.related_articles,
            "crime_category": self.crime_category,
            **self.metadata,
        }


# 法條到 UCF-Crime 類別的對映（可擴充）
ARTICLE_TO_CATEGORY: Dict[str, str] = {
    "271": "Shooting",      # 殺人罪
    "272": "Shooting",      # 殺直系血親尊親屬
    "277": "Assault",       # 傷害罪
    "278": "Assault",       # 重傷罪
    "281": "Assault",       # 加重傷害罪
    "325": "Robbery",       # 搶奪罪
    "328": "Robbery",       # 強盜罪
    "329": "Robbery",       # 準強盜罪
    "330": "Robbery",       # 加重強盜罪
    "320": "Stealing",      # 竊盜罪
    "321": "Stealing",      # 加重竊盜罪（含侵入建築物）
    "354": "Vandalism",     # 毀損罪
    "150": "Fighting",      # 公然聚眾施強暴
    "185-3": "RoadAccidents",  # 不能安全駕駛罪
    "276": "RoadAccidents", # 過失致死罪
    "173": "Arson",         # 放火罪
    "174": "Arson",         # 放火建築物罪
    "184": "Explosion",     # 爆炸罪
    "185-1": "Explosion",   # 公共危險罪（爆炸物）
    "304": "Arrest",        # 強制罪
}


class LawPreprocessor:
    """
    台灣刑事法條結構化前處理

    支援兩種輸入格式：
    1. process_file()  ← JSON，每筆記錄含 article_id, law_name, content
    2. process_text()  ← 刑法全文文字檔（支援「第NNN條」格式）

    切割單位：以「條」為單位，保留完整條文
    BM25 關鍵字段：article_id + keywords（條文中的法律術語）
    """

    def process_file(self, filepath: str) -> List[LawChunk]:
        """從 JSON 格式的法條資料建立 chunks"""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"法條檔案不存在：{filepath}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for item in data:
            article_id = str(item.get("article_id", ""))
            crime_category = item.get("crime_category") or ARTICLE_TO_CATEGORY.get(article_id, "Other")

            chunk = LawChunk(
                article_id=article_id,
                law_name=item.get("law_name", "中華民國刑法"),
                crime_category=crime_category,
                content=item.get("content", ""),
                metadata={
                    "chapter": item.get("chapter", ""),
                    "keywords": self._extract_keywords(item.get("content", "")),
                },
            )
            chunks.append(chunk)

        logger.info(f"法條前處理完成（JSON）：{len(chunks)} 條")
        return chunks

    def process_text(self, filepath: str, law_name: str = "中華民國刑法") -> List[LawChunk]:
        """
        從刑法全文文字檔解析法條，每條一個 chunk。

        支援格式（含章節標題自動忽略）：
            第二七七條　（傷害罪）
            傷害他人之身體或健康者，處五年以下有期徒刑...
        或：
            第277條
            傷害他人之身體或健康者...
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"法條檔案不存在：{filepath}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return self._parse_criminal_code_text(text, law_name)

    def _parse_criminal_code_text(
        self, text: str, law_name: str = "中華民國刑法"
    ) -> List[LawChunk]:
        """
        以正規表達式切分台灣刑法全文。

        條號格式：第NNN條 或 第NNN-N條（含之一、之二等附條）
        例：第321條、第185-3條、第321-1條
        """
        # 同時匹配阿拉伯數字條號（第277條）與附條（第185-3條）
        article_re = re.compile(
            r'第\s*(\d+(?:-\d+)?)\s*條\s*(?:（[^）]*）)?\s*\n(.*?)(?=第\s*\d+(?:-\d+)?\s*條|\Z)',
            re.DOTALL,
        )

        chunks = []
        for match in article_re.finditer(text):
            article_id = match.group(1).strip()
            body = match.group(2).strip()
            # 過濾章節標題行（全為漢字且長度 < 20）
            body_lines = [
                ln for ln in body.splitlines()
                if not re.fullmatch(r'[第一二三四五六七八九十章節篇\s]+', ln.strip())
                and ln.strip()
            ]
            content = f"第{article_id}條\n" + "\n".join(body_lines)

            if len(content.strip()) < 10:
                continue

            crime_category = ARTICLE_TO_CATEGORY.get(article_id, "Other")
            chunk = LawChunk(
                article_id=article_id,
                law_name=law_name,
                crime_category=crime_category,
                content=content,
                metadata={"keywords": self._extract_keywords(content)},
            )
            chunks.append(chunk)

        logger.info(f"法條文字前處理完成：{len(chunks)} 條")
        return chunks

    # 法律術語詞典（用於 jieba 自定義詞 + BM25 索引增強）
    LEGAL_TERMS = {
        # 行為態樣
        "傷害", "殺人", "竊盜", "搶奪", "強盜", "毀損", "放火", "爆炸",
        "強制", "逮捕", "重傷", "施強暴", "脅迫", "遺棄", "拘禁",
        "侵入住宅", "竊取", "搶奪", "勒贖", "縱火",
        # 主觀要件
        "故意", "過失", "意圖", "不法所有意圖", "殺傷意圖",
        "直接故意", "間接故意",
        # 構成要件
        "因果關係", "違法性", "有責性", "構成要件",
        "傷害行為", "傷害結果", "竊取行為", "毀損行為", "放火行為",
        "強暴", "互毆", "秘密竊取",
        # 阻卻事由
        "正當防衛", "緊急避難", "阻卻違法",
        # 法律用語
        "公共危險", "妨害自由", "妨害秩序",
        "有期徒刑", "無期徒刑", "拘役", "罰金",
        "未遂", "既遂", "加重", "減輕",
    }
    _jieba_loaded = False

    def _extract_keywords(self, content: str) -> List[str]:
        """
        使用 jieba 分詞 + 法律術語詞典提取關鍵詞。
        兩層策略：(1) 詞典直接匹配 (2) jieba 分詞後過濾法律詞彙。
        """
        # 載入自定義詞典（只做一次）
        if not LawPreprocessor._jieba_loaded:
            try:
                import jieba
                for term in self.LEGAL_TERMS:
                    jieba.add_word(term, freq=10000)
                LawPreprocessor._jieba_loaded = True
            except ImportError:
                pass

        # 層 1：直接匹配法律術語
        matched = {t for t in self.LEGAL_TERMS if t in content}

        # 層 2：jieba 分詞，保留 >= 2 字的詞（過濾虛詞）
        try:
            import jieba
            tokens = jieba.cut(content)
            for tok in tokens:
                if len(tok) >= 2 and tok in self.LEGAL_TERMS:
                    matched.add(tok)
        except ImportError:
            pass

        return sorted(matched)


class JudgmentPreprocessor:
    """
    最高法院裁判書前處理
    切割單位：以「事實認定段落」為單位，保留跨段落語境
    重要：法條與裁判書分開處理，建立不同的向量索引
    """

    # 段落類型識別的正規表達式
    SEGMENT_PATTERNS = {
        "facts": r"(事\s*實|犯罪事實|案情概要)",
        "legal_reasoning": r"(理\s*由|法律理由|論罪科刑)",
        "verdict": r"(主\s*文|判決主文)",
    }

    # 所有段落標題的聯合正規表達式（用於定位切割邊界）
    _ALL_HEADERS_RE = re.compile(
        r'(?:事\s*實|犯罪事實|案情概要'
        r'|理\s*由|法律理由|論罪科刑'
        r'|主\s*文|判決主文)'
    )

    def process_file(self, filepath: str) -> List[JudgmentChunk]:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"裁判書檔案不存在：{filepath}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for item in data:
            case_chunks = self._split_judgment(item)
            chunks.extend(case_chunks)

        logger.info(f"裁判書前處理完成：{len(chunks)} 段落")
        return chunks

    def _split_judgment(self, judgment: Dict) -> List[JudgmentChunk]:
        """
        將一份裁判書切分為多個段落：
        - 事實段落：包含犯罪行為描述
        - 法律理由段落：包含法條引用與推理
        - 主文段落：判決結果
        """
        case_id = judgment.get("case_id", "unknown")
        full_text = judgment.get("content", "")
        related_articles = self._extract_article_references(full_text)
        crime_category = self._infer_category(related_articles)

        chunks = []

        for seg_type, pattern in self.SEGMENT_PATTERNS.items():
            segments = self._extract_segment(full_text, pattern)
            for seg_text in segments:
                if len(seg_text.strip()) < 50:
                    continue
                chunk = JudgmentChunk(
                    case_id=case_id,
                    segment_type=seg_type,
                    content=seg_text.strip(),
                    related_articles=related_articles,
                    crime_category=crime_category,
                    metadata={"source": "supreme_court"},
                )
                chunks.append(chunk)

        # 若無法分段，整份作為一個 chunk
        if not chunks and full_text:
            chunks.append(JudgmentChunk(
                case_id=case_id,
                segment_type="full",
                content=full_text[:1000],
                related_articles=related_articles,
                crime_category=crime_category,
            ))

        return chunks

    def _extract_article_references(self, text: str) -> List[str]:
        """提取文中引用的刑法條號，例：'刑法第277條' → '277'"""
        pattern = r"刑法第(\d+(?:-\d+)?)條"
        return list(set(re.findall(pattern, text)))

    def _infer_category(self, articles: List[str]) -> str:
        """根據引用條號推斷犯罪類別"""
        for article in articles:
            if article in ARTICLE_TO_CATEGORY:
                return ARTICLE_TO_CATEGORY[article]
        return "Other"

    def _extract_segment(self, text: str, pattern: str) -> List[str]:
        """
        用正規表達式提取特定段落類型的文字。
        找到匹配 pattern 的標題，取到下一個段落標題為止的文字。
        """
        target_re = re.compile(pattern)

        # 找出所有段落標題的位置
        all_header_positions = [
            (m.start(), m.end())
            for m in self._ALL_HEADERS_RE.finditer(text)
        ]

        if not all_header_positions:
            return []

        segments = []
        for idx, (h_start, h_end) in enumerate(all_header_positions):
            header_text = text[h_start:h_end]
            if not target_re.search(header_text):
                continue

            # 本段落結束於下一個標題的開頭，或文末
            if idx + 1 < len(all_header_positions):
                seg_end = all_header_positions[idx + 1][0]
            else:
                seg_end = len(text)

            segment = text[h_start:seg_end].strip()
            if segment:
                segments.append(segment)

        return segments


class LegalArticlePreprocessor:
    """
    法學文章前處理（司法週刊 / 律點通等）

    切割單位：以「段落」為單位（空行分隔），每段落保留來源與標題。
    適合 BGE-M3 語意索引；不加入 BM25 keyword 層（無精確條號查詢需求）。

    輸入格式（純文字 .txt）：
        [標題]
        文章內文，以空行分隔段落...

    或 JSON：
        [{"source": "律點通", "title": "...", "content": "..."}]
    """

    MIN_PARAGRAPH_LEN = 80   # 低於此長度的段落略過

    def process_file(self, filepath: str) -> List[ArticleChunk]:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"法學文章檔案不存在：{filepath}")
            return []

        if path.suffix.lower() == ".json":
            return self._process_json(path)
        else:
            return self._process_text(path)

    def _process_json(self, path: Path) -> List[ArticleChunk]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for item in data:
            article_chunks = self._split_article(
                source=item.get("source", path.stem),
                title=item.get("title", ""),
                content=item.get("content", ""),
            )
            chunks.extend(article_chunks)

        logger.info(f"法學文章前處理完成（JSON）：{len(chunks)} 段落")
        return chunks

    def _process_text(self, path: Path) -> List[ArticleChunk]:
        """
        純文字格式：
        第一行為標題（若以「標題：」或「題目：」開頭則去掉前綴）；
        其後每個空行分隔一個段落。
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        lines = text.splitlines()
        # 擷取標題
        title = ""
        start_idx = 0
        if lines:
            first = lines[0].strip()
            title_match = re.match(r'^(?:標題|題目|Title)[：:]\s*(.+)', first)
            title = title_match.group(1) if title_match else first
            start_idx = 1

        body = "\n".join(lines[start_idx:])
        chunks = self._split_article(
            source=path.stem,
            title=title,
            content=body,
        )
        logger.info(f"法學文章前處理完成（TXT）：{len(chunks)} 段落，來源：{path.stem}")
        return chunks

    def _split_article(
        self, source: str, title: str, content: str
    ) -> List[ArticleChunk]:
        """
        以空行（或雙換行）切分段落，過濾過短段落。
        每段落記錄引用條號和推斷犯罪類別。
        """
        raw_paragraphs = re.split(r'\n{2,}', content)
        related_articles = self._extract_article_references(content)
        crime_category = self._infer_category(related_articles)

        chunks = []
        for idx, para in enumerate(raw_paragraphs):
            para = para.strip()
            if len(para) < self.MIN_PARAGRAPH_LEN:
                continue

            # 段落內自己也可能提及特定條號，再細化
            para_articles = self._extract_article_references(para)
            para_category = self._infer_category(para_articles) if para_articles else crime_category

            chunk = ArticleChunk(
                source=source,
                title=title,
                paragraph_idx=idx,
                content=para,
                related_articles=para_articles or related_articles,
                crime_category=para_category,
                metadata={"char_count": len(para)},
            )
            chunks.append(chunk)

        return chunks

    def _extract_article_references(self, text: str) -> List[str]:
        """提取文中引用的刑法條號"""
        pattern = r"刑法第(\d+(?:-\d+)?)條"
        return list(set(re.findall(pattern, text)))

    def _infer_category(self, articles: List[str]) -> str:
        for article in articles:
            if article in ARTICLE_TO_CATEGORY:
                return ARTICLE_TO_CATEGORY[article]
        return "Other"
