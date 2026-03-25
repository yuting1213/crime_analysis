"""
Semantic Agent - 法律語義代理
負責：連結法條、生成初步鑑定報告
工具：H-RAG（BM25 + BGE-M3 + RRF）、LoRA 微調、HyDE 查詢增強
"""
from typing import Any, Dict, List, Optional
import logging

from .base_agent import BaseAgent, AgentReport
from config import cfg

logger = logging.getLogger(__name__)


# 台灣刑法構成要件模板（法律要件覆蓋率 Rlegal 的計算基準）
LEGAL_ELEMENTS: Dict[str, List[str]] = {
    "Assault": ["主觀故意", "傷害行為", "因果關係", "傷害結果", "違法性"],
    "Robbery": ["強暴/脅迫手段", "取財意圖", "他人財物", "不法所有意圖"],
    "Fighting": ["互毆事實", "傷害故意", "雙方積極攻擊行為"],
    "Stealing": ["竊取行為", "他人財物", "不法所有意圖", "秘密竊取"],
    "Shooting": ["使用槍械", "殺傷力", "故意/過失", "危害公共安全"],
    # ... 其他類別
}


class SemanticAgent(BaseAgent):
    """
    法律語義代理
    - 以 H-RAG 檢索相關法條與裁判書
    - 以 LoRA 微調的 Qwen3 進行法律要件比對
    - 生成結構化初步鑑定報告
    """

    def __init__(self, rag_system=None, model_name: str = None):
        super().__init__(
            name="法律語義專家",
            model_name=model_name or cfg.model.base_model
        )
        self.rag = rag_system   # H-RAG 實例（由 pipeline 注入）
        self._action_metadata: Dict = {}   # 由 analyze() 從 video_metadata 注入
        # TODO: 載入 LoRA 微調後的 Qwen3
        # self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    def preliminary_legal_framework(
        self,
        action_report: AgentReport,
        te_report: Optional[AgentReport],
        video_metadata: Dict,
    ) -> Dict:
        """
        Step 2.5：根據 Action + TimeEmotion 的初步結果，確定可能適用的罪名範圍。

        回傳格式：
        {
            "candidate_articles": ["第277條", "第278條"],
            "key_elements_to_verify": ["傷害行為", "傷害結果", "主觀故意"],
            "primary_category": "Assault",
            "candidate_categories": ["Assault", "Fighting"],
        }

        設計邏輯：
        - primary_category  → Action Agent 判定的主要類別（信心最高）
        - candidate_categories → 若 TimeEmotion 給出不同判定，一同納入備選
        - key_elements_to_verify → 合併各備選類別的法律構成要件，去重後回傳
        - candidate_articles → 依 group_legal_context 查表
        """
        primary_cat = action_report.crime_category
        candidates = {primary_cat}

        # 若 TimeEmotion 給出不同且信心不低的判定，加入備選
        if (te_report
                and te_report.crime_category not in ("Normal", "ENVIRONMENTAL_ASSESSMENT")
                and te_report.crime_category != primary_cat
                and te_report.confidence >= 0.4):
            candidates.add(te_report.crime_category)

        # 合併各備選類別的構成要件（去重保序）
        elements_seen: set = set()
        key_elements: List[str] = []
        for cat in candidates:
            for elem in LEGAL_ELEMENTS.get(cat, []):
                if elem not in elements_seen:
                    elements_seen.add(elem)
                    key_elements.append(elem)

        # 候選法條：依 group_legal_context 查表
        group_legal_context = {
            "Fighting":      "刑法第277條傷害罪、第281條加重傷害罪",
            "Assault":       "刑法第277條傷害罪、第278條重傷罪",
            "Shooting":      "刑法第271條殺人罪、第185-1條公共危險罪",
            "Robbery":       "刑法第328條強盜罪、第330條加重強盜罪",
            "Abuse":         "刑法第277條傷害罪、家庭暴力防治法",
            "Arrest":        "刑事訴訟法逮捕程序、刑法第304條強制罪",
            "Stealing":      "刑法第320條竊盜罪、第321條加重竊盜罪",
            "Shoplifting":   "刑法第320條竊盜罪",
            "Burglary":      "刑法第321條加重竊盜罪（侵入建築物）",
            "Vandalism":     "刑法第354條毀損罪",
            "Arson":         "刑法第173條放火罪、第174條放火建築物罪",
            "Explosion":     "刑法第185-1條公共危險罪、第184條爆炸罪",
            "RoadAccidents": "刑法第276條過失致死罪、第185-3條不能安全駕駛罪",
        }
        articles_seen: set = set()
        candidate_articles: List[str] = []
        for cat in candidates:
            for article in group_legal_context.get(cat, "").split("、"):
                if article and article not in articles_seen:
                    articles_seen.add(article)
                    candidate_articles.append(article)

        framework = {
            "candidate_articles":      candidate_articles,
            "key_elements_to_verify":  key_elements,
            "primary_category":        primary_cat,
            "candidate_categories":    sorted(candidates),
        }
        logger.info(
            f"[SemanticAgent] 初步法律框架 | 主罪名={primary_cat} "
            f"| 備選={sorted(candidates)} | 要件={key_elements}"
        )
        return framework

    def analyze(self, frames: List, video_metadata: Dict) -> AgentReport:
        """
        法律語義分析流程（最終定性）：
        1. HyDE：生成假設性法條描述以增強查詢
           - 若已有 Step 2.5 的 legal_framework，用 candidate_articles 縮小查詢範圍
        2. H-RAG：檢索相關法條與裁判書判例
        3. 法律要件比對：依照構成要件逐一核查
        4. 生成初步鑑定報告
        """
        # 注入 Action Agent metadata（action_category / action_confidence 由 Planner 填入）
        self._action_metadata = video_metadata

        # Step 1: 從視覺描述生成初步法律假設（HyDE）
        # 若 Planner 已注入 legal_framework，使用 candidate_articles 聚焦 HyDE 查詢
        visual_description = self._describe_visual_content(frames)
        hyde_query = self._generate_hyde_query(
            visual_description,
            legal_framework=video_metadata.get("legal_framework"),
        )

        # Step 2: H-RAG 檢索
        rag_results = self._retrieve_legal_context(hyde_query)

        # Step 3: 法律要件比對
        legal_analysis = self._analyze_legal_elements(visual_description, rag_results)

        # Step 4: 生成初步鑑定報告
        draft_report = self._generate_forensic_report(
            visual_description, legal_analysis, rag_results
        )

        evidence = [
            {
                "type": "retrieved_laws",
                "articles": rag_results.get("laws", []),
            },
            {
                "type": "retrieved_judgments",
                "cases": rag_results.get("judgments", []),
            },
            {
                "type": "legal_elements_coverage",
                "elements": legal_analysis["covered_elements"],
                "missing": legal_analysis["missing_elements"],
                "rlegal": legal_analysis["rlegal_score"],
            },
        ]

        report = AgentReport(
            agent_name=self.name,
            crime_category=legal_analysis["crime_category"],
            confidence=legal_analysis["confidence"],
            evidence=evidence,
            reasoning=draft_report,
            frame_references=self._extract_key_frames(frames),
            metadata={
                "rlegal": legal_analysis["rlegal_score"],
                "retrieved_article_ids": [
                    r.get("article_id") for r in rag_results.get("laws", [])
                ],
            },
        )
        self._position = report
        return report

    def refine(self, other_reports: List[AgentReport]) -> AgentReport:
        """
        根據 Action、Environment、Time&Emotion 代理人的結論，
        更新法律論述，強化或修正構成要件的涵蓋範圍。
        """
        if self._position is None:
            raise RuntimeError("analyze() 必須先執行")

        # 整合其他代理人的犯罪類別判斷
        category_votes = {}
        for r in other_reports:
            if r.crime_category != "ENVIRONMENTAL_ASSESSMENT":
                cat = r.crime_category
                category_votes[cat] = category_votes.get(cat, 0) + r.confidence

        if not category_votes:
            return self._position

        # 多數決犯罪類別（加權信心分數）
        consensus_category = max(category_votes, key=lambda k: category_votes[k])

        if consensus_category != self._position.crime_category:
            # 重新以共識類別進行法律要件比對
            new_legal_elements = LEGAL_ELEMENTS.get(consensus_category, [])
            rlegal = self._compute_rlegal(new_legal_elements, other_reports)

            self._position = AgentReport(
                agent_name=self.name,
                crime_category=consensus_category,
                confidence=category_votes[consensus_category] / len(other_reports),
                evidence=self._position.evidence,
                reasoning=(
                    f"根據代理人共識更新為 {consensus_category}，"
                    f"法律要件覆蓋率 Rlegal = {rlegal:.2f}。\n"
                    + self._position.reasoning
                ),
                frame_references=self._position.frame_references,
                metadata={**self._position.metadata, "rlegal": rlegal},
            )

        return self._position

    # ── 內部方法 ──────────────────────────────────────────

    def _describe_visual_content(self, frames: List) -> str:
        """
        以 ASK-HINT 結構化格式生成視覺描述（Zou et al. 2025 §3.3）：
        先由 Action Agent 的 metadata 取得 Q* 群組與細粒度類別，
        再用對應群組的 Q* 問題 + class-wise 問題建構結構化描述，
        作為 HyDE 法律查詢的基礎。

        格式：「Abnormal Event → [Group]. [fine-grained action cues]」
              （對應論文 Fig. 5 的 Answer Format）

        TODO: 接入 Qwen2.5-VL-7B-Instruct 做真正的 VLM 推論時，
              以 Q* 問題為 prompt 取得模型輸出，替換此 placeholder。
        """
        # 從 metadata 取得 Action Agent 的 ASK-HINT 結果
        action_category = self._action_metadata.get("action_category", "")
        action_conf     = self._action_metadata.get("action_confidence", 0.0)

        # 匯入 ASK-HINT 對應表（避免循環依賴，直接在此定義 group lookup）
        group_map = {
            "Fighting": "Violence or Harm to People",
            "Assault":  "Violence or Harm to People",
            "Shooting": "Violence or Harm to People",
            "Robbery":  "Violence or Harm to People",
            "Abuse":    "Violence or Harm to People",
            "Arrest":   "Violence or Harm to People",
            "Stealing":     "Crimes Against Property",
            "Shoplifting":  "Crimes Against Property",
            "Burglary":     "Crimes Against Property",
            "Vandalism":    "Crimes Against Property",
            "Arson":        "Public Safety Incidents",
            "Explosion":    "Public Safety Incidents",
            "RoadAccidents": "Public Safety Incidents",
        }
        group_label = group_map.get(action_category, "")

        if action_category and group_label:
            # ASK-HINT 結構化描述（對應 Fig. 5 Answer Format）
            return (
                f"Abnormal Event → {group_label}. "
                f"Visual evidence consistent with {action_category} "
                f"(Action Agent confidence: {action_conf:.2f}). "
                f"Fine-grained action cues observed in the scene."
            )

        # fallback：無 Action Agent 資訊時用通用描述
        return "監視影片中偵測到異常行為，具體行為類型待法律要件核查確認。"

    def _generate_hyde_query(
        self,
        visual_description: str,
        legal_framework: Optional[Dict] = None,
    ) -> str:
        """
        HyDE (Hypothetical Document Embeddings) + ASK-HINT 群組導引：
        讓 LLM 先生成一段假設性的法條描述，再用它來檢索實際法條。
        比直接用影像描述查詢效果更好（H-RAG 的查詢品質關鍵）。

        Step 2.5 改進：若已有 legal_framework，使用 candidate_articles 精準聚焦：
        - 直接以 Step 2.5 確認的候選法條作為 HyDE 查詢範圍
        - 不再依賴 action_category lookup，減少重複計算

        TODO: 呼叫 Qwen2.5-VL 生成假設性法條描述替換 placeholder。
        """
        action_category = self._action_metadata.get("action_category", "")

        # 若已有 Step 2.5 的初步法律框架，直接使用候選法條（更精準）
        if legal_framework and legal_framework.get("candidate_articles"):
            articles_str = "、".join(legal_framework["candidate_articles"][:4])
            elements_str = "、".join(
                legal_framework.get("key_elements_to_verify", [])[:3]
            )
            return (
                f"依中華民國刑法，{visual_description}"
                f"可能涉及{articles_str}。"
                f"重點核查構成要件：{elements_str}。"
            )

        # Fallback：原本的群組導引查詢（無 legal_framework 時使用）
        group_legal_context = {
            "Fighting":      "刑法第277條傷害罪、第281條加重傷害罪",
            "Assault":       "刑法第277條傷害罪、第278條重傷罪",
            "Shooting":      "刑法第271條殺人罪、第185-1條公共危險罪",
            "Robbery":       "刑法第328條強盜罪、第330條加重強盜罪",
            "Abuse":         "刑法第277條傷害罪、家庭暴力防治法",
            "Arrest":        "刑事訴訟法逮捕程序、刑法第304條強制罪",
            "Stealing":      "刑法第320條竊盜罪、第321條加重竊盜罪",
            "Shoplifting":   "刑法第320條竊盜罪",
            "Burglary":      "刑法第321條加重竊盜罪（侵入建築物）",
            "Vandalism":     "刑法第354條毀損罪",
            "Arson":         "刑法第173條放火罪、第174條放火建築物罪",
            "Explosion":     "刑法第185-1條公共危險罪、第184條爆炸罪",
            "RoadAccidents": "刑法第276條過失致死罪、第185-3條不能安全駕駛罪",
        }
        legal_hint = group_legal_context.get(action_category, "刑事不法行為")

        return (
            f"依中華民國刑法，{visual_description}"
            f"可能涉及{legal_hint}。"
            f"構成要件包含：主觀故意或過失、客觀行為、因果關係及違法性。"
        )

    def _retrieve_legal_context(self, query: str) -> Dict[str, List]:
        """
        呼叫 H-RAG 系統進行雙層檢索：
        - Keyword Layer (BM25)：精確條號查詢
        - Data Layer (BGE-M3)：語意查詢
        - RRF 融合結果
        """
        if self.rag is None:
            logger.warning("RAG 系統未初始化，跳過法律檢索")
            return {"laws": [], "judgments": []}

        return self.rag.query(query)

    def _analyze_legal_elements(
        self, description: str, rag_results: Dict
    ) -> Dict[str, Any]:
        """
        逐一核查法律構成要件（Rlegal 計算）：
        Rlegal = 涵蓋要件數 / 標準要件總數
        """
        # TODO: 實作要件核查邏輯
        predicted_category = "Assault"
        elements = LEGAL_ELEMENTS.get(predicted_category, [])
        covered = elements[:3]   # placeholder

        rlegal = len(covered) / len(elements) if elements else 0.0

        return {
            "crime_category": predicted_category,
            "confidence": rlegal,
            "covered_elements": covered,
            "missing_elements": [e for e in elements if e not in covered],
            "rlegal_score": rlegal,
        }

    def _generate_forensic_report(
        self, description: str, analysis: Dict, rag_results: Dict
    ) -> str:
        """
        生成結構化初步鑑定報告：
        [觀察事實] → [法律要件比對] → [引用法條] → [初步結論]
        每個邏輯跳躍點標註對應的影像幀與法律依據。
        """
        laws = rag_results.get("laws", [])
        law_citations = "；".join(
            f"刑法第{l.get('article_id', '?')}條" for l in laws[:3]
        ) or "（待補充法條引用）"

        return (
            f"【觀察事實】{description}\n\n"
            f"【法律要件比對】\n"
            f"犯罪類別：{analysis['crime_category']}\n"
            f"涵蓋要件：{', '.join(analysis['covered_elements'])}\n"
            f"待補充要件：{', '.join(analysis['missing_elements'])}\n\n"
            f"【引用法條】{law_citations}\n\n"
            f"【初步結論】"
            f"依據現有影像證據，本案疑似構成 {analysis['crime_category']}，"
            f"法律要件覆蓋率 Rlegal = {analysis['rlegal_score']:.2f}。"
        )

    def _compute_rlegal(
        self, elements: List[str], other_reports: List[AgentReport]
    ) -> float:
        """TODO: 根據其他代理人提供的證據，重新計算 Rlegal"""
        if not elements:
            return 0.0
        covered = max(1, len(elements) // 2)  # placeholder
        return covered / len(elements)

    def _extract_key_frames(self, frames: List) -> List[int]:
        return list(range(0, len(frames), max(1, len(frames) // 5)))[:5]
