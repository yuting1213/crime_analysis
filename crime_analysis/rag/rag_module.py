"""
Shared RAG query module for crime analysis agents and Planner.

Wraps HierarchicalRAG and provides a simple interface for legal element
coverage computation, article lookup, and document retrieval.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Legal elements required to establish each crime type under Taiwan Criminal Code
LEGAL_ELEMENTS = {
    "Assault":       ["故意", "傷害行為", "傷害結果", "因果關係"],
    "Fighting":      ["互毆事實", "傷害故意", "雙方積極攻擊"],
    "Shooting":      ["使用槍械或凶器", "殺傷意圖", "危害公共安全"],
    "Robbery":       ["不法所有意圖", "強暴或脅迫手段", "取得財物", "因果關係"],
    "Stealing":      ["不法所有意圖", "竊取行為", "他人動產", "秘密竊取"],
    "Shoplifting":   ["不法所有意圖", "竊取行為", "他人動產"],
    "Burglary":      ["侵入建築物", "不法所有意圖", "竊取行為"],
    "Vandalism":     ["故意", "毀損行為", "他人財物", "毀損結果"],
    "Arson":         ["故意", "放火行為", "燒燬建築物或財物"],
    "Explosion":     ["故意或過失", "爆炸行為", "公共危險"],
    "RoadAccidents": ["過失", "違規行為", "傷亡結果", "因果關係"],
    "Abuse":         ["故意", "傷害或遺棄行為", "被害人身分（家庭成員）"],
    "Arrest":        ["強制行為", "妨害自由", "被害人意願"],
}

# Relevant statutory articles for each crime type
GROUP_LEGAL_CONTEXT = {
    "Fighting":      ["刑法第277條傷害罪", "刑法第281條加重傷害罪"],
    "Assault":       ["刑法第277條傷害罪", "刑法第278條重傷罪"],
    "Shooting":      ["刑法第271條殺人罪", "刑法第185-1條公共危險罪"],
    "Robbery":       ["刑法第328條強盜罪", "刑法第330條加重強盜罪"],
    "Abuse":         ["刑法第277條傷害罪", "家庭暴力防治法第2條"],
    "Arrest":        ["刑法第304條強制罪", "刑法第302條妨害自由罪"],
    "Stealing":      ["刑法第320條竊盜罪", "刑法第321條加重竊盜罪"],
    "Shoplifting":   ["刑法第320條竊盜罪"],
    "Burglary":      ["刑法第321條加重竊盜罪（侵入建築物）"],
    "Vandalism":     ["刑法第354條毀損罪"],
    "Arson":         ["刑法第173條放火罪", "刑法第174條放火建築物罪"],
    "Explosion":     ["刑法第185-1條公共危險罪", "刑法第184條爆炸罪"],
    "RoadAccidents": ["刑法第276條過失致死罪", "刑法第185-3條不能安全駕駛罪"],
}


class RAGModule:
    """
    Shared RAG query module wrapping HierarchicalRAG.

    Provides retrieval, HyDE document generation, legal element coverage
    scoring, and article lookup for all 13 UCF-Crime categories.
    """

    def __init__(self, rag_system=None):
        """
        Parameters
        ----------
        rag_system : HierarchicalRAG, optional
            An already-initialised HierarchicalRAG instance. When None the
            module operates in a degraded mode that returns empty results.
        """
        self._rag = rag_system
        if self._rag is None:
            logger.warning(
                "RAGModule initialised without a rag_system. "
                "All retrieval calls will return empty results."
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, text: str, query_type: str = "semantic") -> Dict[str, List]:
        """
        Retrieve relevant documents from the RAG system.

        Parameters
        ----------
        text : str
            Query text.
        query_type : str
            "semantic" (default) uses dense retrieval;
            "keyword" hints the underlying system to use BM25 mode.

        Returns
        -------
        dict
            Keys "laws" and "judgments", each a list of retrieved strings.
            Returns empty lists when no RAG system is available.
        """
        empty: Dict[str, List] = {"laws": [], "judgments": [], "articles": []}

        if self._rag is None:
            return empty

        try:
            if query_type == "keyword":
                results = self._rag.query(text, mode="bm25")
            else:
                results = self._rag.query(text)

            # Normalise: ensure both expected keys are present
            if isinstance(results, dict):
                return {
                    "laws":      results.get("laws", []),
                    "judgments": results.get("judgments", []),
                    "articles":  results.get("articles", []),
                }
            # Fallback if h_rag returns a plain list
            return {"laws": results if isinstance(results, list) else [],
                    "judgments": [],
                    "articles": []}

        except Exception as exc:
            logger.error("RAGModule.query failed: %s", exc, exc_info=True)
            return empty

    # ------------------------------------------------------------------
    # HyDE helper
    # ------------------------------------------------------------------

    def generate_hypothetical_doc(self, text: str, crime_type: str) -> str:
        """
        Generate a hypothetical document for HyDE (Hypothetical Document
        Embeddings) retrieval.

        Constructs a short hypothetical legal narrative that combines the
        input text with known legal elements for the given crime type.  The
        result can be embedded and used as a query vector against the RAG
        index instead of the raw query.

        Parameters
        ----------
        text : str
            Original query or report excerpt.
        crime_type : str
            One of the 13 UCF-Crime category strings.

        Returns
        -------
        str
            A hypothetical document string for embedding.
        """
        elements = LEGAL_ELEMENTS.get(crime_type, [])
        articles = GROUP_LEGAL_CONTEXT.get(crime_type, [])

        elements_str = "、".join(elements) if elements else "相關犯罪構成要件"
        articles_str = "、".join(articles) if articles else "相關法條"

        hypothetical = (
            f"本案涉及{crime_type}類型犯罪。{text} "
            f"依據{articles_str}，構成要件包括：{elements_str}。"
            f"依現行刑事法律，行為人須具備上述要件方構成本罪。"
        )
        return hypothetical

    # ------------------------------------------------------------------
    # Legal coverage scoring
    # ------------------------------------------------------------------

    def compute_rlegal(self, crime_type: str, report_text: str) -> float:
        """
        Compute a coarse legal-element coverage score (Rlegal).

        Counts how many of the required legal elements for *crime_type*
        appear as substrings in *report_text* and returns the ratio.

        Parameters
        ----------
        crime_type : str
            One of the 13 UCF-Crime category strings.
        report_text : str
            The generated or reference report text to evaluate.

        Returns
        -------
        float
            Fraction of legal elements present in report_text, in [0.0, 1.0].
            Returns 0.0 if crime_type is not recognised.
        """
        elements = LEGAL_ELEMENTS.get(crime_type)
        if not elements:
            logger.warning(
                "compute_rlegal: unknown crime_type '%s'. Returning 0.0.",
                crime_type,
            )
            return 0.0

        covered = sum(1 for el in elements if el in report_text)
        return covered / len(elements)

    # ------------------------------------------------------------------
    # Article & element lookup
    # ------------------------------------------------------------------

    def get_candidate_articles(self, crime_type: str) -> List[str]:
        """
        Return the list of relevant statutory articles for *crime_type*.

        Parameters
        ----------
        crime_type : str
            One of the 13 UCF-Crime category strings.

        Returns
        -------
        list of str
            Article strings, or an empty list if crime_type is not found.
        """
        articles = GROUP_LEGAL_CONTEXT.get(crime_type, [])
        if not articles:
            logger.warning(
                "get_candidate_articles: no articles found for crime_type '%s'.",
                crime_type,
            )
        return articles

    def get_legal_elements(self, crime_type: str) -> List[str]:
        """
        Return the list of legal elements required to establish *crime_type*.

        Parameters
        ----------
        crime_type : str
            One of the 13 UCF-Crime category strings.

        Returns
        -------
        list of str
            Legal element strings, or an empty list if crime_type is not found.
        """
        return LEGAL_ELEMENTS.get(crime_type, [])
