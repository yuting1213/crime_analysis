"""
所有代理人的抽象基底類別
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentReport:
    """代理人產出的標準報告格式"""
    agent_name: str
    crime_category: str             # 預測的犯罪類別
    confidence: float               # 信心分數 [0, 1]
    evidence: List[Dict[str, Any]]  # 支持論點的證據列表
    reasoning: str                  # 思維鏈推理過程（CoT）
    frame_references: List[int]     # 對應的影像幀編號
    conflict_flags: List[str] = field(default_factory=list)  # Reflector 標記的衝突點
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "crime_category": self.crime_category,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "frame_references": self.frame_references,
            "conflict_flags": self.conflict_flags,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    所有 Local Solver 代理人的基底類別
    遵循：獨立觀測 → 交叉質詢 → 共識收斂 的辯論協議
    """

    def __init__(self, name: str, model_name: str):
        self.name = name
        self.model_name = model_name
        self._position: Optional[AgentReport] = None   # 當前立場
        self._debate_history: List[Dict] = []           # 辯論歷史
        self._legal_framework: Optional[Dict] = None   # Step 2.5 注入的初步法律框架

    @abstractmethod
    def analyze(self, frames: List, video_metadata: Dict) -> AgentReport:
        """
        第一階段：獨立觀測
        在不看其他代理人意見的前提下，針對影像產出初步分析。
        """
        raise NotImplementedError

    @abstractmethod
    def refine(self, other_reports: List[AgentReport]) -> AgentReport:
        """
        第三階段：共識收斂
        根據其他代理人的質疑更新自己的立場。
        """
        raise NotImplementedError

    def get_current_position(self) -> Optional[AgentReport]:
        return self._position

    def add_to_history(self, round_num: int, report: AgentReport):
        self._debate_history.append({
            "round": round_num,
            "report": report.to_dict()
        })

    def set_legal_framework(self, framework: Dict) -> None:
        """
        Step 2.5 注入初步法律框架，供 Step 3 refine() 階段進行針對性視覺佐證補充。
        framework 格式：
        {
            "candidate_articles": ["第277條", "第271條"],
            "key_elements_to_verify": ["傷害結果", "故意"],
            "primary_category": "Assault",
            "candidate_categories": ["Assault", "Fighting"],
        }
        """
        self._legal_framework = framework

    def reset(self):
        """每次新影片前重置狀態"""
        self._position = None
        self._debate_history = []
        self._legal_framework = None

    def _build_system_prompt(self) -> str:
        """Cold-start 角色定義"""
        return (
            f"你是一位專業的刑事偵查{self.name}。"
            "請根據影像證據，以嚴謹的邏輯推理分析犯罪行為，"
            "並生成符合司法標準的鑑識報告。"
            "Let's think step by step."  # CoT 觸發詞
        )
