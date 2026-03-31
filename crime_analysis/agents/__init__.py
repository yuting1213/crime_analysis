from .base_agent import BaseAgent, AgentReport
from .environment_agent import EnvironmentAgent
from .action_emotion_agent import ActionEmotionAgent
from .reflector import ReflectorAgent
from .planner import PlannerAgent

__all__ = [
    "BaseAgent", "AgentReport",
    "EnvironmentAgent",
    "ActionEmotionAgent",
    "ReflectorAgent", "PlannerAgent",
]


# 舊版 Agent（deprecated，延遲 import 避免拉入不必要的依賴）
def __getattr__(name):
    if name == "ActionAgent":
        from .action_agent import ActionAgent
        return ActionAgent
    if name == "TimeEmotionAgent":
        from .time_emotion_agent import TimeEmotionAgent
        return TimeEmotionAgent
    if name == "SemanticAgent":
        from .semantic_agent import SemanticAgent
        return SemanticAgent
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
