"""SPAR Debate Module."""

from .state import DebateState
from .action import DebateAction, format_plan_prompt, format_utterance_prompt
from .judge import (
    JudgeInterface,
    JudgeEvaluation,
    OpenAIJudge,
    LocalLLMJudge,
    BatchJudge,
    create_judge,
)
from .agents import (
    BaseDebateAgent,
    LLMDebateAgent,
    FrozenAgent,
    TrainableAgent,
    create_agents,
)
from .environment import (
    DebateResult,
    DebateEnvironment,
    create_environment,
)
from .dataset_agent import DatasetOpponentAgent

__all__ = [
    # State and Action
    "DebateState",
    "DebateAction",
    "format_plan_prompt",
    "format_utterance_prompt",
    # Judge
    "JudgeInterface",
    "JudgeEvaluation",
    "OpenAIJudge",
    "LocalLLMJudge",
    "BatchJudge",
    "create_judge",
    # Agents
    "BaseDebateAgent",
    "LLMDebateAgent",
    "FrozenAgent",
    "TrainableAgent",
    "create_agents",
    "DatasetOpponentAgent",
    # Environment
    "DebateResult",
    "DebateEnvironment",
    "create_environment",
]
