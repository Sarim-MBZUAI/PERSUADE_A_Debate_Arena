"""
Debate Environment

Orchestrates multi-turn debates between FOR and AGAINST agents.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

from .state import DebateState
from .action import DebateAction
from .judge import JudgeInterface, JudgeEvaluation
from .agents import BaseDebateAgent
from ..config import DebateConfig, JudgeConfig


@dataclass
class DebateResult:
    """Complete result of a debate."""

    topic: str
    domain: str
    judge_initial_position: str
    judge_initial_belief: Optional[float]
    judge_final_decision: str
    winner: str
    history: List[Dict[str, Any]]
    for_actions: List[DebateAction]
    against_actions: List[DebateAction]
    intermediate_beliefs: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def belief_shifted(self) -> bool:
        """Whether judge changed position during debate."""
        return self.judge_initial_position != self.judge_final_decision

    @property
    def for_won(self) -> bool:
        return self.winner == "FOR"

    @property
    def against_won(self) -> bool:
        return self.winner == "AGAINST"

    @property
    def num_turns(self) -> int:
        return len(self.for_actions)

    @property
    def total_belief_shift(self) -> float:
        """Total shift in judge belief toward FOR."""
        if not self.intermediate_beliefs:
            return 0.0
        return self.intermediate_beliefs[-1] - (self.judge_initial_belief or 0.5)

    # Property aliases for compatibility
    @property
    def initial_belief(self) -> Optional[float]:
        """Alias for judge_initial_belief."""
        return self.judge_initial_belief

    @property
    def final_belief(self) -> Optional[float]:
        """Final belief (last intermediate or initial if none)."""
        if self.intermediate_beliefs:
            return self.intermediate_beliefs[-1]
        return self.judge_initial_belief

    @property
    def belief_trajectory(self) -> List[float]:
        """Full belief trajectory including initial."""
        trajectory = []
        if self.judge_initial_belief is not None:
            trajectory.append(self.judge_initial_belief)
        trajectory.extend(self.intermediate_beliefs)
        return trajectory

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "topic": self.topic,
            "domain": self.domain,
            "judge_initial_position": self.judge_initial_position,
            "judge_initial_belief": self.judge_initial_belief,
            "judge_final_decision": self.judge_final_decision,
            "winner": self.winner,
            "debate": self.history,
            "intermediate_beliefs": self.intermediate_beliefs,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save debate result to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebateResult":
        """Load from dictionary."""
        return cls(
            topic=data["topic"],
            domain=data.get("domain", ""),
            judge_initial_position=data["judge_initial_position"],
            judge_initial_belief=data.get("judge_initial_belief"),
            judge_final_decision=data["judge_final_decision"],
            winner=data["judge_final_decision"],  # Winner is the final decision
            history=data.get("debate", []),
            for_actions=[],  # Not serialized
            against_actions=[],  # Not serialized
            intermediate_beliefs=data.get("intermediate_beliefs", []),
            metadata=data.get("metadata", {}),
        )


class DebateEnvironment:
    """
    Environment that manages debate execution.

    Orchestrates:
    - FOR and AGAINST agents
    - Multi-turn interaction
    - Judge evaluation
    - Belief tracking
    """

    def __init__(
        self,
        for_agent: BaseDebateAgent,
        against_agent: BaseDebateAgent,
        judge: JudgeInterface,
        config: DebateConfig,
        judge_config: Optional[JudgeConfig] = None,
    ):
        self.for_agent = for_agent
        self.against_agent = against_agent
        self.judge = judge
        self.config = config
        self.judge_config = judge_config

        # Track current state
        self._current_state: Optional[DebateState] = None
        self._for_actions: List[DebateAction] = []
        self._against_actions: List[DebateAction] = []
        self._beliefs: List[float] = []

    def reset(self, topic: str, domain: str = "") -> DebateState:
        """Reset environment for new debate."""
        # Get judge's initial position
        initial_eval = self.judge.get_initial_position(topic)

        # Create initial state
        self._current_state = DebateState(
            topic=topic,
            domain=domain,
            position="FOR",
            turn=1,
            history=(),
            judge_initial_position=initial_eval.decision,
            judge_initial_belief=initial_eval.belief,
        )

        self._for_actions = []
        self._against_actions = []
        self._beliefs = []

        if initial_eval.belief is not None:
            self._beliefs.append(initial_eval.belief)

        return self._current_state

    def step(
        self,
        for_action: Optional[DebateAction] = None,
        against_action: Optional[DebateAction] = None,
        start_with: str = "FOR",
    ) -> Tuple[DebateState, Dict[str, Any]]:
        """
        Execute one full turn (FOR + AGAINST).

        If actions not provided, agents generate them.
        """
        if self._current_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        turn = self._current_state.turn
        info = {}

        if start_with not in ("FOR", "AGAINST"):
            raise ValueError(f"start_with must be 'FOR' or 'AGAINST', got {start_with}")

        if start_with == "FOR":
            # FOR agent acts
            if for_action is None:
                for_action = self.for_agent.act(self._current_state)

            self._for_actions.append(for_action)
            self._current_state = self._current_state.append_entry(
                agent="FOR",
                argument=for_action.utterance,
                plan=for_action.plan,
                turn=turn,
            )

            # AGAINST agent acts
            if against_action is None:
                against_action = self.against_agent.act(self._current_state)

            self._against_actions.append(against_action)
            self._current_state = self._current_state.append_entry(
                agent="AGAINST",
                argument=against_action.utterance,
                plan=against_action.plan,
                turn=turn,
            )
        else:
            # AGAINST agent acts first
            if against_action is None:
                against_action = self.against_agent.act(self._current_state)

            self._against_actions.append(against_action)
            self._current_state = self._current_state.append_entry(
                agent="AGAINST",
                argument=against_action.utterance,
                plan=against_action.plan,
                turn=turn,
            )

            # FOR agent acts second
            if for_action is None:
                for_action = self.for_agent.act(self._current_state)

            self._for_actions.append(for_action)
            self._current_state = self._current_state.append_entry(
                agent="FOR",
                argument=for_action.utterance,
                plan=for_action.plan,
                turn=turn,
            )

        # Query intermediate belief if configured
        if self.judge_config and turn % self.judge_config.belief_query_interval == 0:
            belief_eval = self.judge.get_intermediate_belief(
                self._current_state.topic,
                list(self._current_state.history),
            )
            if belief_eval.belief is not None:
                self._beliefs.append(belief_eval.belief)
                info["belief"] = belief_eval.belief

        # Update turn
        self._current_state = self._current_state.with_turn(turn + 1)

        info["for_action"] = for_action
        info["against_action"] = against_action

        return self._current_state, info

    def run_debate(
        self,
        topic: str,
        domain: str = "",
        num_turns: Optional[int] = None,
        start_with: str = "FOR",
    ) -> DebateResult:
        """Run complete debate and return result."""
        num_turns = num_turns or self.config.num_turns

        state = self.reset(topic, domain)

        for turn in range(1, num_turns + 1):
            state, _ = self.step(start_with=start_with)

        # Get final judgment
        final_eval = self.judge.get_final_decision(
            topic,
            list(state.history),
        )

        return DebateResult(
            topic=topic,
            domain=domain,
            judge_initial_position=state.judge_initial_position,
            judge_initial_belief=state.judge_initial_belief,
            judge_final_decision=final_eval.decision,
            winner=final_eval.decision,
            history=list(state.history),
            for_actions=self._for_actions.copy(),
            against_actions=self._against_actions.copy(),
            intermediate_beliefs=self._beliefs.copy(),
            metadata={
                "num_turns": num_turns,
                "model": self.config.__class__.__name__,
                "start_with": start_with,
            },
        )

    def run_debate_with_plans(
        self,
        topic: str,
        domain: str,
        for_plans: List[str],
        num_turns: Optional[int] = None,
    ) -> DebateResult:
        """
        Run debate with pre-selected plans for FOR agent.

        Useful for GRPO training where we sample plans first.
        """
        num_turns = num_turns or min(len(for_plans), self.config.num_turns)

        state = self.reset(topic, domain)

        for turn_idx in range(num_turns):
            turn = turn_idx + 1

            # FOR agent uses provided plan
            plan = for_plans[turn_idx]
            utterance = self.for_agent.generate_utterance(state, plan)
            for_action = DebateAction.create_with_separate_generation(
                plan=plan,
                utterance=utterance,
                position="FOR",
                turn=turn,
            )

            # AGAINST agent generates normally
            against_action = self.against_agent.act(state.with_position("AGAINST"))

            # Update state
            self._for_actions.append(for_action)
            state = state.append_entry(
                agent="FOR",
                argument=for_action.utterance,
                plan=for_action.plan,
                turn=turn,
            )

            self._against_actions.append(against_action)
            state = state.append_entry(
                agent="AGAINST",
                argument=against_action.utterance,
                plan=against_action.plan,
                turn=turn,
            )

            # Query intermediate belief
            if self.judge_config and turn % self.judge_config.belief_query_interval == 0:
                belief_eval = self.judge.get_intermediate_belief(
                    state.topic,
                    list(state.history),
                )
                if belief_eval.belief is not None:
                    self._beliefs.append(belief_eval.belief)

            self._current_state = state.with_turn(turn + 1)

        # Get final judgment
        final_eval = self.judge.get_final_decision(topic, list(state.history))

        return DebateResult(
            topic=topic,
            domain=domain,
            judge_initial_position=state.judge_initial_position,
            judge_initial_belief=state.judge_initial_belief,
            judge_final_decision=final_eval.decision,
            winner=final_eval.decision,
            history=list(state.history),
            for_actions=self._for_actions.copy(),
            against_actions=self._against_actions.copy(),
            intermediate_beliefs=self._beliefs.copy(),
            metadata={
                "num_turns": num_turns,
                "plans_provided": True,
            },
        )

    def collect_rollouts(
        self,
        topics: List[Tuple[str, str]],  # (topic, domain) pairs
        num_samples_per_topic: int = 1,
        num_turns: Optional[int] = None,
    ) -> List[DebateResult]:
        """
        Collect multiple debate rollouts for training.

        Args:
            topics: List of (topic, domain) tuples
            num_samples_per_topic: Number of debates per topic
            num_turns: Number of turns per debate

        Returns:
            List of DebateResult objects
        """
        results = []

        for topic, domain in topics:
            for _ in range(num_samples_per_topic):
                result = self.run_debate(topic, domain, num_turns)
                results.append(result)

        return results


def create_environment(
    for_agent: BaseDebateAgent,
    against_agent: BaseDebateAgent,
    judge: JudgeInterface,
    config: DebateConfig,
    judge_config: Optional[JudgeConfig] = None,
) -> DebateEnvironment:
    """Factory function to create debate environment."""
    return DebateEnvironment(
        for_agent=for_agent,
        against_agent=against_agent,
        judge=judge,
        config=config,
        judge_config=judge_config,
    )
