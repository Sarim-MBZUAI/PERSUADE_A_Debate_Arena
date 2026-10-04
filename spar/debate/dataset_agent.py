"""
Dataset Opponent Agent

Replays stored arena debate responses as opponent actions,
enabling hybrid training where the trainable agent generates live
while the opponent uses pre-recorded expert responses.
"""

import random
from typing import Optional, List, Dict, TYPE_CHECKING

from .agents import BaseDebateAgent
from .state import DebateState
from .action import DebateAction
from ..config import DebateConfig
from ..utils.text import pseudo_plan_from_argument

if TYPE_CHECKING:
    from ..data.arena_loader import ArenaDataLoader


class DatasetOpponentAgent(BaseDebateAgent):
    """
    Debate agent that replays stored arena responses.

    For each (topic, turn), looks up available arguments from the arena
    dataset and returns one as a DebateAction. Falls back to a simple
    placeholder if no matching response is found.
    """

    def __init__(
        self,
        position: str,
        arena: "ArenaDataLoader",
        config: DebateConfig,
        model_filter: Optional[str] = None,
    ):
        super().__init__(position, config)
        self.arena = arena
        self.model_filter = model_filter

        # Cache: topic -> {turn -> [arguments]}
        self._response_cache: Dict[str, Dict[int, List[str]]] = {}

    def _ensure_cached(self, topic: str) -> None:
        """Ensure responses for this topic are cached."""
        if topic not in self._response_cache:
            self._response_cache[topic] = self.arena.get_opponent_responses(
                topic=topic,
                position=self.position,
                model_filter=self.model_filter,
                split="train",
            )

    def act(self, state: DebateState) -> DebateAction:
        """Return a stored arena argument for the current state."""
        topic = state.topic
        turn = state.turn

        self._ensure_cached(topic)
        responses = self._response_cache.get(topic, {})
        turn_responses = responses.get(turn, [])

        if turn_responses:
            argument = random.choice(turn_responses)
        else:
            # Fallback: use any available turn's response, or placeholder
            all_responses = [r for rs in responses.values() for r in rs]
            if all_responses:
                argument = random.choice(all_responses)
            else:
                argument = (
                    f"I maintain my position {self.position} on this topic. "
                    f"The evidence clearly supports my side of the argument."
                )

        plan = pseudo_plan_from_argument(argument)

        return DebateAction.create_with_separate_generation(
            plan=plan,
            utterance=argument,
            position=self.position,
            turn=turn,
        )

    def generate_plan(
        self,
        state: DebateState,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate plan (extract from stored argument)."""
        action = self.act(state)
        return action.plan

    def generate_utterance(
        self,
        state: DebateState,
        plan: str,
    ) -> str:
        """Generate utterance (returns stored argument)."""
        action = self.act(state)
        return action.utterance

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._response_cache.clear()
