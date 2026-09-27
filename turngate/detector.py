"""End-of-turn detectors.

A detector answers one question: given the conversation so far and the
caller's candidate text, how likely is it that the caller has finished?

Plug in a semantic turn-detection model (e.g. a small transformer served
over HTTP, LiveKit's turn detector, Pipecat's smart-turn) by implementing
``predict``. ``HeuristicDetector`` is a dependency-free baseline that is
good enough for tests and demos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    content: str


class EndOfTurnDetector(Protocol):
    async def predict(self, messages: Sequence[Message]) -> Optional[float]:
        """Probability in [0, 1] that the last user message ends the turn.

        Return ``None`` when the detector cannot score the input; the gate
        then applies its default hold.
        """
        ...


_TRAILING_CONTINUATIONS = {
    # words people end on when they are *not* done
    "and", "or", "but", "so", "because", "the", "a", "an", "to", "for", "of",
    "with", "my", "is", "was", "um", "uh", "hmm", "like", "then", "from",
    "aur", "ki", "ka", "ke", "toh", "matlab",
}
_COMPLETE_ENDINGS = re.compile(r"[.!?।]\s*$")
_DIGIT_RUN = re.compile(r"\d[\d\s-]*$")


class HeuristicDetector:
    """Lexical baseline: punctuation, trailing continuation words, digit runs.

    Deliberately simple. Its job in this repo is to make the gate testable
    without a model, and to show how much a real semantic model buys you
    (see ``bench/``).
    """

    async def predict(self, messages: Sequence[Message]) -> Optional[float]:
        if not messages or messages[-1].role != "user":
            return None
        text = messages[-1].content.strip()
        if not text:
            return None
        last_word = re.sub(r"[^\w]", "", text.split()[-1].lower())
        score = 0.25
        if _COMPLETE_ENDINGS.search(text):
            score += 0.45
        if last_word in _TRAILING_CONTINUATIONS:
            score -= 0.22
        if _DIGIT_RUN.search(text):
            # "my PNR is 4 5 2" — people read numbers in chunks
            score -= 0.15
        prev = next((m for m in reversed(messages[:-1]) if m.role == "assistant"), None)
        if prev and prev.content.strip().endswith("?") and len(text.split()) <= 3:
            score += 0.15  # short answer to a direct question
        return max(0.0, min(1.0, score))
