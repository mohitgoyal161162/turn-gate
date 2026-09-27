"""The turn gate: decides *when* a caller's turn is over.

Speech-to-text providers tell you when the audio went quiet ("endpointing").
That is not the same as the caller being done: people pause mid-sentence,
hesitate, read out numbers in chunks. Committing on every endpoint makes the
agent talk over the caller; waiting a fixed second on every endpoint makes it
feel slow.

The gate sits between the STT endpoint event and the LLM:

    STT endpoint ──► TurnGate ──► commit(text) ──► LLM

and for every candidate turn it picks one of:

    p(end of turn) >= high      commit now
    mid <= p < high             short hold, then commit
    p < mid                     long hold, then commit
    not scorable                default hold (0 = commit now)

A hold is not a sleep. It is cancelled the moment the caller keeps talking,
through any of three provider-agnostic signals:

    * a speech-start event            (VAD / provider "speech started")
    * interim text growing            (append-style or replace-style interims)
    * a new endpoint fragment         (the provider split one utterance in two)

Fragments are accumulated, so "I want to book" + "a flight to Delhi" is scored
and committed as one turn. Every decision is versioned: an await that returns
after the caller spoke again can never commit or arm a stale hold.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Sequence

from .detector import EndOfTurnDetector, Message

CommitFn = Callable[[str], Awaitable[None]]
EventFn = Callable[[str, dict], None]
HistoryFn = Callable[[], Awaitable[Sequence[Message]]]


@dataclass
class GateConfig:
    high: float = 0.30
    mid: float = 0.05
    short_hold_ms: int = 800
    long_hold_ms: int = 1500
    # Hold applied when the model cannot score the turn (single word, no
    # assistant context, unsupported script, detector error). 0 keeps the
    # classic behaviour: commit at once.
    default_hold_ms: int = 0
    # Interims that arrive right after a hold starts are usually the tail of
    # the fragment we already have, not new speech.
    interim_grace_ms: int = 200
    # A hold re-arms while the caller is audibly speaking, but never forever:
    # a provider that never sends a final must not keep the turn open.
    max_rearms: int = 8
    predict_timeout_ms: int = 250
    score_single_words: bool = False
    # Scripts the detector was not trained on. Default: none.
    unsupported_script: Optional[re.Pattern] = None
    history_window: int = 6


@dataclass
class _Hold:
    task: asyncio.Task
    started_at: float
    snapshot_len: int
    text: str
    hold_ms: int
    reason: str
    prob: Optional[float]
    rearms: int = 0


@dataclass
class TurnGate:
    detector: EndOfTurnDetector
    commit: CommitFn
    history: HistoryFn = field(default=None)  # type: ignore[assignment]
    config: GateConfig = field(default_factory=GateConfig)
    on_event: EventFn = field(default=lambda name, data: None)
    # Lets the controller raise holds for one turn, e.g. after a barge-in the
    # caller is mid-thought far more often than on a normal turn.
    hold_override: Optional[Callable[[], Optional[tuple[int, int]]]] = None

    def __post_init__(self) -> None:
        if self.history is None:
            async def _empty() -> Sequence[Message]:
                return []
            self.history = _empty
        self._version = 0
        self._accumulated = ""
        self._hold: Optional[_Hold] = None
        self._user_speaking = False

    # ------------------------------------------------------------------ API

    async def on_speech_ended(self, fragment: str) -> None:
        fragment = (fragment or "").strip()
        if not fragment:
            return
        self._user_speaking = False
        self._version += 1
        version = self._version
        self._cancel_hold()

        self._accumulated = f"{self._accumulated} {fragment}".strip()
        candidate = self._accumulated

        if not self.config.score_single_words and len(candidate.split()) == 1:
            await self._unscored("SINGLE_WORD", candidate, version)
            return

        try:
            history = list(await self.history())
        except Exception as exc:  # history store down: still answer the caller
            self._emit("GATE_ERROR", error=repr(exc))
            history = []
        if version != self._version:
            return

        messages = history[-self.config.history_window:] + [Message("user", candidate)]
        if not any(m.role == "assistant" and m.content.strip() for m in messages):
            await self._unscored("INSUFFICIENT_CONTEXT", candidate, version)
            return
        pattern = self.config.unsupported_script
        if pattern is not None and any(pattern.search(m.content) for m in messages):
            await self._unscored("UNSUPPORTED_SCRIPT", candidate, version)
            return

        started = time.monotonic()
        try:
            prob = await asyncio.wait_for(
                self.detector.predict(messages),
                timeout=self.config.predict_timeout_ms / 1000,
            )
        except Exception as exc:
            self._emit("PREDICT_ERROR", error=repr(exc))
            prob = None
        if version != self._version:
            return
        if prob is None:
            await self._unscored("PREDICT_UNAVAILABLE", candidate, version)
            return

        self._emit("PREDICT", prob=prob, text=candidate,
                   latency_ms=round((time.monotonic() - started) * 1000, 1))
        if prob >= self.config.high:
            await self._commit_now(candidate, reason="MODEL_HIGH")
            return

        short_ms, long_ms = self.config.short_hold_ms, self.config.long_hold_ms
        if self.hold_override:
            override = self.hold_override()
            if override:
                short_ms, long_ms = override
        hold_ms = short_ms if prob >= self.config.mid else long_ms
        self._start_hold(candidate, hold_ms, "MODEL", prob, version)

    def on_interim(self, text: str) -> None:
        self._user_speaking = True
        hold = self._hold
        if hold is None:
            # Speech resumed while history / prediction was in flight.
            self._version += 1
            return
        if (time.monotonic() - hold.started_at) * 1000 < self.config.interim_grace_ms:
            return
        text = (text or "").strip()
        if len(text) > hold.snapshot_len + 1:
            self._emit("HOLD_CANCELLED_BY_INTERIM", grew_by=len(text) - hold.snapshot_len)
            self._version += 1
            self._cancel_hold()

    def on_speech_started(self) -> None:
        self._user_speaking = True
        self._version += 1
        if self._hold is not None:
            self._emit("HOLD_CANCELLED_BY_SPEECH_RESTART", held_text=self._hold.text)
            self._cancel_hold()

    def reset(self) -> None:
        self._version += 1
        self._cancel_hold()
        self._accumulated = ""
        self._user_speaking = False

    @property
    def holding(self) -> bool:
        return self._hold is not None

    # ------------------------------------------------------------ internals

    async def _unscored(self, reason: str, text: str, version: int) -> None:
        self._emit("PREDICT_SKIPPED", reason=reason, text=text)
        if self.config.default_hold_ms > 0:
            self._start_hold(text, self.config.default_hold_ms, reason, None, version)
        else:
            await self._commit_now(text, reason=reason)

    async def _commit_now(self, text: str, reason: str) -> None:
        self._accumulated = ""
        self._emit("COMMIT", reason=reason, text=text)
        await self.commit(text)

    def _start_hold(self, text: str, hold_ms: int, reason: str,
                    prob: Optional[float], version: int, rearms: int = 0) -> None:
        self._emit("HOLD", reason=reason, prob=prob, hold_ms=hold_ms, rearms=rearms)
        task = asyncio.get_running_loop().create_task(
            self._hold_then_commit(text, hold_ms, reason, prob, version, rearms))
        self._hold = _Hold(task, time.monotonic(), len(text), text, hold_ms, reason, prob, rearms)

    async def _hold_then_commit(self, text: str, hold_ms: int, reason: str,
                                prob: Optional[float], version: int, rearms: int) -> None:
        await asyncio.sleep(max(0, hold_ms) / 1000)
        if version != self._version:
            return
        if self._user_speaking and rearms < self.config.max_rearms:
            self._emit("HOLD_REARMED_USER_SPEAKING", rearms=rearms + 1)
            self._start_hold(text, hold_ms, reason, prob, version, rearms + 1)
            return
        self._hold = None
        self._accumulated = ""
        self._emit("HOLD_TIMEOUT", reason=reason, text=text)
        try:
            await self.commit(text)
        except Exception as exc:
            self._emit("GATE_ERROR", error=repr(exc))

    def _cancel_hold(self) -> None:
        hold, self._hold = self._hold, None
        if hold is not None and not hold.task.done() and hold.task is not asyncio.current_task():
            hold.task.cancel()

    def _emit(self, name: str, **data) -> None:
        try:
            self.on_event(name, data)
        except Exception:
            pass
