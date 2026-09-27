"""Premature-commit vs added-latency trade-off, on a synthetic corpus.

Each synthetic caller turn is a list of fragments separated by pauses (the
gaps where an STT provider would fire an endpoint). A policy looks at each
fragment and picks a hold. If the hold is shorter than the pause that follows
a non-final fragment, the agent would have started talking over the caller:
a *premature commit*. On the final fragment, the hold is pure *added latency*.

The corpus is synthetic (templates + seeded random pauses), so the absolute
numbers are illustrative. The shape of the curve is the point: fixed holds
buy safety linearly with latency; a scored gate buys most of the safety for
a fraction of the latency.

    python bench/tradeoff.py
"""

from __future__ import annotations

import asyncio
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turngate import HeuristicDetector, Message  # noqa: E402

QUESTIONS = [
    "How can I help you today?",
    "Could you tell me your booking ID?",
    "Which date would you like to travel?",
    "Is there anything else I can help with?",
]

TURNS = [
    ["I want to book", "a flight to Delhi."],
    ["My PNR is 4 5 2", "7 8 1."],
    ["Yes."],
    ["So the thing is", "I cancelled yesterday and", "the refund has not come."],
    ["Next Friday."],
    ["No, that's all, thank you."],
    ["I was charged twice for", "the same ticket."],
    ["Can you change my seat?"],
    ["Um", "I think the booking ID is", "IX 9 9 3 1."],
    ["The flight got delayed and", "I missed my connection."],
    ["Okay."],
    ["Actually wait, can you", "check the return date too?"],
]


def pause_after(fragment: str, rng: random.Random) -> float:
    """Seconds of silence after a non-final fragment."""
    last = fragment.split()[-1].lower().strip(".,?")
    if last.isdigit():
        return rng.lognormvariate(-0.2, 0.45)   # reading digits: ~0.8 s
    if last in {"and", "for", "is", "to", "the", "um", "that"}:
        return rng.lognormvariate(-0.1, 0.5)    # thinking: ~0.9 s
    return rng.lognormvariate(-0.5, 0.5)        # breath: ~0.6 s


def build_corpus(n: int, seed: int = 7):
    rng = random.Random(seed)
    corpus = []
    for _ in range(n):
        frags = rng.choice(TURNS)
        # STT without smart formatting: no punctuation, lower case
        plain = [re.sub(r"[.,?!]", "", f).lower() for f in frags]
        corpus.append((rng.choice(QUESTIONS), plain, [pause_after(f, rng) for f in frags[:-1]]))
    return corpus


def fixed(hold_s):
    async def policy(_q, _text, _is_single):
        return hold_s
    return policy


def gate(detector, high=0.3, mid=0.05, short=0.8, long=1.5, default=0.0):
    async def policy(question, text, is_single):
        if is_single:
            return default
        p = await detector.predict([Message("assistant", question), Message("user", text)])
        if p is None:
            return default
        return 0.0 if p >= high else (short if p >= mid else long)
    return policy


async def evaluate(policy, corpus):
    premature = 0
    latencies = []
    for question, frags, pauses in corpus:
        acc = ""
        for i, frag in enumerate(frags):
            acc = f"{acc} {frag}".strip()
            hold = await policy(question, acc, len(acc.split()) == 1)
            if i < len(frags) - 1:
                if hold < pauses[i]:
                    premature += 1
                    break
            else:
                latencies.append(hold)
    n = len(corpus)
    return premature / n, statistics.mean(latencies) if latencies else 0.0


async def main():
    corpus = build_corpus(4000)
    det = HeuristicDetector()
    policies = [
        ("commit on endpoint", fixed(0.0)),
        ("fixed hold 800 ms", fixed(0.8)),
        ("fixed hold 1500 ms", fixed(1.5)),
        ("fixed hold 3000 ms", fixed(3.0)),
        ("gate (heuristic, 0.8/1.5 s)", gate(det)),
        ("gate (heuristic, 1.5/3.0 s)", gate(det, short=1.5, long=3.0)),
    ]
    print(f"{'policy':32} {'premature':>10} {'added latency (final)':>24}")
    print("-" * 68)
    for name, pol in policies:
        rate, lat = await evaluate(pol, corpus)
        print(f"{name:32} {rate:>9.1%} {lat * 1000:>20.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
