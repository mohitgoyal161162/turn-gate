<div align="center">

# turn-gate

**Know when the caller is actually done talking.**

A semantic end-of-turn gate for real-time voice agents: score every STT endpoint,
hold when unsure, cancel the moment the caller keeps talking, commit exactly once.

![python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen)
![tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## The problem

Your STT provider fires an endpoint every time the audio goes quiet. Callers go quiet all the time
without being done:

```text
caller:  "my booking ID is 4 5 2 …"            ← endpoint fires here
agent:   "Sorry, I couldn't find booking 452."  ← agent talks over the caller
caller:  "… 7 8 1"
```

The two naive fixes both hurt:

| Policy | What goes wrong |
|---|---|
| Commit on every endpoint | Agent interrupts mid-sentence, answers half a question, then answers the rest again |
| Fixed hold (wait N ms) | Every turn pays N ms, including "yes", and short holds barely help |

`turn-gate` scores each candidate turn with an end-of-turn model and spends latency **only on
turns that look unfinished**.

## How it works

```mermaid
flowchart LR
    STT[STT endpoint<br/>fragment] --> ACC[accumulate<br/>fragments]
    ACC --> S{scorable?}
    S -- "single word / no context /<br/>unsupported script / model down" --> D[default hold<br/>0 = commit now]
    S -- yes --> M[end-of-turn model<br/>p = P(done)]
    M -- "p ≥ high" --> C((commit))
    M -- "mid ≤ p < high" --> SH[short hold]
    M -- "p < mid" --> LH[long hold]
    SH --> C
    LH --> C
    D --> C
    X[speech restart<br/>interim growth<br/>new fragment] -. cancels .-> SH
    X -. cancels .-> LH
    X -. cancels .-> D
```

A hold is **not** a sleep. It is a cancellable timer that three provider-agnostic signals kill:

1. **Speech restart**: VAD or the provider says the caller started speaking again.
2. **Interim growth**: interim text grows past what the hold snapshotted. Works for
   *append-style* and *replace-style* interims, and a grace window ignores the provider
   echoing back the fragment we already have.
3. **New fragment**: the provider split one utterance into two endpoints. The fragments are
   **accumulated** and re-scored as one turn, so `"I want to book" + "a flight to Delhi"` is
   committed once, as a whole sentence.

### The race nobody tests

Scoring is async: you read history, call a model, get a probability back 80–200 ms later. If the
caller resumed speaking during that await, the answer is stale. Every decision in the gate carries a
**version**; any event that means "the caller is still talking" bumps it, and a stale decision can
neither commit nor arm a hold. See `test_speech_during_prediction_drops_stale_decision`.

### Holds that don't hang

If the hold expires while VAD still says the caller is speaking, the hold re-arms instead of
committing, up to `max_rearms`. An STT stream that never sends a final can't keep the turn open
forever.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
python bench/tradeoff.py
```

```python
from turngate import TurnGate, GateConfig, HeuristicDetector

gate = TurnGate(
    detector=HeuristicDetector(),          # or your semantic model, see below
    commit=run_llm_turn,                   # async (text) -> None
    history=load_conversation,             # async () -> [Message]
    config=GateConfig(short_hold_ms=800, long_hold_ms=1500, default_hold_ms=0),
    on_event=lambda name, data: log.info(name, **data),
)

# wire your STT callbacks
stt.on_endpoint(lambda text: gate.on_speech_ended(text))
stt.on_interim(gate.on_interim)
vad.on_speech_start(gate.on_speech_started)
agent.on_reply_committed(gate.reset)
```

Bring your own model by implementing one method:

```python
class MyDetector:
    async def predict(self, messages) -> float | None:   # None = "can't score"
        return await my_model.eou_probability(messages)
```

## Benchmark

`bench/tradeoff.py` replays a synthetic corpus of 4,000 caller turns (templated utterances,
seeded log-normal pauses; digit runs and continuation words pause longer, STT text has no
punctuation). *Premature* = the agent would have started talking before the caller finished.
*Added latency* = hold on the caller's real final fragment.

| Policy | Premature | Added latency |
|---|---:|---:|
| Commit on endpoint | 59.2% | 0 ms |
| Fixed hold 800 ms | 33.6% | 800 ms |
| Fixed hold 1500 ms | 9.3% | 1500 ms |
| Fixed hold 3000 ms | 0.3% | 3000 ms |
| **Gate, heuristic detector, 0.8 / 1.5 s** | **23.3%** | **544 ms** |
| **Gate, heuristic detector, 1.5 / 3.0 s** | **10.5%** | **1089 ms** |

Even a lexical heuristic gets **fixed-1.5 s safety at ~27% less latency**. A real semantic
end-of-turn model moves the whole curve down-left. The corpus is synthetic: read the shape, not
the digits.

## Design notes from running this in production

These are lessons from running a gate like this on live multilingual phone traffic at scale:

- **Short holds are nearly useless.** An 800 ms hold caught only a handful of percent of real
  continuations; 1.5 s caught roughly a third. If you hold, hold for long enough to matter,
  and only when the model is unsure.
- **Models are miscalibrated per language.** The same threshold was over-confident on one language
  (hesitations scored as "done") and under-confident on another (finished sentences held).
  Per-language thresholds beat one global number.
- **Know what your model can't score.** Single words, the very first turn (no assistant context
  yet; seed it with the greeting), scripts the model never saw, and model timeouts all need an
  explicit, configurable path. They shouldn't silently fall through to "commit".
- **Digits are the worst case.** People read IDs and phone numbers in chunks, and some STT smart
  formatting never endpoints a lone numeral at all. Test with digit runs.
- **After a barge-in, raise the holds.** A caller who just interrupted is mid-thought far more
  often. `hold_override` exists for exactly this.
- **Log every decision with a reason.** `HOLD`, `HOLD_TIMEOUT`, `HOLD_CANCELLED_BY_INTERIM`,
  `PREDICT_SKIPPED{reason}` are how you audit premature commits after the fact.

## Integrations

- **Pipecat**: wrap the gate in a `FrameProcessor` between the STT service and the LLM context
  aggregator: `TranscriptionFrame` → `on_speech_ended`, `InterimTranscriptionFrame` →
  `on_interim`, `UserStartedSpeakingFrame` → `on_speech_started`; `commit` pushes the aggregated
  user message downstream.
- **LiveKit Agents**: use as the turn-detection hook: feed final/interim transcripts and VAD
  events, and let `commit` trigger the reply.

## Project layout

```text
turngate/gate.py       the gate: scoring, holds, cancellation, versioning
turngate/detector.py   detector protocol + dependency-free heuristic baseline
tests/test_gate.py     16 behavioural tests (holds, cancels, races, re-arm, fallbacks)
bench/tradeoff.py      premature-vs-latency benchmark
```

## License

MIT
