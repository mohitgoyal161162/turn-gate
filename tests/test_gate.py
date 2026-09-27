import asyncio
import re

from turngate import GateConfig, Message, TurnGate


class FixedDetector:
    def __init__(self, prob=0.5, delay=0.0, fail=False):
        self.prob, self.delay, self.fail, self.calls = prob, delay, fail, []

    async def predict(self, messages):
        self.calls.append([m.content for m in messages])
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("detector down")
        return self.prob(messages) if callable(self.prob) else self.prob


HISTORY = [Message("assistant", "How can I help you today?")]
FAST = dict(short_hold_ms=40, long_hold_ms=90, interim_grace_ms=5, predict_timeout_ms=100)


def make(prob=0.5, history=HISTORY, **cfg):
    commits, events = [], []

    async def commit(text):
        commits.append(text)

    async def hist():
        return history

    det = FixedDetector(prob) if not isinstance(prob, FixedDetector) else prob
    gate = TurnGate(detector=det, commit=commit, history=hist,
                    config=GateConfig(**{**FAST, **cfg}),
                    on_event=lambda n, d: events.append((n, d)))
    return gate, commits, events, det


def names(events):
    return [n for n, _ in events]


def run(coro):
    return asyncio.run(coro)


def test_high_probability_commits_immediately():
    async def go():
        gate, commits, events, _ = make(prob=0.9)
        await gate.on_speech_ended("I want to book a flight to Delhi.")
        assert commits == ["I want to book a flight to Delhi."]
        assert "HOLD" not in names(events)
    run(go())


def test_mid_probability_uses_short_hold():
    async def go():
        gate, commits, events, _ = make(prob=0.1)
        await gate.on_speech_ended("I want to book")
        assert commits == [] and gate.holding
        hold = dict(events)["HOLD"]
        assert hold["hold_ms"] == 40
        await asyncio.sleep(0.08)
        assert commits == ["I want to book"]
    run(go())


def test_low_probability_uses_long_hold():
    async def go():
        gate, commits, events, _ = make(prob=0.01)
        await gate.on_speech_ended("my PNR is four five")
        assert dict(events)["HOLD"]["hold_ms"] == 90
        await asyncio.sleep(0.05)
        assert commits == []
        await asyncio.sleep(0.07)
        assert commits == ["my PNR is four five"]
    run(go())


def test_new_fragment_supersedes_hold_and_accumulates():
    async def go():
        gate, commits, _, det = make(prob=lambda m: 0.9 if "Delhi" in m[-1].content else 0.01)
        await gate.on_speech_ended("I want to book")
        await gate.on_speech_ended("a flight to Delhi")
        assert commits == ["I want to book a flight to Delhi"]
        assert det.calls[-1][-1] == "I want to book a flight to Delhi"
        await asyncio.sleep(0.12)
        assert commits == ["I want to book a flight to Delhi"]  # old hold never fires
    run(go())


def test_speech_restart_cancels_hold():
    async def go():
        gate, commits, events, _ = make(prob=0.1)
        await gate.on_speech_ended("I want to book")
        gate.on_speech_started()
        await asyncio.sleep(0.08)
        assert commits == []
        assert "HOLD_CANCELLED_BY_SPEECH_RESTART" in names(events)
        await gate.on_speech_ended("a flight")
        await asyncio.sleep(0.08)
        assert commits == ["I want to book a flight"]
    run(go())


def test_interim_growth_cancels_but_echo_does_not():
    async def go():
        gate, commits, events, _ = make(prob=0.1)
        await gate.on_speech_ended("I want to book")
        await asyncio.sleep(0.01)
        gate.on_interim("I want to book")          # replace-style interim echo
        assert gate.holding
        gate.on_interim("I want to book a flight")  # real growth
        assert not gate.holding
        assert "HOLD_CANCELLED_BY_INTERIM" in names(events)
        await asyncio.sleep(0.08)
        assert commits == []
    run(go())


def test_interim_inside_grace_window_is_ignored():
    async def go():
        gate, commits, _, _ = make(prob=0.1, interim_grace_ms=500, max_rearms=0)
        await gate.on_speech_ended("I want to book")
        gate.on_interim("I want to book a flight")
        assert gate.holding
        await asyncio.sleep(0.08)
        assert commits == ["I want to book"]
    run(go())


def test_speech_during_prediction_drops_stale_decision():
    async def go():
        det = FixedDetector(prob=0.9, delay=0.03)
        gate, commits, _, _ = make(prob=det)
        task = asyncio.create_task(gate.on_speech_ended("I want to"))
        await asyncio.sleep(0.01)
        gate.on_speech_started()
        await task
        assert commits == []
    run(go())


def test_hold_rearms_while_caller_is_speaking_then_gives_up():
    async def go():
        gate, commits, events, _ = make(prob=0.1, max_rearms=2)
        await gate.on_speech_ended("so the thing is")
        gate._user_speaking = True  # VAD says speaking, provider never sends a final
        await asyncio.sleep(0.2)
        assert names(events).count("HOLD_REARMED_USER_SPEAKING") == 2
        assert commits == ["so the thing is"]
    run(go())


def test_single_word_is_unscored_and_commits_by_default():
    async def go():
        gate, commits, events, det = make(prob=0.01)
        await gate.on_speech_ended("yes")
        assert commits == ["yes"] and det.calls == []
        assert dict(events)["PREDICT_SKIPPED"]["reason"] == "SINGLE_WORD"
    run(go())


def test_default_hold_applies_to_unscored_turns():
    async def go():
        gate, commits, events, _ = make(prob=0.01, default_hold_ms=50)
        await gate.on_speech_ended("five")
        assert commits == [] and dict(events)["HOLD"]["reason"] == "SINGLE_WORD"
        await gate.on_speech_ended("six seven")  # digits read in chunks
        assert dict(events)["HOLD"]["hold_ms"] == 90
        await asyncio.sleep(0.12)
        assert commits == ["five six seven"]
    run(go())


def test_no_assistant_context_is_unscored():
    async def go():
        gate, commits, events, det = make(prob=0.01, history=[])
        await gate.on_speech_ended("hello is this support")
        assert commits == ["hello is this support"] and det.calls == []
        assert dict(events)["PREDICT_SKIPPED"]["reason"] == "INSUFFICIENT_CONTEXT"
    run(go())


def test_unsupported_script_is_unscored():
    async def go():
        tamil = re.compile(r"[஀-௿]")
        gate, commits, events, det = make(prob=0.01, unsupported_script=tamil)
        await gate.on_speech_ended("வணக்கம் எனக்கு உதவி")
        assert commits and det.calls == []
        assert dict(events)["PREDICT_SKIPPED"]["reason"] == "UNSUPPORTED_SCRIPT"
    run(go())


def test_detector_failure_and_timeout_never_block_the_caller():
    async def go():
        gate, commits, _, _ = make(prob=FixedDetector(fail=True))
        await gate.on_speech_ended("I need help with a refund")
        assert commits == ["I need help with a refund"]
        gate2, commits2, _, _ = make(prob=FixedDetector(prob=0.01, delay=1.0), predict_timeout_ms=30)
        await gate2.on_speech_ended("I need help with a refund")
        assert commits2 == ["I need help with a refund"]
    run(go())


def test_hold_override_for_a_barged_in_turn():
    async def go():
        gate, _, events, _ = make(prob=0.1)
        gate.hold_override = lambda: (70, 140)
        await gate.on_speech_ended("wait I meant")
        assert dict(events)["HOLD"]["hold_ms"] == 70
        gate.reset()
    run(go())


def test_reset_discards_accumulated_text():
    async def go():
        gate, commits, _, _ = make(prob=0.1)
        await gate.on_speech_ended("half a sentence")
        gate.reset()
        await asyncio.sleep(0.08)
        assert commits == []
        gate.detector.prob = 0.9
        await gate.on_speech_ended("a new turn entirely")
        assert commits == ["a new turn entirely"]
    run(go())
