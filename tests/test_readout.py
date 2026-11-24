import asyncio
import threading

import pytest

from openjev_server.profile import Profile
from openjev_server.readout import TOKEN_CHARS_MAX, BadQuestion, Readout, choice_confidence, compose_chunks, score_confidence, softmax


def test_confidence_formulas():
    assert choice_confidence([1.0]) == 1.0
    assert choice_confidence([0.25] * 4) == 0.0
    assert abs(choice_confidence([0.7, 0.1, 0.1, 0.1]) - (0.7 - 0.25) / 0.75) < 1e-12
    assert score_confidence([0.0, 0.0, 1.0]) == 1.0
    assert 0.0 <= score_confidence([0.5, 0.0, 0.5]) <= 1.0


def test_compose_chunks_is_a_distribution_and_order_free():
    parts = [([0.7, 0.2, 0.1], 0), ([0.1, 0.6, 0.3], 1)]
    p = compose_chunks(parts, [0.4, 0.6])
    assert abs(sum(p) - 1) < 1e-12 and all(x > 0 for x in p)
    assert abs(p[0] / p[4] - (0.4 * 0.7 / 0.7) / (0.6 * 0.6 / 0.6)) < 1e-12  # winners keep the final ratio


def test_softmax():
    p = softmax([0.0, 0.0])
    assert abs(p[0] - 0.5) < 1e-12


def test_bad_questions(fake_backend):
    ro = Readout(fake_backend, Profile())
    with pytest.raises(BadQuestion):
        asyncio.run(ro.answer_all("s", {"q": {"type": "choice", "instructions": "x", "criteria": {}}}))
    with pytest.raises(BadQuestion):
        asyncio.run(ro.answer_all("s", {"q": {"type": "score", "instructions": "x", "criteria": []}}))
    with pytest.raises(BadQuestion):
        asyncio.run(ro.answer_all("s", {"q": {"type": "essay"}}))


def test_many_options_sum_to_one_and_use_two_stages(fake_backend):
    ro = Readout(fake_backend, Profile())
    ans, _ = asyncio.run(ro.answer_all("s", {"q": {"type": "choice", "instructions": "pick", "criteria": {f"o{i}": None for i in range(130)}}}))
    p = ans["q"]["probabilities"]
    assert len(p) == 130 and abs(sum(p.values()) - 1) < 0.01 and ans["q"]["choice"] in p
    assert len(fake_backend.prompts) == 4  # 3 chunks of <= 52 + 1 over the winners


def test_stagger_runs_first_question_alone(fake_backend):
    ro = Readout(fake_backend, Profile(stagger=True, stagger_min_chars=10))
    ans, toks = asyncio.run(ro.answer_all("x" * 100, {"a": {"type": "noul", "instructions": "yes?"}, "b": {"type": "noul", "instructions": "no?"}}))
    assert set(ans) == {"a", "b"} and toks > 0 and 0 <= ans["a"]["noul"] <= 1


def test_read_twice_counts_tokens_off_the_event_loop(fake_backend, monkeypatch):
    """With a known window, the doubled prompt is counted in a worker thread: a long state must not stall every other request."""
    threads = []
    count = fake_backend.header_tokens
    monkeypatch.setattr(fake_backend, "header_tokens", lambda text: threads.append(threading.get_ident()) or count(text))
    fake_backend.window = 16384

    async def answer():
        await Readout(fake_backend, Profile(repeat=True)).answer_all("short state", {"q": {"type": "noul", "instructions": "ok?"}})
        return threading.get_ident()

    loop_thread = asyncio.run(answer())
    assert threads and loop_thread not in threads


@pytest.mark.parametrize("window", [0, 100])
def test_read_twice_sends_a_prompt_too_long_for_any_count_once(fake_backend, monkeypatch, window):
    """More characters than the window times the longest token cannot fit: such a prompt goes out once, counted, never tokenized."""

    def count(text):
        raise AssertionError("tokenized")

    monkeypatch.setattr(fake_backend, "header_tokens", count)
    fake_backend.window = window
    ans, _ = asyncio.run(Readout(fake_backend, Profile(repeat=True)).answer_all("x" * (window * 128), {"q": {"type": "noul", "instructions": "ok?"}}))
    assert ans["q"]["read_once"] == 1 and [p.count("\nQuestion: ") for p in fake_backend.prompts] == [1]


def test_no_token_is_longer_than_the_read_twice_bound(fake_backend):
    tok = fake_backend.tok
    assert max(len(t) for t in tok.convert_ids_to_tokens(list(range(len(tok))))) <= TOKEN_CHARS_MAX


def test_noul_calibration_direction(fake_backend):
    ro = Readout(fake_backend, Profile(noul_t=1.0, noul_bias=0.0))
    ans, _ = asyncio.run(ro.answer_all("s", {"q": {"type": "noul", "instructions": "?"}}))
    ro2 = Readout(fake_backend, Profile(noul_t=1.0, noul_bias=3.0))
    ans2, _ = asyncio.run(ro2.answer_all("s", {"q": {"type": "noul", "instructions": "?"}}))
    assert ans2["q"]["noul"] > ans["q"]["noul"]
