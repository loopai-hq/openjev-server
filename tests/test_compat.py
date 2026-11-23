"""Byte-for-byte compatibility with the released helper: same prompts, same probabilities, on the published profile."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from openjev_server.profile import Profile
from openjev_server.readout import Readout
from openjev_server.server import build_app

STATES = [
    "Customer message: I was charged twice for my order last week and nobody has replied.",
    {
        "page": {"url": "https://example.com/checkout", "title": "Checkout", "text": "Shipping address " * 40},
        "elements": [f"[{i}] button Option {i}" for i in range(12)],
        "recent_actions": [{"action": "CLICK", "text": "Cart"}],
    },
    {"task": "Book a flight", "previous_actions": ["click Flights"], "screenshot": "data:image/png;base64," + "A" * 3000},
]
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": None, "shipping": "Delivery problems", "technical": None},
    },
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today", "right now"]},
    "dict_instr": {"type": "choice", "instructions": {"goal": "find flights", "rules": "pick one"}, "criteria": {str(i): f"opt {i}" for i in range(7)}},
    "many": {"type": "choice", "instructions": "Pick the element.", "criteria": {f"e{i}": f"element {i}" for i in range(120)}},
}


def shim_answers(shim, state, questions):
    st = shim.with_image(state)
    return {qid: shim.ANSWER[q["type"]](st, q)[0] for qid, q in questions.items()}


@pytest.mark.parametrize("state", STATES, ids=["text", "dict", "screenshot"])
def test_answers_equal_released_helper(shim, fake_backend, state):
    ro = Readout(fake_backend, Profile())
    ours, _ = asyncio.run(ro.answer_all(state, QUESTIONS))
    theirs = shim_answers(shim, state, QUESTIONS)
    assert json.dumps(ours, sort_keys=True) == json.dumps(theirs, sort_keys=True)


def test_prompts_equal_released_helper(shim, fake_backend):
    ro = Readout(fake_backend, Profile())
    asyncio.run(ro.answer_all(STATES[1], {"route": QUESTIONS["route"], "urgency": QUESTIONS["urgency"]}))
    shim._seen_prompts.clear()
    shim_answers(shim, STATES[1], {"route": QUESTIONS["route"], "urgency": QUESTIONS["urgency"]})
    assert sorted(map(json.dumps, fake_backend.prompts)) == sorted(map(json.dumps, shim._seen_prompts))


def test_http_requests_send_the_released_helpers_prompts(shim, fake_backend):
    """Requests the server accepted before 0.3.0, sent over HTTP as plain or ASCII-escaped JSON: the model sees the released helper's
    prompts, byte for byte."""
    c = TestClient(build_app(Readout(fake_backend, Profile())))
    for state in [*STATES, "Message reçu : facturé deux fois, 3 € 😡"]:
        body = {"model": "x", "state": state, "questions": QUESTIONS}
        shim._seen_prompts.clear()
        shim_answers(shim, state, QUESTIONS)
        for raw in (json.dumps(body).encode(), json.dumps(body, ensure_ascii=False).encode()):
            fake_backend.prompts.clear()
            assert c.post("/v1/systemone", content=raw, headers={"Content-Type": "application/json"}).status_code == 200
            assert sorted(map(json.dumps, fake_backend.prompts)) == sorted(map(json.dumps, shim._seen_prompts))


def helper_1_1_answers(shim_1_1, state, questions):
    """The 1.1 helper's answers through its answer_one, which adds read_once when read twice sent a read once."""
    st = shim_1_1.with_image(state)
    return {qid: shim_1_1.answer_one(st, q)[0] for qid, q in questions.items()}


@pytest.mark.parametrize("state", STATES, ids=["text", "dict", "screenshot"])
def test_read_twice_equals_released_helper(shim_1_1, fake_backend, monkeypatch, state):
    """`repeat` against OpenJev 1.1's helper with READOUT_REPEAT=1: the same prompts (text doubled, screenshots once), the same answers."""
    monkeypatch.setattr(shim_1_1, "REPEAT", True)
    shim_1_1._seen_prompts.clear()
    ours, _ = asyncio.run(Readout(fake_backend, Profile(repeat=True)).answer_all(state, QUESTIONS))
    theirs = helper_1_1_answers(shim_1_1, state, QUESTIONS)
    assert json.dumps(ours, sort_keys=True) == json.dumps(theirs, sort_keys=True)
    assert sorted(map(json.dumps, fake_backend.prompts)) == sorted(map(json.dumps, shim_1_1._seen_prompts))
    texts = [p for p in fake_backend.prompts if isinstance(p, str)]
    assert all(p.count("\n\nQuestion: ") == 2 for p in texts) and len(texts) == (0 if "screenshot" in state else len(fake_backend.prompts))


def test_read_twice_window_edge_equals_released_helper(shim_1_1, fake_backend, monkeypatch):
    """A doubled prompt that fills the window exactly goes out twice; one token over, it goes out once and the answer says so. Our count
    of the doubled prompt is the 1.1 helper's, which counts the way vLLM's chat endpoint does."""
    monkeypatch.setattr(shim_1_1, "REPEAT", True)
    route = {"route": QUESTIONS["route"]}
    ro = Readout(fake_backend, Profile(repeat=True))
    asyncio.run(ro.answer_all(STATES[0], route))
    (twice,) = fake_backend.prompts
    need = shim_1_1._prompt_tokens([{"role": "user", "content": twice}]) + 1  # + the one output token of a read
    for window, sent, once in ((need, twice, None), (need - 1, twice[: (len(twice) - 2) // 2], 1)):
        monkeypatch.setattr(shim_1_1, "WINDOW", window)
        fake_backend.window = window
        fake_backend.prompts.clear()
        ours = asyncio.run(ro.answer_all(STATES[0], route))[0]["route"]
        assert fake_backend.prompts == [sent] and ours.get("read_once") == once
        assert ours == helper_1_1_answers(shim_1_1, STATES[0], route)["route"]


def test_read_twice_counts_every_read_sent_once(shim_1_1, fake_backend, monkeypatch):
    """A window too small for any doubled prompt: every read goes once and each answer counts its reads (120 options: 3 chunks + 1)."""
    monkeypatch.setattr(shim_1_1, "REPEAT", True)
    monkeypatch.setattr(shim_1_1, "WINDOW", 1)
    fake_backend.window = 1
    ours, _ = asyncio.run(Readout(fake_backend, Profile(repeat=True)).answer_all(STATES[1], QUESTIONS))
    assert json.dumps(ours, sort_keys=True) == json.dumps(helper_1_1_answers(shim_1_1, STATES[1], QUESTIONS), sort_keys=True)
    assert ours["many"]["read_once"] == 4 and ours["route"]["read_once"] == 1


def test_perms_average_matches(shim, fake_backend, monkeypatch):
    monkeypatch.setattr(shim, "PERMS", 4)
    ro = Readout(fake_backend, Profile(perms=4))
    ours, _ = asyncio.run(ro.answer_all(STATES[0], {"route": QUESTIONS["route"], "dict_instr": QUESTIONS["dict_instr"]}))
    theirs = shim_answers(shim, STATES[0], {"route": QUESTIONS["route"], "dict_instr": QUESTIONS["dict_instr"]})
    assert json.dumps(ours, sort_keys=True) == json.dumps(theirs, sort_keys=True)
