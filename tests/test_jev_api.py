"""The hosted Jev API's request surface (Jev 1.14) on POST /v1/systemone: compressed bodies, the 32 MiB limit, one image through
`assets`, optional instructions, one-level scores, the requested model echoed, GET /v1/models and errors that carry `detail`."""

import base64
import gzip
import json
import tracemalloc

import pytest
import zstandard
from fastapi.testclient import TestClient

from openjev_server import body
from openjev_server.backends.base import BackendError
from openjev_server.profile import Profile
from openjev_server.readout import Readout
from openjev_server.server import build_app

MIB32 = 32 * 1024 * 1024
REQ = {
    "state": {"text": "A customer writes: charged twice."},
    "questions": {
        "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Billing", "tech": None}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "tone": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
    },
}
NOUL = {"q": {"type": "noul", "instructions": "Is there a person at the door?"}}
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 3 + b"\r\n--\r\n"  # carries bytes that look like a multipart boundary
JPEG = b"\xff\xd8\xff\xe0" + bytes(range(255, -1, -1)) + b"\r\n"
GIF = b"GIF89a" + bytes(40)


def api(fake_backend, **kw):
    return TestClient(build_app(Readout(fake_backend, Profile()), model_dir="/models/openjev", **kw))


def post(c, raw, **headers):
    return c.post("/v1/systemone", content=raw, headers={"Content-Type": "application/json", **headers})


def data_url(mime, raw):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def b64(raw):
    return {"type": "image", "format": "png", "data_base64": base64.b64encode(raw).decode()}


def multipart(parts):
    """The hosted API's multipart shape: the JSON as the first part, named `request`, then one part per image."""
    b = "9f2c41d0b7e84a6c"

    def head(name, kind):
        file = "" if name == "request" else f'; filename="{name}"'
        return f'--{b}\r\nContent-Disposition: form-data; name="{name}"{file}\r\nContent-Type: {kind}\r\n\r\n'.encode()

    return b"".join(head(n, t) + c + b"\r\n" for n, t, c in parts) + f"--{b}--\r\n".encode(), {"Content-Type": f"multipart/form-data; boundary={b}"}


def sized(n):
    """A valid body of exactly n bytes: one yes/no question over a state of 'a's."""
    head, tail = b'{"state": "', b'", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}'
    return head + b"a" * (n - len(head) - len(tail)) + tail


def zstd_frame(content, window_log):
    """A zstd frame written by hand: no content size, so the decoder must reserve the window the frame declares (2**window_log
    bytes), then the content as one raw block."""
    descriptor = (window_log - 10) << 3  # Window_Descriptor: exponent, mantissa 0
    block = (len(content) << 3 | 1).to_bytes(3, "little")  # Block_Header: a raw block, the last one
    return b"\x28\xb5\x2f\xfd" + b"\x00" + bytes([descriptor]) + block + content


@pytest.mark.parametrize("encoding", ["gzip", "x-gzip", "zstd"])
def test_compressed_bodies_send_the_same_prompts(fake_backend, encoding):
    c = api(fake_backend)
    raw = json.dumps(REQ).encode()
    plain = post(c, raw).json()["answers"]
    sent = sorted(map(json.dumps, fake_backend.prompts))
    fake_backend.prompts.clear()
    packed = zstandard.ZstdCompressor().compress(raw) if encoding == "zstd" else gzip.compress(raw)
    r = post(c, packed, **{"Content-Encoding": encoding})
    assert r.status_code == 200 and r.json()["answers"] == plain and sorted(map(json.dumps, fake_backend.prompts)) == sent


@pytest.mark.parametrize(
    ("raw", "headers", "status", "says"),
    [
        (b'{"state": ', {}, 400, "malformed JSON"),
        (b'{"state": ' + b"[" * 100_000 + b"]" * 100_000 + b"}", {}, 400, "malformed JSON"),
        (b"[1, 2]", {}, 400, "JSON object"),
        (b"not gzip", {"Content-Encoding": "gzip"}, 400, "malformed gzip"),
        (gzip.compress(json.dumps(REQ).encode())[:-8], {"Content-Encoding": "gzip"}, 400, "truncated"),
        (gzip.compress(json.dumps(REQ).encode()) + gzip.compress(b""), {"Content-Encoding": "gzip"}, 400, "one gzip member"),
        (b"not zstd", {"Content-Encoding": "zstd"}, 400, "malformed zstd"),
        (zstandard.ZstdCompressor().compress(json.dumps(REQ).encode())[:-5], {"Content-Encoding": "zstd"}, 400, "malformed zstd body: truncated"),
        (b"".join(map(zstandard.ZstdCompressor().compress, (b'{"state": "s", ', b'"questions": {}}'))), {"Content-Encoding": "zstd"}, 400, "one zstd frame"),
        (json.dumps(REQ).encode(), {"Content-Encoding": "br"}, 415, "unsupported"),
        (sized(MIB32 + 1), {}, 413, "32 MiB"),
        (gzip.compress(sized(MIB32 + 1)), {"Content-Encoding": "gzip"}, 413, "32 MiB"),
    ],
    ids=[
        "json",
        "nested",
        "not-object",
        "gzip",
        "gzip-truncated",
        "gzip-members",
        "zstd",
        "zstd-truncated",
        "zstd-frames",
        "encoding",
        "size",
        "size-inflated",
    ],
)
def test_unreadable_bodies_are_json_errors(fake_backend, raw, headers, status, says):
    r = post(api(fake_backend), raw, **headers)
    e = r.json()
    assert r.status_code == status and says in e["detail"] and e["error"] == {"code": status, "message": e["detail"]}


@pytest.mark.parametrize("encoding", ["identity", "gzip"])
def test_a_body_of_exactly_32_mib_is_read(fake_backend, encoding):
    raw = sized(MIB32)
    r = post(api(fake_backend), gzip.compress(raw) if encoding == "gzip" else raw, **{"Content-Encoding": encoding})
    assert r.status_code == 200 and "a" * (MIB32 - 100) in fake_backend.prompts[0]


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_bodies_that_inflate_past_the_limit_are_413(fake_backend, monkeypatch, encoding):
    """A few KiB that inflate to one byte past the limit (lowered here to 4 MiB) are a 413."""
    monkeypatch.setattr(body, "MAX_BODY", 4 * 1024 * 1024)
    raw = sized(4 * 1024 * 1024 + 1)
    packed = zstandard.ZstdCompressor().compress(raw) if encoding == "zstd" else gzip.compress(raw)
    r = post(api(fake_backend), packed, **{"Content-Encoding": encoding})
    assert len(packed) < 64 * 1024 and r.status_code == 413 and r.json()["detail"] == "the body is over 4 MiB after decompression"


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_a_small_body_inflates_in_small_steps(encoding):
    """Inflating reads in steps of at most 1 MiB: a small body never makes the server allocate anything near the 32 MiB limit."""
    raw = json.dumps(REQ).encode()
    packed = zstandard.ZstdCompressor().compress(raw) if encoding == "zstd" else gzip.compress(raw)
    tracemalloc.start()
    try:
        assert body.inflate(packed, encoding) == raw
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024, peak


@pytest.mark.parametrize(("window_log", "status"), [(25, 200), (26, 400)])
def test_a_zstd_window_above_the_limit_is_400(fake_backend, window_log, status):
    """A frame of a few bytes may declare any window: up to the 32 MiB limit (2**25) it is read, above it the frame is refused."""
    r = post(api(fake_backend), zstd_frame(json.dumps(REQ).encode(), window_log), **{"Content-Encoding": "zstd"})
    assert r.status_code == status and (status == 200 or "malformed zstd" in r.json()["detail"])


def test_instructions_may_be_left_out(fake_backend):
    """The hosted API's questions may come without instructions: each type gets a neutral default."""
    qs = {
        "c": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        "n": {"type": "noul", "instructions": None},
        "s": {"type": "score", "criteria": ["low", "high"]},
    }
    assert post(api(fake_backend), json.dumps({"state": "plain text state", "questions": qs}).encode()).status_code == 200
    asked = sorted(p.split("\nQuestion: ")[1].split("\n")[0] for p in fake_backend.prompts)
    assert asked == ["Is the statement true?", "Rate the content above. Rate along the ordered levels below (lowest first).", "Which option is correct?"]


def test_one_level_score_needs_no_model(fake_backend):
    """The hosted API takes scores with one level: all probability on it, no model call. Zero levels stay a 422."""
    c = api(fake_backend)
    r = post(c, json.dumps({"state": "s", "questions": {"one": {"type": "score", "criteria": [{"level": "Broken"}]}}}).encode()).json()
    assert r["answers"]["one"] == {"type": "score", "score": 0.0, "legend": {"0": {"level": "Broken"}}, "probabilities": {"0": 1.0}, "confidence": 1.0}
    assert r["usage"]["input_tokens"] == 0 and fake_backend.prompts == []
    assert post(c, json.dumps({"state": "s", "questions": {"zero": {"type": "score", "instructions": "x", "criteria": []}}}).encode()).status_code == 422


def test_array_state_is_json_text(fake_backend):
    arr = ["Hi", "Mon reçu: charged twice", {"n": 1}]
    post(api(fake_backend), json.dumps({"state": arr, "questions": {"q": REQ["questions"]["urgent"]}}).encode())
    assert fake_backend.prompts[0].startswith("State:\n" + json.dumps(arr, ensure_ascii=False) + "\n\nQuestion: ")


def test_requested_model_is_echoed(fake_backend):
    c = api(fake_backend)
    assert post(c, json.dumps({**REQ, "model": "jev-latest"}).encode()).json()["model"] == "jev-latest"
    assert post(c, json.dumps(REQ).encode()).json()["model"].startswith("openjev T=0.85")  # no model named: the build string, as before


def test_models_lists_the_served_model(fake_backend):
    m = api(fake_backend).get("/v1/models").json()
    assert list(m) == ["models"] and m["models"][0]["name"] == "openjev" and len(m["models"][0]["release_date"]) == 10
    assert set(m["models"][0]) == {"name", "release_date", "description"}
    c = api(fake_backend, token="t0ken")
    assert c.get("/v1/models").status_code == 401 and c.get("/v1/models", headers={"Authorization": "Bearer t0ken"}).status_code == 200


def test_errors_carry_detail(fake_backend, monkeypatch):
    """Every error is JSON with our `error` object and the hosted API's `detail`: the message, or for 422 a list of {loc, msg, type}."""
    c = api(fake_backend)
    box = post(c, json.dumps({"state": "s", "questions": {"b": {"type": "box", "instructions": "find it"}}}).encode())
    empty = post(c, json.dumps({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}}).encode())
    for r, says in ((box, "choice, score, noul"), (empty, "criteria")):
        e = r.json()
        assert r.status_code == 422 and e["error"]["code"] == 422 and says in e["error"]["message"]
        assert [set(d) for d in e["detail"]] == [{"loc", "msg", "type"}] and e["detail"][0]["loc"][0] == "body" and says in e["detail"][0]["msg"]
    assert c.get("/v1/nothing").json()["detail"] == "Not Found"
    assert api(fake_backend, token="t").get("/v1/version").json()["detail"] == "missing or invalid bearer token"

    async def down(content, token_ids):
        raise BackendError("model server down")

    monkeypatch.setattr(fake_backend, "logprobs", down)
    r = post(c, json.dumps(REQ).encode())
    assert r.status_code == 502 and "model server down" in r.json()["detail"]


def test_one_image_through_assets(fake_backend):
    """One image sent the hosted way (multipart, or base64 in the JSON) reaches the model the way state.image does."""
    door = {
        "model": "jev-latest",
        "state": {"door": "ref-7c1e", "note": "front door"},
        "assets": {"ref-7c1e": {"type": "image", "part": "asset-0"}},
        "questions": NOUL,
    }
    raw, headers = multipart([("request", "application/json", json.dumps(door).encode()), ("asset-0", "image/png", PNG)])
    c = api(fake_backend)
    for sent, extra in ((raw, {}), (gzip.compress(raw), {"Content-Encoding": "gzip"})):  # the SDK compresses multipart bodies too
        fake_backend.prompts.clear()
        r = c.post("/v1/systemone", content=sent, headers={**headers, **extra})
        ((image, text),) = fake_backend.prompts
        assert r.status_code == 200 and r.json()["model"] == "jev-latest"
        assert image == {"type": "image_url", "image_url": {"url": data_url("image/png", PNG)}} and 'State:\n{"note": "front door"}\n' in text["text"]
    fake_backend.prompts.clear()
    post(c, json.dumps({"state": "photo", "assets": {"photo": {**b64(JPEG), "format": "jpg"}}, "questions": NOUL}).encode())
    ((image, text),) = fake_backend.prompts
    assert image["image_url"]["url"] == data_url("image/jpeg", JPEG) and "State:\n(see screenshot)\n" in text["text"]


def test_an_image_beside_a_deeply_nested_state(fake_backend):
    """Nesting deeper than Python's recursion limit, which the JSON parser still reads: the image reference is found all the same."""
    deep = "[" * 5000 + "]" * 5000
    raw = f'{{"state": {{"door": "r1", "deep": {deep}}}, "assets": {json.dumps({"r1": b64(PNG)})}, "questions": {json.dumps(NOUL)}}}'
    r = post(api(fake_backend), raw.encode())
    assert r.status_code == 200 and fake_backend.prompts[0][0]["image_url"]["url"] == data_url("image/png", PNG)


def test_a_multipart_body_holds_at_most_two_parts(fake_backend, monkeypatch):
    """The JSON and at most one image: a third part is a 400, found by counting boundaries before any part is parsed."""

    def unread(piece):
        raise AssertionError("a part was parsed")

    monkeypatch.setattr(body, "read_part", unread)
    raw, headers = multipart([("request", "application/json", json.dumps(REQ).encode()), ("asset-0", "image/png", PNG), ("asset-1", "image/png", PNG)])
    r = api(fake_backend).post("/v1/systemone", content=raw, headers=headers)
    assert r.status_code == 400 and "at most one image" in r.json()["detail"]


@pytest.mark.parametrize(
    "disposition", [b"form-data; name=" + b"(" * 500, b'a;n*\ra=-t;c"a*b*pi\x00Dn'], ids=["parentheses-500-deep", "broken-rfc2231-parameter"]
)
def test_hostile_part_headers_are_400(fake_backend, disposition):
    """Part headers that crash the email package's modern header parser (RecursionError, IndexError) are a JSON 400."""
    b = "9f2c41d0b7e84a6c"
    raw = f"--{b}\r\nContent-Disposition: ".encode() + disposition + b"\r\n\r\n" + json.dumps(REQ).encode() + f"\r\n--{b}--\r\n".encode()
    r = api(fake_backend).post("/v1/systemone", content=raw, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    assert r.status_code == 400 and r.json()["error"]["code"] == 400


@pytest.mark.parametrize(
    ("content_type", "status"),
    [(None, 415), ("text/plain", 415), ("application/x-www-form-urlencoded", 415), ("application/json; charset=utf-8", 200), ("application/vnd.api+json", 200)],
)
def test_only_json_or_multipart_bodies_are_read(fake_backend, content_type, status):
    """A web page may send text/plain or form bodies to any server it reaches without asking first: only JSON or multipart is read."""
    r = api(fake_backend).post("/v1/systemone", content=json.dumps(REQ).encode(), headers={} if content_type is None else {"Content-Type": content_type})
    assert r.status_code == status and (status == 200 or "Content-Type" in r.json()["detail"])


@pytest.mark.parametrize(("size", "status"), [(8 * 1024, 200), (8 * 1024 + 1, 400)])
def test_a_multipart_part_has_at_most_8_kib_of_headers(fake_backend, size, status):
    prefix = 'Content-Disposition: form-data; name="asset-0"; filename="asset-0"\r\nContent-Type: image/png\r\nX-Pad: '
    kind = "image/png\r\nX-Pad: " + "a" * (size - len(prefix))  # the part's header block is prefix + padding: `size` bytes
    door = {"state": "ref-7c1e", "assets": {"ref-7c1e": {"part": "asset-0"}}, "questions": NOUL}
    raw, headers = multipart([("request", "application/json", json.dumps(door).encode()), ("asset-0", kind, PNG)])
    assert api(fake_backend).post("/v1/systemone", content=raw, headers=headers).status_code == status


BAD_ASSETS = {
    "two images": {"state": {"a": "r1", "b": "r2"}, "assets": {"r1": b64(PNG), "r2": b64(PNG)}},
    "used twice": {"state": {"a": "r1", "b": "r1"}, "assets": {"r1": b64(PNG)}},
    "in instructions": {"state": {"a": "r1"}, "assets": {"r1": b64(PNG)}, "questions": {"q": {"type": "noul", "instructions": "Same as r1?"}}},
    "in criteria": {
        "state": "s",
        "assets": {"r1": b64(PNG)},
        "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"x": "r1", "y": None}}},
    },
    "nested": {"state": {"frames": [{"f": "r1"}]}, "assets": {"r1": b64(PNG)}},
    "inside text": {"state": "the photo r1 shows", "assets": {"r1": b64(PNG)}},
    "not referenced": {"state": "s", "assets": {"r1": b64(PNG)}},
    "beside a screenshot": {"state": {"a": "r1", "screenshot": "data:image/png;base64,AAAA"}, "assets": {"r1": b64(PNG)}},
    "not PNG or JPEG": {"state": "r1", "assets": {"r1": b64(GIF)}},
    "bad base64": {"state": "r1", "assets": {"r1": {"format": "png", "data_base64": "!!"}}},
    "base64 not a string": {"state": "r1", "assets": {"r1": {"format": "png", "data_base64": 5}}},
    "part not a string": {"state": "r1", "assets": {"r1": {"part": ["asset-0"]}}},
}


@pytest.mark.parametrize("why", BAD_ASSETS)
def test_images_the_model_cannot_read_are_400(fake_backend, why):
    r = post(api(fake_backend), json.dumps({"questions": NOUL, **BAD_ASSETS[why]}).encode())
    assert r.status_code == 400 and isinstance(r.json()["detail"], str) and fake_backend.prompts == []


def door_json(asset):
    return json.dumps({"state": "ref-7c1e", "assets": {"ref-7c1e": asset}, "questions": NOUL}).encode()


@pytest.mark.parametrize(
    ("parts", "says"),
    [
        ([("asset-0", "image/png", PNG), ("request", "application/json", b"{}")], "first part, named request"),
        ([("request", "application/json", door_json(b64(PNG))), ("asset-0", "image/png", PNG)], "belong to no asset"),
        ([("request", "application/json", door_json({"part": "asset-9"})), ("asset-0", "image/png", PNG)], "multipart part is missing"),
    ],
    ids=["request-not-first", "part-without-asset", "asset-part-missing"],
)
def test_malformed_multipart_is_400(fake_backend, parts, says):
    raw, headers = multipart(parts)
    r = api(fake_backend).post("/v1/systemone", content=raw, headers=headers)
    assert r.status_code == 400 and says in r.json()["detail"]
