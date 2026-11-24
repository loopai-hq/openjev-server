import gzip
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from openjev_server import server
from openjev_server.profile import Profile
from openjev_server.readout import Readout
from openjev_server.server import build_app

REQ = {
    "model": "x",
    "state": "I was charged twice.",
    "questions": {"route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}}},
}


def client(fake_backend, token="", profile=None):
    return TestClient(build_app(Readout(fake_backend, profile or Profile()), token=token, model_dir="/models/openjev", backend_name="fake"))


def metric(c, name):
    """The current value of one unlabelled Prometheus series."""
    return float(next(line.split()[1] for line in c.get("/metrics").text.splitlines() if line.startswith(name + " ")))


def test_systemone_shape(fake_backend):
    r = client(fake_backend).post("/v1/systemone", json=REQ)
    assert r.status_code == 200
    d = r.json()
    assert d["id"].startswith("oj-") and d["usage"]["output_tokens"] == 0 and d["usage"]["input_tokens"] > 0
    a = d["answers"]["route"]
    assert a["type"] == "choice" and a["choice"] in ("billing", "shipping") and abs(sum(a["probabilities"].values()) - 1) < 0.01 and 0 <= a["confidence"] <= 1


def test_auth(fake_backend):
    c = client(fake_backend, token="secret")
    assert c.post("/v1/systemone", json=REQ).status_code == 401
    assert c.get("/v1/version").status_code == 401
    assert c.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer secret"}).status_code == 200
    assert c.get("/healthz").status_code == 200  # never behind auth


def test_cross_site_browser_requests_need_a_token(fake_backend, monkeypatch):
    """Without --token a web page could make a visitor's browser post work here (multipart/form-data needs no CORS preflight).
    Browsers mark such requests Sec-Fetch-Site: cross-site; every /v1 route refuses them with 403 before any body is read."""

    async def unread(request):
        raise AssertionError("the body was read")

    monkeypatch.setattr(server, "read_body", unread)
    c = client(fake_backend)
    cross = {"Sec-Fetch-Site": "cross-site", "Content-Type": "multipart/form-data; boundary=x"}
    for method, path in (("POST", "/v1/systemone"), ("POST", "/v1/prewarm"), ("GET", "/v1/version"), ("GET", "/v1/models")):
        r = c.request(method, path, content=b"--x\r\n" if method == "POST" else None, headers=cross)
        assert r.status_code == 403 and r.json()["error"]["code"] == 403, (path, r.status_code)


def test_same_origin_calls_and_tokens_are_unchanged(fake_backend):
    """Same-origin pages, clients that send no Sec-Fetch-Site (SDKs, curl), and cross-site calls with a valid token are served."""
    c = client(fake_backend)
    assert c.post("/v1/systemone", json=REQ, headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert c.post("/v1/systemone", json=REQ).status_code == 200
    t = client(fake_backend, token="secret")
    assert t.post("/v1/systemone", json=REQ, headers={"Sec-Fetch-Site": "cross-site", "Authorization": "Bearer secret"}).status_code == 200
    assert t.post("/v1/systemone", json=REQ, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 401


def test_readout_hash_covers_the_code_that_writes_the_state(fake_backend):
    """readout_sha256 covers prompt.py, readout.py and body.py, which moves an image from `assets` into the state the model reads."""
    here = Path(server.__file__).parent
    want = hashlib.sha256(b"".join((here / m).read_bytes() for m in ("prompt.py", "readout.py", "body.py"))).hexdigest()
    assert client(fake_backend).get("/v1/version").json()["readout_sha256"] == want


def test_validation(fake_backend):
    c = client(fake_backend)
    assert c.post("/v1/systemone", json={"state": "s", "questions": {}}).status_code == 422
    assert c.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "essay"}}}).status_code == 422
    r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}})
    assert r.status_code == 422 and "criteria" in r.json()["error"]["message"]


def test_version_health_metrics_prewarm(fake_backend):
    c = client(fake_backend)
    v = c.get("/v1/version").json()
    assert v["model_dir"] == "openjev" and v["T"] == 0.85 and v["flags"]["targeted"] is True and len(v["readout_sha256"]) == 64
    assert c.get("/readyz").json()["ok"] is True
    c.post("/v1/systemone", json=REQ)
    m = c.get("/metrics").text
    assert "openjev_requests_total" in m and "openjev_request_seconds" in m
    assert c.post("/v1/prewarm", json={"state": REQ["state"]}).json()["ok"] is True


def test_prewarm_checks_the_token_before_it_reads_the_body(fake_backend):
    """/v1/prewarm reads its body as /v1/systemone does: after the bearer check, capped at 32 MiB, compressed bodies too."""
    c = client(fake_backend, token="secret")
    json_type, auth = {"Content-Type": "application/json"}, {"Content-Type": "application/json", "Authorization": "Bearer secret"}
    assert c.post("/v1/prewarm", content=b'{"state": ', headers=json_type).status_code == 401
    assert c.post("/v1/prewarm", content=b'{"state": "' + b"a" * (32 * 1024 * 1024) + b'"}', headers=auth).status_code == 413
    r = c.post("/v1/prewarm", content=gzip.compress(json.dumps({"state": REQ["state"]}).encode()), headers={**auth, "Content-Encoding": "gzip"})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_read_twice_is_reported_and_its_fallbacks_counted(fake_backend):
    """/v1/version names the switch; an answer whose doubled prompt did not fit the window carries read_once, and /metrics adds it up."""
    assert client(fake_backend).get("/v1/version").json()["flags"]["repeat"] is False
    fake_backend.window = 1
    c = client(fake_backend, profile=Profile(repeat=True))
    assert c.get("/v1/version").json()["flags"]["repeat"] is True
    before = metric(c, "openjev_read_once_total")
    assert c.post("/v1/systemone", json=REQ).json()["answers"]["route"]["read_once"] == 1
    assert metric(c, "openjev_read_once_total") == before + 1
