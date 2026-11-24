"""The HTTP surface: POST /v1/systemone (decisions), POST /v1/prewarm, GET /v1/version, GET /v1/models, GET /healthz, GET /readyz,
GET /metrics, POST /v1/chat/completions (passthrough for clients that also type text). Bearer auth on /v1/* when a token is configured.
Every error is JSON: our `error` object next to the hosted API's `detail`."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .backends.base import BackendError
from .body import BadBody, read_body
from .readout import BadQuestion, Readout, ReadoutIncomplete

log = logging.getLogger("openjev")

REQS = Counter("openjev_requests_total", "systemone requests", ["status"])
QUESTIONS = Counter("openjev_questions_total", "questions answered", ["type"])
TOKENS = Counter("openjev_prompt_tokens_total", "prompt tokens sent to the model")
LAT = Histogram("openjev_request_seconds", "systemone request latency", buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13))
INFLIGHT = Gauge("openjev_inflight_requests", "requests in flight")
READ_ONCE = Counter("openjev_read_once_total", "reads sent once because their doubled prompt (read twice) would not fit the window")

ANSWER_MODULES = ("prompt.py", "readout.py", "body.py")  # the code that shapes what the model reads and the answers; hashed in /v1/version


class Question(BaseModel):
    type: str
    instructions: Any = None
    criteria: Any = None

    @field_validator("type")
    @classmethod
    def _type(cls, v):
        if v not in ("choice", "score", "noul"):
            raise ValueError("type must be one of choice, score, noul")
        return v


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: Any
    questions: dict[str, Question] = Field(min_length=1)


class PrewarmRequest(BaseModel):
    state: Any


Body = TypeVar("Body", bound=BaseModel)


def http_error(status: int, message: str) -> HTTPException:
    return HTTPException(status, {"code": status, "message": message})


def error_response(status: int, message: str, detail: Any = None, headers: dict[str, str] | None = None) -> JSONResponse:
    """Our {"error": {code, message}} next to the hosted API's `detail`: the message, or for 422 a list of {loc, msg, type}."""
    if detail is None:
        detail = [{"loc": ["body"], "msg": message, "type": "value_error"}] if status == 422 else message
    return JSONResponse({"error": {"code": status, "message": message}, "detail": detail}, status_code=status, headers=headers)


async def read_request(request: Request, model: type[Body]) -> Body:
    """A POST body read the way the hosted API's clients send it (capped, compressed, multipart, an image through `assets`), then
    validated as `model`. Routes call it themselves, so the bearer check runs before any of the body is read."""
    try:
        return model.model_validate(await read_body(request))
    except BadBody as e:
        raise http_error(e.status, str(e)) from e
    except ValidationError as e:
        raise RequestValidationError([{**err, "loc": ("body", *err["loc"])} for err in e.errors()]) from e


def version_info(readout: Readout, *, backend_name: str, model_dir: str, extra: dict | None) -> dict:
    """What GET /v1/version reports: the profile constants, the backend and a hash of the answer-determining code."""
    p = readout.profile
    here = Path(__file__).parent
    code = b"".join((here / m).read_bytes() for m in ANSWER_MODULES)
    return {
        "server": "openjev-server",
        "version": __version__,
        "backend": backend_name,
        "model_dir": os.path.basename(model_dir.rstrip("/")) or model_dir,
        "profile": p.name,
        "T": p.temp,
        "noul_t": p.noul_t,
        "noul_bias": p.noul_bias,
        "flags": {
            "perms": p.perms,
            "stagger": p.stagger,
            "compact": p.compact,
            "compact_cap": p.compact_cap,
            "layout": p.layout,
            "pad": p.pad,
            "targeted": True,
            "instr_style": p.instr_style,
            "repeat": p.repeat,
        },
        "readout_sha256": hashlib.sha256(code).hexdigest(),
        **(extra or {}),
    }


def access_check(token: str):
    """The dependency guarding /v1/*. With a token configured, `Authorization: Bearer <token>` is required. Without one, a request
    that a browser marks cross-site is refused: a web page could otherwise make a visitor's browser post work here, as a
    multipart/form-data post needs no CORS preflight. SDKs and curl send no Sec-Fetch-Site header."""

    async def check(request: Request):
        if token:
            if request.headers.get("Authorization", "") != f"Bearer {token}":
                raise http_error(401, "missing or invalid bearer token")
        elif request.headers.get("Sec-Fetch-Site") == "cross-site":
            raise http_error(403, "cross-site requests from a browser need a server started with --token")

    return check


def ops_router(readout: Readout, backend_name: str) -> APIRouter:
    """Unauthenticated operations endpoints: liveness, readiness, Prometheus metrics."""
    r = APIRouter()

    @r.get("/healthz")
    async def healthz():
        return {"ok": True}

    @r.get("/readyz")
    async def readyz():
        ok = await readout.backend.healthy()
        return JSONResponse({"ok": ok, "backend": backend_name}, status_code=200 if ok else 503)

    @r.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return r


def api_router(readout: Readout, version: dict, model_string: str, token: str) -> APIRouter:
    """The /v1 API: version, models, prewarm, systemone, chat passthrough."""
    r = APIRouter(prefix="/v1", dependencies=[Depends(access_check(token))])
    started = time.strftime("%Y-%m-%d", time.gmtime())

    @r.get("/version")
    @r.post("/version")
    async def get_version():
        return version

    @r.get("/models")
    async def models():
        """The served model in the hosted API's shape (its SDKs list models). The model's own release date is unknown here."""
        description = "The decision model this server serves. release_date is the day the server started; GET /v1/version has the calibration and flags."
        return {"models": [{"name": version["model_dir"], "release_date": started, "description": description}]}

    @r.post("/prewarm")
    async def prewarm(request: Request):
        body = await read_request(request, PrewarmRequest)
        t0 = time.perf_counter()
        try:
            n = await readout.prewarm(body.state)
        except (BackendError, ReadoutIncomplete) as e:
            raise http_error(502, str(e)[:200]) from e
        log.info(json.dumps({"prewarm_tokens": n, "ms": round((time.perf_counter() - t0) * 1000)}))
        return {"ok": True, "prompt_tokens": n}

    @r.post("/systemone")
    async def systemone(request: Request):
        body = await read_request(request, SystemOneRequest)
        rid = f"oj-{uuid.uuid4().hex[:16]}"
        t0 = time.perf_counter()
        answers, toks = await answer_request(readout, body)
        record(rid, answers, toks, time.perf_counter() - t0)
        model = model_string if body.model is None else body.model  # the requested model is echoed, as on the hosted API
        return {"id": rid, "model": model, "answers": answers, "usage": {"input_tokens": toks, "output_tokens": 0}}

    @r.post("/chat/completions")
    async def chat(request: Request):
        fn = getattr(readout.backend, "passthrough", None)
        if fn is None:
            raise http_error(501, "this backend has no chat passthrough")
        code, data = await fn(await request.json())
        return Response(data, status_code=code, media_type="application/json")

    return r


async def answer_request(readout: Readout, body: SystemOneRequest) -> tuple[dict[str, dict], int]:
    """The answers and prompt tokens of one request. A bad question is a 422; a model server that fails is a 502."""
    INFLIGHT.inc()
    try:
        return await readout.answer_all(body.state, {k: q.model_dump() for k, q in body.questions.items()})
    except BadQuestion as e:
        REQS.labels("422").inc()
        raise http_error(422, str(e)) from e
    except ReadoutIncomplete as e:
        REQS.labels("502").inc()
        raise http_error(502, f"readout incomplete: {e}") from e
    except BackendError as e:
        REQS.labels("502").inc()
        raise http_error(502, str(e)[:300]) from e
    finally:
        INFLIGHT.dec()


def record(rid: str, answers: dict[str, dict], toks: int, dt: float) -> None:
    """Metrics and the one-line JSON log of a served request."""
    LAT.observe(dt)
    REQS.labels("200").inc()
    TOKENS.inc(toks)
    for a in answers.values():
        QUESTIONS.labels(a["type"]).inc()
        READ_ONCE.inc(a.get("read_once", 0))
    ops = {q: a.get("choice") for q, a in answers.items() if "choice" in a}
    log.info(json.dumps({"id": rid, "questions": len(answers), "tokens": toks, "ms": round(dt * 1000), "ops": ops}))


def build_app(readout: Readout, *, token: str = "", model_dir: str = "", backend_name: str = "", extra_version: dict | None = None) -> FastAPI:
    version = version_info(readout, backend_name=backend_name, model_dir=model_dir, extra=extra_version)
    p = readout.profile
    model_string = f"{version['model_dir']} T={p.temp} noul={p.noul_t},{p.noul_bias} profile={p.name} openjev-server@{__version__}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        close = getattr(readout.backend, "aclose", None)
        if close:
            await close()

    app = FastAPI(title="openjev-server", version=__version__, lifespan=lifespan, docs_url="/docs", redoc_url=None)

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request, exc: StarletteHTTPException):
        message = exc.detail["message"] if isinstance(exc.detail, dict) else str(exc.detail)
        return error_response(exc.status_code, message, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request, exc: RequestValidationError):
        detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return error_response(422, "; ".join(f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in detail), detail)

    app.include_router(ops_router(readout, backend_name))
    app.include_router(api_router(readout, version, model_string, token))
    return app
