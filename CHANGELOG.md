# Changelog

## 0.3.0 (2026-11-24)
- OpenJev 1.1 and OpenJev 1.1 FP4 in the README's models table and quick start. 1.1 keeps v1's calibration, so the `openjev` profile serves both.
- Read twice: the `repeat` profile field, `--repeat`, or the helper's `READOUT_REPEAT=1`; off by default. Each text prompt goes out twice in one request, a blank line between the copies; screenshot prompts are read once. A doubled prompt that would not fit the backend's window goes out once, and the answer carries `read_once`; `/metrics` counts them as `openjev_read_once_total`. `/v1/version` reports `flags.repeat`.
- The vLLM backend reads the window from `GET /v1/models` (`max_model_len`) at start, now also when `--served-model-name` is given, and reports it in the startup probe.
- Requests written for the hosted Jev API (Jev 1.14) on `POST /v1/systemone`, in the new `body.py`: gzip, x-gzip and zstd bodies (415 for other encodings), 32 MiB per body as sent and after decompression (413), one image through `assets` (base64 or multipart), optional `instructions` with a neutral default, one-level scores answered without the model, the requested `model` echoed, and `GET /v1/models`.
- Bounded work per request body: a multipart body is split on its boundary without a line parser and holds at most two parts (the JSON and one image), each with at most 8 KiB of headers; a compressed body is one gzip member or one zstd frame (window up to 32 MiB), inflated in steps of at most 1 MiB, and a truncated one is a 400; JSON nested too deeply for the parser is a 400.
- Every error is JSON with this server's `error` object and the hosted API's `detail`. A 422's `detail` items carry only `loc`, `msg` and `type`: FastAPI's `input` and `ctx` are no longer sent back. Malformed JSON and non-object bodies are 400 (they were 422).
- `openjev_requests_total` still counts only the requests that reach the readout (200, 422 for a bad question, 502). Body errors (400, 413, 415), the cross-site 403 and validation 422s are not counted, as validation errors were not in 0.2.0.
- Only JSON (`application/json`, `application/*+json`) and `multipart/form-data` bodies are read; anything else is a 415, as it was refused (422) by 0.2.0. Set `--token` whenever a browser can reach the port.
- `/v1/prewarm` reads its body like `/v1/systemone`: after the bearer check, capped at 32 MiB (FastAPI parsed it whole before the check).
- Without `--token`, every `/v1` route refuses a request that a browser marks `Sec-Fetch-Site: cross-site` (403): a multipart/form-data post needs no CORS preflight, so a web page could otherwise make a visitor's browser post work to the server. SDKs and curl send no such header; with a token the rule does not apply.
- `/v1/version.readout_sha256` also hashes `body.py`, which writes an image from `assets` into the state the model reads.
- Requests that 0.2.0 answered send the same prompts, byte for byte, with two exceptions: a body over 32 MiB is now a 413, and a request with an `assets` field, which 0.2.0 ignored, now sends its image or is a 400. A new regression test holds the prompts to the released helper's; `tests/test_compat.py` also holds read twice to OpenJev 1.1's helper.
- Read twice counts tokens in a worker thread, and not at all for a doubled prompt with more than 128 characters per window token, which cannot fit. A named model whose backend has no `GET /v1/models` starts again, with window 0 (every prompt read once).
- Multipart part headers go through the compat32 header parser, which hostile headers do not crash.
- New dependency: `zstandard` (the standard library reads zstd only from Python 3.14).
- Demos: the source names in `demos/out/recorded_gpu/colocated-browser-provenance.json` are neutral (`jev_ultrafast/…`, `workflow_demo/…`, `run.py`); their hashes and the rest of the recording are unchanged.

## 0.2.0 (2026-09-22)
- Modules by job: `profile.py` (calibration constants), `prompt.py` (state text and prompt layouts), `readout.py` (scores to answers), `backends/base.py` (the backend contract), `backends/make_backend` (the only place that knows backend names). Answers unchanged: the compat tests against the released helper still hold.
- Server: `/v1` routes on one router with bearer auth as a router dependency; operations endpoints on their own router.
- vLLM backend: `VllmOptions` replaces ten constructor arguments; one response parser for the chat and raw-completion paths; tests against a fake OpenAI-compatible server.
- `/v1/version.readout_sha256` now hashes `prompt.py` + `readout.py` (the answer-determining code).
- Tests fetch the public OpenJev tokenizer and the released helper from the Hub when no local path is set, so CI needs nothing local.
- Landing page: links open in the top window (the Space iframe could not frame huggingface.co).

## 0.1.0 (2026-09-22)
- First release: the OpenJev decision API as an installable server. vLLM and MLX backends, profiles, calibration fitter, bench, metrics, bearer auth.
- Answers identical to the helper shipped with the OpenJev model (`helper/shim.py`) under the `openjev` profile.
- Any-model support: label tokenization detection, exact-readout detection, raw prompting without a chat template, assistant prefix for reasoning models, startup probe that refuses to serve noise.
- MLX: the shared part of a request is read once and reused across its questions.
