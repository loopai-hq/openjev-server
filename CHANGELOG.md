# Changelog

## 0.3.0 (2026-11-24)
- OpenJev 1.1 and OpenJev 1.1 FP4 in the README's models table and quick start. 1.1 keeps v1's calibration, so the `openjev` profile serves both.
- Read twice: the `repeat` profile field, `--repeat`, or the helper's `READOUT_REPEAT=1`; off by default. Each text prompt goes out twice in one request, a blank line between the copies; screenshot prompts are read once. A doubled prompt that would not fit the backend's window goes out once, and the answer carries `read_once`; `/metrics` counts them as `openjev_read_once_total`. `/v1/version` reports `flags.repeat`.
- The vLLM backend reads the window from `GET /v1/models` (`max_model_len`) at start, now also when `--served-model-name` is given, and reports it in the startup probe.
- Requests written for the hosted Jev API (Jev 1.14) on `POST /v1/systemone`, in the new `body.py`: gzip, x-gzip and zstd bodies (415 for other encodings), 32 MiB per body as sent and after decompression (413), one image through `assets` (base64 or multipart), optional `instructions` with a neutral default, one-level scores answered without the model, the requested `model` echoed, and `GET /v1/models`.
- Bounded work per request body: a multipart body is split on its boundary without a line parser and holds at most two parts (the JSON and one image), each with at most 8 KiB of headers; a compressed body is one gzip member or one zstd frame (window up to 32 MiB), inflated in steps of at most 1 MiB, and a truncated one is a 400; JSON nested too deeply for the parser is a 400.
- Every error is JSON with this server's `error` object and the hosted API's `detail`. Malformed JSON and non-object bodies are 400 (they were 422).
- Requests that 0.2.0 accepted send the same prompts, byte for byte: a new regression test holds them to the released helper's. `tests/test_compat.py` also holds read twice to OpenJev 1.1's helper.
- New dependency: `zstandard` (the standard library reads zstd only from Python 3.14).

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
