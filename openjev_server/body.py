"""Request bodies the way clients of the hosted Jev API send them: one compression layer (gzip or zstd), at most 32 MiB as sent and
after decompression, and one image through `assets`, as base64 in the JSON or as a part of a multipart/form-data body."""

from __future__ import annotations

import base64
import email
import email.policy
import gzip
import io
import json
import zlib
from typing import Any

import zstandard
from fastapi import Request

MAX_BODY = 32 * 1024 * 1024  # the hosted API's limit, as sent and after decompression
IMAGE_TYPES = {b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg"}  # file signature -> type, for the formats the model reads


class BadBody(ValueError):
    """A body the server cannot read. `status` is the HTTP answer: 400 malformed, 413 too large, 415 unsupported encoding."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


async def read_body(request: Request) -> dict[str, Any]:
    """The request's JSON object, decoded the way the hosted API's clients send it; an image sent through `assets` becomes state.image."""
    data = inflate(await read_capped(request), request.headers.get("content-encoding"))
    content_type = request.headers.get("content-type", "")
    files: dict[str, bytes] = {}
    if content_type.split(";")[0].strip().lower() == "multipart/form-data":
        data, files = split_multipart(data, content_type)
    body = parse_object(data)
    if files or "assets" in body:
        attach_image(body, files)
    return body


async def read_capped(request: Request) -> bytes:
    """The body as sent, refused as soon as it passes MAX_BODY, so no more than that is ever held."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY:
            raise BadBody(413, "the body is over 32 MiB")
    return bytes(body)


def inflate(raw: bytes, encoding: str | None) -> bytes:
    """The body as sent -> its JSON or multipart bytes. One compression layer; never inflated more than one byte past MAX_BODY."""
    enc = (encoding or "identity").strip().lower()
    if enc == "identity":
        return raw
    if enc not in ("gzip", "x-gzip", "zstd"):
        raise BadBody(415, f"unsupported Content-Encoding {encoding!r}: use gzip, x-gzip or zstd, one layer")
    source = io.BytesIO(raw)
    reader = zstandard.ZstdDecompressor().stream_reader(source, read_across_frames=True) if enc == "zstd" else gzip.GzipFile(fileobj=source)
    try:
        data = reader.read(MAX_BODY + 1)
    except (OSError, EOFError, zlib.error, zstandard.ZstdError) as e:
        raise BadBody(400, f"malformed {enc} body: {e}") from None
    if len(data) > MAX_BODY:
        raise BadBody(413, "the body is over 32 MiB after decompression")
    return data


def split_multipart(data: bytes, content_type: str) -> tuple[bytes, dict[str, bytes]]:
    """multipart/form-data (RFC 7578) -> (the JSON of its first part, named `request`; {name: bytes} of the other parts)."""
    msg = email.message_from_bytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + data, policy=email.policy.HTTP)
    parts = [(p.get_param("name", header="content-disposition"), p.get_payload(decode=True)) for p in msg.iter_parts()]
    names = [n for n, _ in parts]
    if not names or names[0] != "request" or not all(names) or len(set(names)) < len(names):
        raise BadBody(400, "a multipart body is the JSON as its first part, named request, then one uniquely named part per image")
    return parts[0][1] or b"", dict(parts[1:])


def parse_object(data: bytes) -> dict[str, Any]:
    """The body's JSON, which must be an object. An empty body reads as {}, so validation names what is missing."""
    try:
        body = json.loads(data or b"{}")
    except ValueError as e:
        raise BadBody(400, f"malformed JSON body: {e}") from None
    if not isinstance(body, dict):
        raise BadBody(400, "the body must be a JSON object")
    return body


def attach_image(body: dict[str, Any], files: dict[str, bytes]) -> None:
    """The `assets` map -> its one image as state.image, a data URL, on the path a screenshot state takes. The model reads one
    leading image, so the reference must be the whole state or one whole top-level value of an object state, used once."""
    assets = body.pop("assets", None)
    if not isinstance(assets, dict) or len(assets) != 1:
        raise BadBody(400, "this model reads exactly one image per request: `assets` needs exactly one entry")
    ((ref, asset),) = assets.items()
    raw = asset_bytes(ref, asset if isinstance(asset, dict) else {}, files)
    mime = next((kind for magic, kind in IMAGE_TYPES.items() if raw.startswith(magic)), None)
    if mime is None:
        raise BadBody(400, f"asset {ref!r} is not a PNG or JPEG file")
    state = body.get("state")
    key = next((k for k, v in state.items() if v == ref), None) if isinstance(state, dict) else None
    rest = {k: v for k, v in state.items() if k != key} if key is not None else {}
    if uses(body.get("questions"), ref) or uses(state, ref) != 1 or (key is None and state != ref) or {"image", "screenshot"} & set(rest):
        raise BadBody(
            400,
            "this model reads one image: its reference must be the whole state or one whole top-level value of an object state "
            "(with no other image or screenshot field), used once and not in instructions or criteria",
        )
    body["state"] = {**rest, "image": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}


def asset_bytes(ref: str, asset: dict[str, Any], files: dict[str, bytes]) -> bytes:
    """The one asset's bytes: its multipart part, or its data_base64. Every multipart part must belong to it."""
    part = asset.get("part")
    try:
        raw = files[part] if part is not None else base64.b64decode(asset.get("data_base64") or "", validate=True)
    except (KeyError, TypeError, ValueError):
        raise BadBody(400, f"asset {ref!r}: its multipart part is missing or its data_base64 is not a valid base64 string") from None
    if stray := sorted(set(files) - {part}):
        raise BadBody(400, f"multipart parts {stray} belong to no asset")
    return raw


def uses(value: Any, ref: str) -> int:
    """How often `ref` occurs in the strings of `value`; object keys are not searched, as on the hosted API."""
    if isinstance(value, str):
        return value.count(ref)
    if isinstance(value, dict):
        value = list(value.values())
    return sum(uses(v, ref) for v in value) if isinstance(value, list) else 0
