"""Request bodies the way clients of the hosted Jev API send them: one compression layer (gzip or zstd), at most 32 MiB as sent and
after decompression, and one image through `assets`, as base64 in the JSON or as a part of a multipart/form-data body. Each step is
linear in the body's size and bounded in memory, so a small body cannot make the server inflate, allocate or parse much more."""

from __future__ import annotations

import base64
import email.message
import email.parser
import email.policy
import io
import json
import zlib
from typing import Any

import zstandard
from fastapi import Request

MAX_BODY = 32 * 1024 * 1024  # the hosted API's limit, as sent and after decompression; also the largest zstd window decoded
STEP = 1024 * 1024  # inflated bytes read at a time, so memory grows with what a body holds rather than with the limit
MAX_PARTS = 2  # a multipart body holds the JSON and at most one image
MAX_PART_HEADERS = 8 * 1024  # bytes in the header block of one multipart part
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
            raise BadBody(413, f"the body is over {MAX_BODY >> 20} MiB")
    return bytes(body)


def inflate(raw: bytes, encoding: str | None) -> bytes:
    """The body as sent -> its JSON or multipart bytes. One compression layer, never inflated much past MAX_BODY."""
    enc = (encoding or "identity").strip().lower()
    if enc == "identity":
        return raw
    if enc not in ("gzip", "x-gzip", "zstd"):
        raise BadBody(415, f"unsupported Content-Encoding {encoding!r}: use gzip, x-gzip or zstd, one layer")
    try:
        return unzstd(raw) if enc == "zstd" else gunzip(raw)
    except (zlib.error, zstandard.ZstdError) as e:
        raise BadBody(400, f"malformed {enc} body: {e}") from None


def gunzip(raw: bytes) -> bytes:
    """One gzip member, inflated by zlib to at most MAX_BODY + 1 bytes; zlib grows its output in steps. A second member is
    refused: Python's gzip reader loops once per member, and 32 MiB of empty members kept it busy for 26 s."""
    inflater = zlib.decompressobj(zlib.MAX_WBITS | 16)  # gzip framing
    data = inflater.decompress(raw, MAX_BODY + 1)
    if len(data) > MAX_BODY:
        raise inflated_too_far()
    if not inflater.eof:
        raise zlib.error("truncated")
    if inflater.unused_data:
        raise zlib.error("more than one gzip member")
    return data


def unzstd(raw: bytes) -> bytes:
    """One zstd frame with a window of at most MAX_BODY (a frame of a few bytes may ask for any window). Its size is measured in
    1 MiB steps first, so a bomb stops at the limit; only a frame known to fit is decoded whole, which also shows whether it is
    complete: the zstd library does not report a frame cut short while streaming."""
    decoder = zstandard.ZstdDecompressor(max_window_size=MAX_BODY)
    reader, size = decoder.stream_reader(io.BytesIO(raw)), 0
    while chunk := reader.read(STEP):
        size += len(chunk)
        if size > MAX_BODY:
            raise inflated_too_far()
    frame = decoder.decompressobj()
    data = frame.decompress(raw)
    if not frame.eof:
        raise zstandard.ZstdError("truncated")
    if frame.unused_data:
        raise zstandard.ZstdError("more than one zstd frame")
    return data


def inflated_too_far() -> BadBody:
    return BadBody(413, f"the body is over {MAX_BODY >> 20} MiB after decompression")


def split_multipart(data: bytes, content_type: str) -> tuple[bytes, dict[str, bytes]]:
    """multipart/form-data (RFC 7578) -> (the JSON of its first part, named `request`; {name: bytes} of the other parts). The parts
    are counted and split on the boundary in linear time; only their header blocks, at most 8 KiB each, go through a parser."""
    delimiter = b"--" + multipart_boundary(content_type)
    if data.count(delimiter) > MAX_PARTS + 1:  # each part opens with the delimiter, and one more closes the body
        raise BadBody(400, "a multipart body holds the JSON and at most one image")
    pieces = data.split(delimiter)[1:]  # what comes before the first delimiter is a preamble
    end = next((i for i, piece in enumerate(pieces) if piece.startswith(b"--")), None)
    if end is None:
        raise BadBody(400, "the multipart body does not end with its closing boundary")
    parts = [read_part(piece) for piece in pieces[:end]]
    names = [name for name, _ in parts]
    if not names or names[0] != "request" or not all(names) or len(set(names)) < len(names):
        raise BadBody(400, "a multipart body is the JSON as its first part, named request, then one uniquely named part per image")
    return parts[0][1], dict(parts[1:])


def multipart_boundary(content_type: str) -> bytes:
    """The boundary parameter of a multipart Content-Type: 1 to 70 characters (RFC 2046)."""
    header = email.message.Message()
    header["content-type"] = content_type
    boundary = header.get_param("boundary")
    if not isinstance(boundary, str) or not 0 < len(boundary) <= 70:
        raise BadBody(400, "a multipart body needs a boundary of 1 to 70 characters in its Content-Type")
    return boundary.encode("latin-1")


def read_part(piece: bytes) -> tuple[str | None, bytes]:
    """One part as it sits between two boundaries -> (its form-data name, its content)."""
    piece = piece.lstrip(b" \t")  # transport padding after the boundary
    end = piece.find(b"\r\n\r\n", 0, 2 + MAX_PART_HEADERS + 4)  # the blank line after the header block
    if not piece.startswith(b"\r\n") or end < 0 or not piece.endswith(b"\r\n"):
        raise BadBody(400, f"a multipart part is a line break, at most {MAX_PART_HEADERS} bytes of headers, a blank line, its content and a line break")
    headers = email.parser.BytesHeaderParser(policy=email.policy.HTTP).parsebytes(piece[2:end])
    return headers.get_param("name", header="content-disposition"), piece[end + 4 : -2]


def parse_object(data: bytes) -> dict[str, Any]:
    """The body's JSON, which must be an object. An empty body reads as {}, so validation names what is missing."""
    try:
        body = json.loads(data or b"{}")
    except RecursionError:
        raise BadBody(400, "malformed JSON body: nested too deeply") from None
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
    """How often `ref` occurs in the strings of `value`; object keys are not searched, as on the hosted API. Walked with a stack:
    the JSON parser accepts nesting deeper than Python's recursion limit."""
    count, todo = 0, [value]
    while todo:
        v = todo.pop()
        if isinstance(v, str):
            count += v.count(ref)
        elif isinstance(v, dict):
            todo.extend(v.values())
        elif isinstance(v, list):
            todo.extend(v)
    return count
