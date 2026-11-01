"""
COMPASS AI Local Model Engine API Server (`app.py`).
Provides 100% MozartAPI v2 compatibility (`POST /model_info`, `POST /generate` with `X-Generation-Reason`)
plus local hardware diagnostics (`GET /health`, `GET /hardware_info`), live token-generation progress
(`GET /generation_progress`), and live AppData model rescanning (`POST /rescan_models`).
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import hmac
import json
import logging
import math
import os
import re
import shutil
import struct
import tempfile
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TypeVar
import urllib.parse
import uuid

from collections.abc import AsyncGenerator
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from python_multipart.exceptions import (
    FormParserError,
    MultipartParseError,
    ParseError,
    QuerystringParseError,
)
from starlette.datastructures import FormData, Headers, UploadFile as StarletteUploadFile
from starlette.formparsers import (
    FormParser,
    MultiPartException,
    MultiPartParser,
    MultipartPart,
)
from starlette.requests import ClientDisconnect

from model import ModelController, resolve_physical_models_dir_for_shell
from mortm.models.modules.progress import GenerationCancelledError
from rapper import AbstractModelRapper, GenerateMeta

logger = logging.getLogger("compass_ai")

MAX_MIDI_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB limit per MIDI file
MAX_GENERATE_BODY_BYTES = 34 * 1024 * 1024  # 34 MB total multipart ceiling (3 x 10MB MIDI + meta) (`[R21-HIGH-01]`)
MAX_GENERATE_URLENCODED_BYTES = 1 * 1024 * 1024  # 1 MB urlencoded ceiling (`[R21-HIGH-01]`)
MAX_GENERATE_MULTIPART_FILES = 8
MAX_GENERATE_MULTIPART_FIELDS = 16
MAX_GENERATE_PART_BYTES = 1024 * 1024  # 1 MB text-part ceiling (preserves >64KB meta -> 413 & dup 100KB -> 400)
MAX_SMF_QUARTER_NOTES = 4160  # 1,040 bars of 4/4 (1,996,800 ticks at 480 PPQN) (`[R4-MED-06]`)
ALLOWED_MIDI_EXTENSIONS = {".mid", ".midi"}
_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_JOB_ID_FULL_CHARS_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,4096}$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost", "testclient"})
_ALLOWED_GENERATE_FORM_KEYS = frozenset(
    {"meta", "meta_json", "job_id", "past_midi", "conditions_midi", "future_midi"}
)
_GENERATE_TEXT_FORM_KEYS = frozenset({"meta", "meta_json", "job_id"})
_GENERATE_FILE_FORM_KEYS = frozenset({"past_midi", "conditions_midi", "future_midi"})
_ALLOWED_FORM_CHARSETS = frozenset(
    {b"utf-8", b"utf8", b"us-ascii", b"ascii", b"latin-1", b"iso-8859-1"}
)
_HTTP_TCHAR_BYTES = frozenset(
    b"!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
_BOUNDARY_UNQUOTED_BYTES = _HTTP_TCHAR_BYTES | frozenset(b"/:?=")


def _quote_mime_param_bytes(val: bytes) -> bytes:
    return b'"' + val.replace(b"\\", b"\\\\").replace(b'"', b'\\"') + b'"'


def _parse_strict_mime_header(
    raw_header: bytes | str | None,
    *,
    require_slash: bool,
    allow_utf8_quoted: bool = False,
) -> Tuple[bytes, Dict[bytes, bytes]]:
    """
    Deterministic O(N) zero-backtracking RFC 7230 / RFC 7578 MIME option parser (`[R24-HIGH-01]`).
    - Eliminates `email.message.Message.get_params()` / `collapse_rfc2231_value` so untrusted
      RFC 2231 `param*=charset'lang'value` (`punycode`, `unicode_escape`, `utf-7`, etc.) and
      RFC 2822 `(comment)` injections can NEVER execute Python codecs on the asyncio event loop.
    - Rejects unclosed quotes/escapes, trailing bytes after closing quotes, non-empty `;`-segments
      missing `=`, any `*` in parameter names (`param*=`, `param*0=`), and duplicate case-insensitive
      parameter names (`charset=punycode; charset=utf-8`).
    """
    if raw_header is None:
        raise ValueError("Missing MIME header.")
    if isinstance(raw_header, str):
        try:
            raw_bytes = raw_header.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ValueError("Non-Latin-1 character in MIME header.") from exc
    elif isinstance(raw_header, (bytes, bytearray)):
        raw_bytes = bytes(raw_header)
    else:
        raise ValueError("Invalid MIME header type.")

    n = len(raw_bytes)
    if n == 0 or n > 16 * 1024:
        raise ValueError("MIME header length out of bounds.")

    semi_pos = raw_bytes.find(b";")
    head_end = n if semi_pos < 0 else semi_pos
    main_raw = raw_bytes[:head_end].strip(b" \t")
    if not main_raw:
        raise ValueError("Empty MIME type/disposition.")

    if require_slash:
        slash_idx = main_raw.find(b"/")
        if (
            slash_idx <= 0
            or slash_idx >= len(main_raw) - 1
            or b"/" in main_raw[slash_idx + 1 :]
            or any(b not in _HTTP_TCHAR_BYTES for b in main_raw[:slash_idx])
            or any(b not in _HTTP_TCHAR_BYTES for b in main_raw[slash_idx + 1 :])
        ):
            raise ValueError("Invalid media-type token.")
    else:
        if any(b not in _HTTP_TCHAR_BYTES for b in main_raw):
            raise ValueError("Invalid disposition-type token.")

    main_type = main_raw.lower()
    if semi_pos < 0:
        return main_type, {}

    params: Dict[bytes, bytes] = {}
    pos = semi_pos + 1
    semi_count = 1
    while pos < n:
        while pos < n and raw_bytes[pos] in (0x20, 0x09):
            pos += 1
        if pos >= n:
            break
        if raw_bytes[pos] == 0x3B:
            semi_count += 1
            if semi_count > 16:
                raise ValueError("Too many parameter delimiters in MIME header.")
            pos += 1
            continue

        name_start = pos
        while pos < n and raw_bytes[pos] not in (0x3D, 0x3B, 0x20, 0x09):
            pos += 1
        raw_param_name = raw_bytes[name_start:pos]
        while pos < n and raw_bytes[pos] in (0x20, 0x09):
            pos += 1
        if pos >= n or raw_bytes[pos] != 0x3D:
            raise ValueError("MIME parameter missing '='.")
        if (
            not raw_param_name
            or len(raw_param_name) > 128
            or 0x2A in raw_param_name
            or any(b not in _HTTP_TCHAR_BYTES for b in raw_param_name)
        ):
            raise ValueError("Invalid or RFC 2231 MIME parameter name.")

        param_key = raw_param_name.lower()
        if param_key in params or len(params) >= 16:
            raise ValueError("Duplicate or excessive MIME parameter.")

        pos += 1  # consume '='
        while pos < n and raw_bytes[pos] in (0x20, 0x09):
            pos += 1

        if pos < n and raw_bytes[pos] == 0x22:  # quoted-string
            pos += 1
            val_buf = bytearray()
            closed = False
            while pos < n:
                b = raw_bytes[pos]
                if b == 0x22:
                    pos += 1
                    closed = True
                    break
                if b == 0x5C:
                    pos += 1
                    if pos >= n:
                        raise ValueError("Unclosed escape in quoted MIME parameter.")
                    esc_b = raw_bytes[pos]
                    if (
                        (esc_b < 0x20 and esc_b != 0x09)
                        or esc_b == 0x7F
                        or (not allow_utf8_quoted and esc_b > 0x7E)
                    ):
                        raise ValueError("Invalid escaped byte in quoted MIME parameter.")
                    val_buf.append(esc_b)
                    pos += 1
                else:
                    if (
                        (b < 0x20 and b != 0x09)
                        or b == 0x7F
                        or (not allow_utf8_quoted and b > 0x7E)
                    ):
                        raise ValueError("Invalid byte in quoted MIME parameter.")
                    val_buf.append(b)
                    pos += 1
            if not closed:
                raise ValueError("Unclosed quoted-string MIME parameter.")
            while pos < n and raw_bytes[pos] in (0x20, 0x09):
                pos += 1
            if pos < n:
                if raw_bytes[pos] != 0x3B:
                    raise ValueError("Unexpected trailing data after quoted MIME parameter.")
                semi_count += 1
                if semi_count > 16:
                    raise ValueError("Too many parameter delimiters in MIME header.")
                pos += 1
            param_val = bytes(val_buf)
        else:
            val_start = pos
            while pos < n and raw_bytes[pos] != 0x3B:
                pos += 1
            param_val = raw_bytes[val_start:pos].rstrip(b" \t")
            if not param_val:
                raise ValueError("Empty unquoted MIME parameter value.")
            allowed_unquoted = (
                _BOUNDARY_UNQUOTED_BYTES if param_key == b"boundary" else _HTTP_TCHAR_BYTES
            )
            if any(b not in allowed_unquoted for b in param_val):
                raise ValueError("Invalid byte in unquoted MIME parameter value.")
            if pos < n and raw_bytes[pos] == 0x3B:
                semi_count += 1
                if semi_count > 16:
                    raise ValueError("Too many parameter delimiters in MIME header.")
                pos += 1

        if param_key == b"filename":
            if param_val[1:3] == b":\\" or param_val[:2] == b"\\\\":
                param_val = param_val.split(b"\\")[-1]
        params[param_key] = param_val

    return main_type, params


async def _close_spooled_uploads(form_data: Any, direct_uploads: Tuple[Any, ...] = ()) -> None:
    seen_ids = set()
    candidates: List[Any] = []
    if form_data is not None and hasattr(form_data, "multi_items"):
        try:
            for _k, val in form_data.multi_items():
                if isinstance(val, (StarletteUploadFile, UploadFile)) or callable(
                    getattr(val, "close", None)
                ):
                    candidates.append(val)
        except Exception:
            pass
    for item in direct_uploads:
        if item is not None:
            candidates.append(item)
    for item in candidates:
        obj_id = id(item)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)
        inner_file = getattr(item, "file", None)
        if inner_file is not None and id(inner_file) not in seen_ids:
            seen_ids.add(id(inner_file))
            inner_close = getattr(inner_file, "close", None)
            if callable(inner_close) and not asyncio.iscoroutinefunction(inner_close):
                try:
                    inner_close()
                except Exception:
                    pass
        close_fn = getattr(item, "close", None)
        if callable(close_fn):
            try:
                res = close_fn()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass


class _BoundedMultiPartParser(MultiPartParser):
    """
    Bounded Starlette `MultiPartParser` (`[R21-HIGH-01]`, `oop-extensible-craft` §1.2/§4.2):
    - Caps per-part header bytes (`16 KB`) and header line count (`16`) against quadratic header stalls.
    - Rejects unknown field names before allocating `SpooledTemporaryFile`.
    - Provides synchronous zero-await OS handle closure (`close_all_files_sync`) so rolled `%TEMP%` files
      are immediately closed and unlinked even on `ClientDisconnect`, `HTTPException(413)`, or `CancelledError`.
    """

    MAX_PART_HEADER_BYTES = 16 * 1024
    MAX_PART_HEADERS_COUNT = 16

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._current_part_header_bytes = 0
        self._current_part_seen_headers: set[bytes] = set()
        self._saw_any_part = False
        self._saw_multipart_end = False
        self._in_open_part = False

    def on_part_begin(self) -> None:
        self._current_part_header_bytes = 0
        self._current_part_seen_headers.clear()
        self._saw_any_part = True
        self._in_open_part = True
        super().on_part_begin()

    def on_part_end(self) -> None:
        super().on_part_end()
        self._in_open_part = False

    def on_end(self) -> None:
        self._saw_multipart_end = True
        super().on_end()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._current_part_header_bytes += max(0, end - start)
        if self._current_part_header_bytes > self.MAX_PART_HEADER_BYTES:
            raise MultiPartException("Part header exceeded maximum size.")
        super().on_header_field(data, start, end)

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._current_part_header_bytes += max(0, end - start)
        if self._current_part_header_bytes > self.MAX_PART_HEADER_BYTES:
            raise MultiPartException("Part header exceeded maximum size.")
        super().on_header_value(data, start, end)

    def on_header_end(self) -> None:
        if len(self._current_part.item_headers) >= self.MAX_PART_HEADERS_COUNT:
            raise MultiPartException("Too many headers in multipart part.")
        raw_field = self._current_partial_header_name
        if (
            not raw_field
            or raw_field.startswith(b"--")
            or any(b not in _HTTP_TCHAR_BYTES for b in raw_field)
        ):
            raise MultiPartException("Invalid header field name in multipart part.")
        field_lower = raw_field.lower()
        if field_lower in (b"content-disposition", b"content-type"):
            if field_lower in self._current_part_seen_headers:
                raise MultiPartException("Duplicate header in multipart part.")
            self._current_part_seen_headers.add(field_lower)
        if field_lower == b"content-type":
            try:
                _part_mt, part_ct_params = _parse_strict_mime_header(
                    self._current_partial_header_value,
                    require_slash=True,
                    allow_utf8_quoted=False,
                )
            except ValueError as exc:
                raise MultiPartException("Invalid Content-Type header in multipart part.") from exc
            if b"charset" in part_ct_params:
                if part_ct_params[b"charset"].strip().lower() not in _ALLOWED_FORM_CHARSETS:
                    raise HTTPException(
                        status_code=400,
                        detail="サポートされていない文字エンコーディングです。",
                    )
        super().on_header_end()

    def on_headers_finished(self) -> None:
        try:
            disp_type, options = _parse_strict_mime_header(
                self._current_part.content_disposition,
                require_slash=False,
                allow_utf8_quoted=True,
            )
        except ValueError as exc:
            raise MultiPartException(
                "Invalid Content-Disposition header in multipart part."
            ) from exc
        if disp_type != b"form-data":
            raise MultiPartException("Invalid Content-Disposition type in multipart part.")
        if b"charset" in options:
            if options[b"charset"].strip().lower() not in _ALLOWED_FORM_CHARSETS:
                raise HTTPException(
                    status_code=400,
                    detail="サポートされていない文字エンコーディングです。",
                )
        raw_name = options.get(b"name")
        if raw_name is None:
            raise MultiPartException('The Content-Disposition header field "name" must be provided.')
        charset = self._charset or "utf-8"
        try:
            decoded_name = raw_name.decode(charset)
        except (UnicodeDecodeError, LookupError):
            decoded_name = raw_name.decode("latin-1")
        if decoded_name not in _ALLOWED_GENERATE_FORM_KEYS:
            raise HTTPException(
                status_code=400,
                detail="未対応のフォームフィールドが指定されています。",
            )
        canonical_cd = b"form-data; name=" + _quote_mime_param_bytes(raw_name)
        if b"filename" in options:
            raw_filename = options[b"filename"]
            if len(raw_filename) > 1024:
                raise MultiPartException("Filename in Content-Disposition exceeded maximum length.")
            canonical_cd += b"; filename=" + _quote_mime_param_bytes(raw_filename)
        self._current_part.content_disposition = canonical_cd
        super().on_headers_finished()

    def close_all_files_sync(self) -> None:
        raw_objs: List[Any] = list(getattr(self, "_files_to_close_on_error", ()))
        cur_part = getattr(self, "_current_part", None)
        if cur_part is not None and getattr(cur_part, "file", None) is not None:
            raw_objs.append(cur_part.file)
        for _k, val in getattr(self, "items", ()):
            raw_objs.append(val)
        for part, _chunk in getattr(self, "_file_parts_to_write", ()):
            if getattr(part, "file", None) is not None:
                raw_objs.append(part.file)
        for part in getattr(self, "_file_parts_to_finish", ()):
            if getattr(part, "file", None) is not None:
                raw_objs.append(part.file)

        seen: set[int] = set()
        for obj in raw_objs:
            if obj is None:
                continue
            inner = getattr(obj, "file", None)
            for target in (inner, obj):
                if target is None or id(target) in seen:
                    continue
                seen.add(id(target))
                close_fn = getattr(target, "close", None)
                if callable(close_fn) and not asyncio.iscoroutinefunction(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass

        for attr_name in (
            "_files_to_close_on_error",
            "items",
            "_file_parts_to_write",
            "_file_parts_to_finish",
        ):
            coll = getattr(self, attr_name, None)
            if isinstance(coll, list):
                coll.clear()
        self._current_part = MultipartPart()

_T = TypeVar("_T")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
        return val if math.isfinite(val) else default
    except ValueError:
        return default


# Dedicated isolated thread pools ([HIGH-07]):
# - `_INFER_EXECUTOR` (max_workers=1) serializes heavy PyTorch inference without starving metadata reads.
# - `_META_EXECUTOR` (max_workers=4) serves `/health`, `/hardware_info`, `/model_info`, `/rescan_models` in <5ms.
_INFER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compass-infer")
_META_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="compass-meta")
_GEN_QUEUE_SEMAPHORE = threading.BoundedSemaphore(
    max(1, _env_int("COMPASS_MAX_QUEUED_GENERATIONS", 4))
)


async def _run_meta(fn: Callable[..., _T], *args: Any) -> _T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_META_EXECUTOR, fn, *args)


async def _run_infer(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    loop = asyncio.get_running_loop()
    bound = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(_INFER_EXECUTOR, bound)


class SlidingWindowRateLimiter:
    """
    Thread-safe sliding-window in-memory rate limiter (`secure-coding-guard` §3, `[CRIT-04]`, `[R2-MED-06]`).
    Evicts ONLY expired buckets; pre-reserves the `'loopback'` slot so external LAN clients on
    `0.0.0.0` saturating `max_tracked_keys` can never lock out the local UI, while `'loopback'`
    remains strictly subject to `max_requests` per window.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0, max_tracked_keys: int = 1024):
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = float(window_seconds)
        self.max_tracked_keys = max(1, int(max_tracked_keys))
        self._lock = threading.Lock()
        self._buckets: Dict[str, Deque[float]] = {"loopback": deque()}

    def check_or_raise(self, client_key: str, error_detail: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            existing = self._buckets.get(client_key)
            if existing is not None:
                while existing and existing[0] <= cutoff:
                    existing.popleft()
            else:
                non_loopback_count = sum(1 for k in self._buckets if k != "loopback")
                if non_loopback_count >= self.max_tracked_keys:
                    stale_keys = []
                    for k, q in self._buckets.items():
                        while q and q[0] <= cutoff:
                            q.popleft()
                        if not q and k != "loopback":
                            stale_keys.append(k)
                    for sk in stale_keys:
                        self._buckets.pop(sk, None)
                    non_loopback_count = sum(1 for k in self._buckets if k != "loopback")
                    if non_loopback_count >= self.max_tracked_keys:
                        retry_after = max(1, int(math.ceil(self.window_seconds)))
                        raise HTTPException(
                            status_code=429,
                            detail=error_detail,
                            headers={"Retry-After": str(retry_after)},
                        )
                existing = deque()
                self._buckets[client_key] = existing

            if len(existing) >= self.max_requests:
                retry_after = max(1, int(math.ceil(self.window_seconds - (now - existing[0]))))
                raise HTTPException(
                    status_code=429,
                    detail=error_detail,
                    headers={"Retry-After": str(retry_after)},
                )

            existing.append(now)


app = FastAPI(
    title="COMPASS AI - Local MORTM 4.5 Inference Engine",
    version="4.5.0",
    description="Zero-FlashAttention Local Symbolic Music Foundation & SFT-Gen Inference Server",
)


@app.exception_handler(RequestValidationError)
async def _handle_request_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    await _close_spooled_uploads(getattr(_request, "_form", None))
    logger.warning("FastAPI request validation failed: %s", exc)
    for err in exc.errors():
        loc = err.get("loc", ())
        if isinstance(loc, (tuple, list)) and "job_id" in loc:
            return JSONResponse(
                status_code=400,
                content={"detail": "指定された job_id の形式が不正です。"},
            )
    return JSONResponse(
        status_code=422,
        content={"detail": "リクエストの入力形式またはパラメータ範囲が不正です。"},
    )


# Restrict CORS to local Chromium Frontend (`compass-app://studio`) & localhost DAW Controllers (`secure-coding-guard` §5)
_runtime_port = os.environ.get("COMPASS_PORT", "8000").strip() or "8000"


def _resolve_allowed_cors_origins(env_origins: Optional[str], runtime_port: str) -> list[str]:
    clean_port = (runtime_port or "8000").strip() or "8000"
    defaults = list(
        dict.fromkeys(
            [
                "compass-app://studio",
                "app://.",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                f"http://localhost:{clean_port}",
                f"http://127.0.0.1:{clean_port}",
            ]
        )
    )
    parsed = [o.strip() for o in env_origins.split(",") if o.strip()] if env_origins else []
    return parsed if parsed else defaults


_default_origins = _resolve_allowed_cors_origins(None, _runtime_port)
_allowed_origins = _resolve_allowed_cors_origins(os.environ.get("COMPASS_CORS_ORIGINS"), _runtime_port)
_is_wildcard_origin = "*" in _allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False if _is_wildcard_origin else False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Generation-Reason", "Content-Disposition", "Retry-After"],
)

model_controller = ModelController()

_rate_window_sec = max(1.0, _env_float("COMPASS_RATE_LIMIT_WINDOW_SEC", 60.0))
_generate_limiter = SlidingWindowRateLimiter(
    max_requests=_env_int("COMPASS_RATE_LIMIT_GENERATE", 30),
    window_seconds=_rate_window_sec,
)
_rescan_limiter = SlidingWindowRateLimiter(
    max_requests=_env_int("COMPASS_RATE_LIMIT_RESCAN", 30),
    window_seconds=_rate_window_sec,
)
_cancel_limiter = SlidingWindowRateLimiter(
    max_requests=_env_int("COMPASS_RATE_LIMIT_CANCEL", 60),
    window_seconds=_rate_window_sec,
)
_model_info_limiter = SlidingWindowRateLimiter(
    max_requests=_env_int("COMPASS_RATE_LIMIT_MODEL_INFO", 120),
    window_seconds=_rate_window_sec,
)


def _get_handshake_secret() -> str:
    return os.environ.get("COMPASS_HANDSHAKE_TOKEN", "").strip()


def _verify_handshake(request: Request) -> None:
    """
    Constant-time per-request authentication when `COMPASS_HANDSHAKE_TOKEN` is configured (`[R2-HIGH-03]`).
    Prevents any unauthorized local process or cross-origin caller from invoking engine endpoints,
    while remaining zero-config for standalone CLI / MozartAPI-v2 deployments where no token is set.
    """
    secret = _get_handshake_secret()
    if not secret:
        return
    supplied = (request.headers.get("X-Compass-Handshake") or "").strip()
    if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), secret.encode("utf-8")):
        raise HTTPException(
            status_code=403,
            detail="ローカルエンジンの認証ハンドシェイクが無効です。",
        )


def _client_key_from_request(request: Request) -> str:
    """
    Derives rate-limiter bucket key strictly from peer socket address (`proxy_headers=False`),
    normalizing all IPv4/IPv6 loopback peers into a single `'loopback'` key (`[CRIT-04]`).
    """
    raw_host = (request.client.host if request.client and request.client.host else "loopback").strip().lower()
    if raw_host in _LOOPBACK_HOSTS:
        return "loopback"
    return raw_host


def _classify_domain_validation_error(exc: Exception) -> str:
    """
    Maps internal ValueError / MIDI parse exceptions to safe, static Japanese client messages without
    ever interpolating raw exception text or host filesystem paths (`secure-coding-guard` §7).
    """
    raw_msg = exc.args[0] if exc.args and isinstance(exc.args[0], str) else ""
    lowered = raw_msg.lower()

    if "unsupported task" in lowered or "unsupported sft task" in lowered:
        return "指定された生成タスクはこのモデルでサポートされていません。"
    if "unsupported genre" in lowered:
        return "指定されたジャンルタグはサポートされていません。"
    if "program" in lowered:
        return "指定された楽器プログラムはサポートされていません（PIANO / SAX のみ対応）。"
    if "指定されたモデル" in raw_msg:
        return "指定されたモデルが見つかりません。モデル一覧を再スキャンしてください。"
    if "state_dict" in lowered:
        return "モデルの重みファイル (model.pth) の形式が不正です。"
    if ("トークン長" in raw_msg or "上限" in raw_msg) and "拍子" not in raw_msg:
        return "入力コンテキストMIDIのノート数または小節数がモデルの許容上限を超えています。"
    if "拍子" in raw_msg or "time_signature" in lowered:
        return "指定された拍子記号はモデルの1小節上限 (4/4拍子相当・96ステップ) を超えているか不正です。"
    if "midi" in lowered or "必須" in raw_msg or "コンテキスト" in raw_msg:
        return "選択されたタスクに必要な入力MIDIコンテキストが不足しているか、MIDI形式が不正です。"
    return "リクエストパラメータの検証に失敗しました。入力設定を確認してください。"


def _read_smf_vlq(buf: bytes, pos: int, end: int) -> Optional[Tuple[int, int]]:
    """Reads an SMF variable-length quantity (at most 4 bytes). Returns `(value, next_pos)` or `None`."""
    val = 0
    for count in range(4):
        if pos >= end:
            return None
        b = buf[pos]
        pos += 1
        val = (val << 7) | (b & 0x7F)
        if (b & 0x80) == 0:
            return val, pos
        if count == 3:
            return None
    return None


def _validate_mtrk_events(
    track_payload: bytes, max_track_ticks: int, *, is_smpte: bool = False
) -> bool:
    """
    Walks SMF 1.0 `MTrk` events in O(B) without allocating tick tables (`[R2-MED-06]`, `[R6-LOW-01]`):
    - Rejects malformed >4-byte VLQs or truncated event payloads.
    - Rejects zero-tempo meta events (`FF 51 03 00 00 00`) on metrical PPQN streams (linear SMPTE ignores `0xFF 0x51`).
    - Caps cumulative track ticks at `max_track_ticks` so crafted 38-byte VLQ delta-time bombs
      (`9,999,000` ticks -> ~80 MB in `PrettyMIDI._update_tick_to_time`) are rejected before `PrettyMIDI`.
    """
    pos = 0
    end = len(track_payload)
    cum_ticks = 0
    running_status = 0

    while pos < end:
        vlq_res = _read_smf_vlq(track_payload, pos, end)
        if vlq_res is None:
            return False
        delta, pos = vlq_res
        cum_ticks += delta
        if cum_ticks > max_track_ticks:
            return False
        if pos >= end:
            return False

        first = track_payload[pos]
        if first >= 0x80:
            status = first
            pos += 1
            if status == 0xFF:
                running_status = 0
                if pos >= end:
                    return False
                meta_type = track_payload[pos]
                pos += 1
                len_res = _read_smf_vlq(track_payload, pos, end)
                if len_res is None:
                    return False
                meta_len, pos = len_res
                if pos + meta_len > end:
                    return False
                if meta_type == 0x51:
                    if meta_len != 3 or (
                        not is_smpte and track_payload[pos : pos + 3] == b"\x00\x00\x00"
                    ):
                        return False
                pos += meta_len
                if meta_type == 0x2F:
                    break
            elif status in (0xF0, 0xF7):
                running_status = 0
                len_res = _read_smf_vlq(track_payload, pos, end)
                if len_res is None:
                    return False
                sysex_len, pos = len_res
                if pos + sysex_len > end:
                    return False
                pos += sysex_len
            elif 0x80 <= status <= 0xEF:
                running_status = status
                hi = status & 0xF0
                data_len = 1 if hi in (0xC0, 0xD0) else 2
                if pos + data_len > end:
                    return False
                pos += data_len
            else:
                if status < 0xF8:
                    running_status = 0
                if status in (0xF1, 0xF3):
                    sys_len = 1
                elif status == 0xF2:
                    sys_len = 2
                else:
                    sys_len = 0
                if pos + sys_len > end:
                    return False
                pos += sys_len
        else:
            if running_status == 0:
                return False
            hi = running_status & 0xF0
            data_len = 1 if hi in (0xC0, 0xD0) else 2
            if pos + data_len > end:
                return False
            pos += data_len

    return True


def _encode_smf_vlq(val: int) -> bytes:
    """Encodes a non-negative 28-bit integer as a Standard MIDI File VLQ."""
    v = max(0, min(0x0FFFFFFF, int(val)))
    out = [v & 0x7F]
    v >>= 7
    while v > 0 and len(out) < 4:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.reverse()
    return bytes(out)


def _extract_smf_ticks_per_quarter(division: int) -> Optional[float]:
    """
    Returns exact input ticks per 120-BPM quarter note (`0.5s` for linear SMPTE divisions,
    exact `division` for metrical PPQN) without clamping low-subframe SMPTE rates
    (`[R5-MED-02]`, `[R6-LOW-01]`).
    """
    if isinstance(division, bool) or not isinstance(division, int) or division <= 0 or division > 0xFFFF:
        return None
    if (division & 0x8000) != 0:
        ticks_per_frame = division & 0x00FF
        fps = 256 - ((division >> 8) & 0xFF)
        if fps <= 0 or fps > 120 or ticks_per_frame == 0:
            return None
        actual_fps = 29.97 if fps == 29 else float(fps)
        return (actual_fps * float(ticks_per_frame)) / 2.0
    return float(division)


def _extract_smf_effective_ppqn(division: int) -> Optional[int]:
    tpq = _extract_smf_ticks_per_quarter(division)
    if tpq is None:
        return None
    return max(1, int(round(tpq)))


def _validate_smf_binary_structure(data: bytes) -> bool:
    """
    Layer-3 binary Standard MIDI File (SMF 1.0) header + full multi-chunk & VLQ event validation
    (`[HIGH-10.2]`, `[R2-MED-06]`, `[R4-MED-06]`, `[R5-MED-02]`, `[R6-LOW-01]`).
    Bounds musical duration in quarter notes (`int(MAX_SMF_QUARTER_NOTES * ticks_per_quarter)`) so high-PPQN
    DAW exports (e.g. `9600 PPQN` Pro Tools) and SMPTE streams are accepted while VLQ delta-time bombs are rejected in <1ms.
    """
    if len(data) < 22:
        return False
    magic, header_len = struct.unpack(">4sI", data[:8])
    if magic != b"MThd" or header_len < 6:
        return False
    if 8 + header_len + 8 > len(data):
        return False

    smf_format, num_tracks, division = struct.unpack(">HHH", data[8:14])
    if smf_format not in (0, 1, 2):
        return False
    if num_tracks < 1 or num_tracks > 256 or (smf_format == 0 and num_tracks != 1):
        return False
    ticks_per_quarter = _extract_smf_ticks_per_quarter(division)
    if ticks_per_quarter is None or ticks_per_quarter <= 0:
        return False

    is_smpte = (division & 0x8000) != 0
    max_track_ticks = int(MAX_SMF_QUARTER_NOTES * ticks_per_quarter)

    offset = 8 + header_len
    mtrk_count = 0
    total_len = len(data)
    while offset < total_len:
        if offset + 8 > total_len:
            return False
        chunk_magic, chunk_len = struct.unpack(">4sI", data[offset : offset + 8])
        if not all(32 <= b <= 126 for b in chunk_magic):
            return False
        chunk_end = offset + 8 + chunk_len
        if chunk_end > total_len:
            return False
        if chunk_magic == b"MTrk":
            mtrk_count += 1
            track_payload = data[offset + 8 : chunk_end]
            if not _validate_mtrk_events(track_payload, max_track_ticks, is_smpte=is_smpte):
                return False
        offset = chunk_end

    return mtrk_count == num_tracks and mtrk_count >= 1


def _normalize_high_ppqn_smf_bytes(data: bytes, target_ppqn: int = 480) -> bytes:
    """
    Rescales validated high-PPQN (`> target_ppqn`, e.g. `9600 PPQN` Pro Tools) or SMPTE SMFs
    down to `target_ppqn` (`480`) in single-pass O(B) before `PrettyMIDI` (`[R4-MED-06]`, `[R5-MED-02]`, `[R6-LOW-01]`),
    stripping `0xFF 0x51` Set Tempo events on linear SMPTE files (with lossless delta carry) so
    `PrettyMIDI` interprets normalized `480 PPQN` ticks at exact `120 BPM` (`0.5s`/quarter) wall-clock parity.
    """
    _, header_len = struct.unpack(">4sI", data[:8])
    smf_format, num_tracks, division = struct.unpack(">HHH", data[8:14])
    is_smpte = (division & 0x8000) != 0
    if not is_smpte and division <= target_ppqn:
        return data

    ticks_per_quarter = _extract_smf_ticks_per_quarter(division)
    if ticks_per_quarter is None or ticks_per_quarter <= 0:
        return data

    max_target_ticks = MAX_SMF_QUARTER_NOTES * target_ppqn
    out_chunks: List[bytes] = [
        struct.pack(">4sIHHH", b"MThd", header_len, smf_format, num_tracks, target_ppqn)
        + data[14 : 8 + header_len]
    ]

    offset = 8 + header_len
    total_len = len(data)
    while offset < total_len:
        chunk_magic, chunk_len = struct.unpack(">4sI", data[offset : offset + 8])
        chunk_end = offset + 8 + chunk_len
        if chunk_magic != b"MTrk":
            out_chunks.append(data[offset:chunk_end])
            offset = chunk_end
            continue

        track_payload = data[offset + 8 : chunk_end]
        rebuilt = bytearray()
        pos = 0
        end = len(track_payload)
        cum_ticks = 0
        prev_scaled_ticks = 0
        running_status = 0

        while pos < end:
            vlq_res = _read_smf_vlq(track_payload, pos, end)
            if vlq_res is None:
                return data
            delta, pos = vlq_res
            cum_ticks += delta
            scaled_ticks = min(
                max_target_ticks, round(cum_ticks * target_ppqn / ticks_per_quarter)
            )

            ev_start = pos
            first = track_payload[pos]
            is_eot = False
            skip_event = False
            if first >= 0x80:
                status = first
                pos += 1
                if status == 0xFF:
                    running_status = 0
                    meta_type = track_payload[pos]
                    pos += 1
                    len_res = _read_smf_vlq(track_payload, pos, end)
                    if len_res is None:
                        return data
                    meta_len, pos = len_res
                    pos += meta_len
                    if meta_type == 0x2F:
                        is_eot = True
                    elif is_smpte and meta_type == 0x51:
                        skip_event = True
                elif status in (0xF0, 0xF7):
                    running_status = 0
                    len_res = _read_smf_vlq(track_payload, pos, end)
                    if len_res is None:
                        return data
                    sysex_len, pos = len_res
                    pos += sysex_len
                elif 0x80 <= status <= 0xEF:
                    running_status = status
                    hi = status & 0xF0
                    pos += 1 if hi in (0xC0, 0xD0) else 2
                else:
                    if status < 0xF8:
                        running_status = 0
                    if status in (0xF1, 0xF3):
                        pos += 1
                    elif status == 0xF2:
                        pos += 2
            else:
                hi = running_status & 0xF0
                pos += 1 if hi in (0xC0, 0xD0) else 2

            if skip_event:
                continue

            new_delta = max(0, scaled_ticks - prev_scaled_ticks)
            prev_scaled_ticks = scaled_ticks
            rebuilt.extend(_encode_smf_vlq(new_delta))
            rebuilt.extend(track_payload[ev_start:pos])
            if is_eot:
                break

        out_chunks.append(struct.pack(">4sI", b"MTrk", len(rebuilt)) + bytes(rebuilt))
        offset = chunk_end

    return b"".join(out_chunks)


async def _save_validated_midi_upload(
    upload: Optional[UploadFile], temp_dir: str, slot_name: str
) -> Optional[str]:
    """
    Triple-layer MIDI upload defense (`secure-coding-guard` §4):
    1. Extension allowlist (`.mid` / `.midi`)
    2. Byte size cap (`10 MB` -> `HTTP 413`)
    3. Full SMF `MThd` + all `MTrk` chunks + non-zero tempo structural validation (`HTTP 400`)
    4. High-PPQN (`>480` / SMPTE) normalization to `480 PPQN` + randomized UUID filename (`[R4-MED-06]`)
    """
    if upload is None:
        return None
    if not upload.filename or not upload.filename.strip():
        probe = await upload.read(1)
        if not probe:
            return None
        raise HTTPException(
            status_code=400,
            detail=f"{slot_name} には MIDI ファイル (.mid / .midi) を指定してください。",
        )

    _, ext = os.path.splitext(upload.filename)
    if ext.lower() not in ALLOWED_MIDI_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"{slot_name} には MIDI ファイル (.mid / .midi) を指定してください。",
        )

    data = await upload.read(MAX_MIDI_UPLOAD_BYTES + 1)
    if len(data) > MAX_MIDI_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{slot_name} のファイルサイズが上限 (10MB) を超えています。",
        )
    if not _validate_smf_binary_structure(data):
        raise HTTPException(
            status_code=400,
            detail=f"{slot_name} は有効な Standard MIDI ファイル (MThd / MTrk) ではありません。",
        )

    normalized_data = _normalize_high_ppqn_smf_bytes(data, target_ppqn=480)
    safe_path = os.path.join(temp_dir, f"{slot_name}_{uuid.uuid4().hex}.mid")
    with open(safe_path, "wb") as f:
        f.write(normalized_data)
    return safe_path


def _cleanup_dirs(*dirs: str) -> None:
    for d in dirs:
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


@app.get("/health")
async def health_check(request: Request):
    status = await _run_meta(model_controller.get_hardware_status, False)
    secret = _get_handshake_secret()
    payload: Dict[str, Any] = {
        "status": "ok",
        "engine": "compass-mortm-4.5",
        "device": status["device"],
        "dtype": status["dtype"],
        "models_dir_ready": bool(status.get("user_models_dir")),
        "available_models_count": status["available_models_count"],
        "auth_required": bool(secret),
    }
    if secret:
        challenge = (request.headers.get("X-Compass-Challenge") or "").strip()
        if 1 <= len(challenge) <= 256:
            payload["handshake_hmac"] = hmac.new(
                secret.encode("utf-8"),
                challenge.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
    return payload


@app.get("/hardware_info")
async def hardware_info(request: Request):
    _verify_handshake(request)
    return await _run_meta(model_controller.get_hardware_status, False)


def _normalize_job_id(raw: Optional[str]) -> Optional[str]:
    """
    Canonical `job_id` sanitizer shared across `/generate`, `/cancel_generate`, and `/generation_progress`
    so valid `>128`-char client job IDs normalize identically while rejecting malformed characters or
    comma-folded suffixes anywhere in the input (`[R3-LOW-03]`, `[R4-MED-04]`, `[R7-LOW-03]`).
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned or len(cleaned) > 4096:
        return None
    if not _JOB_ID_FULL_CHARS_PATTERN.fullmatch(cleaned):
        return None
    sliced = cleaned[:128]
    if not _JOB_ID_PATTERN.fullmatch(sliced):
        return None
    return sliced


def _extract_raw_header_values(
    request: Request,
    lower_name: bytes,
    invalid_detail: str = "指定された job_id の形式が不正です。",
) -> List[str]:
    raw_scope = getattr(request, "scope", None)
    raw_scope_headers = raw_scope.get("headers") if isinstance(raw_scope, dict) else None
    if isinstance(raw_scope_headers, list):
        out: List[str] = []
        for entry in raw_scope_headers:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                continue
            k, v = entry
            if isinstance(k, (bytes, bytearray)) and bytes(k).lower() == lower_name:
                if not isinstance(v, (bytes, bytearray)):
                    raise HTTPException(status_code=400, detail=invalid_detail)
                try:
                    out.append(bytes(v).decode("utf-8", errors="strict"))
                except UnicodeDecodeError:
                    raise HTTPException(status_code=400, detail=invalid_detail)
        return out
    ascii_name = lower_name.decode("ascii")
    req_headers = getattr(request, "headers", None)
    if hasattr(req_headers, "getlist"):
        return list(req_headers.getlist(ascii_name))
    if isinstance(req_headers, dict):
        return [str(v) for k, v in req_headers.items() if str(k).lower() == ascii_name]
    return []


def _extract_query_and_header_job_id(
    request: Request,
) -> Tuple[Optional[str], Optional[str], bool, bool]:
    """
    Fail-closed query (`?job_id=...`) + raw ASGI header (`X-Compass-Job-Id`) validator shared across
    `GET /generation_progress`, `POST /cancel_generate`, and `POST /generate` (`[R5-HIGH-01]`, `[R6-MED-01]`).
    Rejects duplicate same-source parameters, comma-folded headers, non-UTF8 bytes, and invalid characters with HTTP 400.
    """
    req_query = getattr(request, "query_params", None)
    if hasattr(req_query, "getlist"):
        q_list = list(req_query.getlist("job_id"))
        has_query_param = "job_id" in req_query or len(q_list) > 0
    elif isinstance(req_query, dict):
        has_query_param = "job_id" in req_query
        q_list = [req_query["job_id"]] if has_query_param else []
    else:
        q_list = []
        has_query_param = False
    if len(q_list) > 1:
        raise HTTPException(
            status_code=400, detail="複数の job_id がクエリに指定されています。"
        )
    q_norm: Optional[str] = None
    if len(q_list) == 1:
        raw_q = q_list[0]
        s_q = raw_q.strip() if isinstance(raw_q, str) else ""
        if s_q:
            q_norm = _normalize_job_id(s_q)
            if q_norm is None:
                raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")

    h_list = _extract_raw_header_values(request, b"x-compass-job-id")
    has_header_param = len(h_list) > 0
    if len(h_list) > 1:
        raise HTTPException(
            status_code=400, detail="複数の job_id がヘッダーに指定されています。"
        )
    h_norm: Optional[str] = None
    if len(h_list) == 1:
        raw_h = h_list[0]
        s_h = raw_h.strip() if isinstance(raw_h, str) else ""
        if s_h:
            h_norm = _normalize_job_id(s_h)
            if h_norm is None:
                raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")

    return q_norm, h_norm, has_query_param, has_header_param


@app.get("/generation_progress")
async def generation_progress(request: Request):
    _verify_handshake(request)
    q_norm, h_norm, has_query_param, has_header_param = _extract_query_and_header_job_id(request)
    if q_norm is not None and h_norm is not None and q_norm != h_norm:
        raise HTTPException(
            status_code=400,
            detail="クエリとヘッダーで異なる job_id が指定されています。",
        )
    target_job_id = q_norm or h_norm
    if target_job_id is None and (has_query_param or has_header_param):
        raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")
    snap = dict(model_controller.get_generation_progress(job_id=target_job_id))
    if target_job_id is None:
        snap["job_id"] = None
        snap["run_id"] = None
    return snap


@app.post("/cancel_generate")
async def cancel_generate(request: Request):
    _cancel_limiter.check_or_raise(
        _client_key_from_request(request),
        "キャンセルリクエストの送信頻度が高すぎます。",
    )
    _verify_handshake(request)

    q_norm, h_norm, has_query_param, has_header_param = _extract_query_and_header_job_id(request)

    cl_values = _extract_raw_header_values(
        request, b"content-length", invalid_detail="Content-Length ヘッダーの形式が不正です。"
    )
    if len(cl_values) > 1:
        raise HTTPException(
            status_code=400, detail="複数の Content-Length ヘッダーが指定されています。"
        )
    if len(cl_values) == 1:
        cl_str = cl_values[0].strip()
        if not cl_str or len(cl_str) > 64 or not (cl_str.isascii() and cl_str.isdigit()):
            raise HTTPException(
                status_code=400, detail="Content-Length ヘッダーの形式が不正です。"
            )
        if int(cl_str) > 4096:
            raise HTTPException(
                status_code=413,
                detail="キャンセルリクエストのペイロードサイズが上限 (4KB) を超えています。",
            )

    cancel_ct_values = _extract_raw_header_values(
        request, b"content-type", invalid_detail="Content-Type ヘッダーの形式が不正です。"
    )
    if len(cancel_ct_values) > 1:
        raise HTTPException(
            status_code=400, detail="複数の Content-Type ヘッダーが指定されています。"
        )
    if len(cancel_ct_values) == 1 and cancel_ct_values[0].strip():
        raw_cancel_ct = cancel_ct_values[0]
        if len(raw_cancel_ct) > 2048 or any(
            (ord(ch) < 0x20 and ch != "\t") or ord(ch) > 0x7E for ch in raw_cancel_ct
        ):
            raise HTTPException(status_code=400, detail="Content-Type ヘッダーの形式が不正です。")
        try:
            _cancel_mt, cancel_ct_params = _parse_strict_mime_header(
                raw_cancel_ct, require_slash=True, allow_utf8_quoted=False
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Content-Type ヘッダーの形式が不正です。"
            ) from exc
        if b"charset" in cancel_ct_params:
            if cancel_ct_params[b"charset"].strip().lower() not in _ALLOWED_FORM_CHARSETS:
                raise HTTPException(
                    status_code=400,
                    detail="サポートされていない文字エンコーディングです。",
                )

    body_chunks: List[bytes] = []
    total_body = 0
    try:
        async for chunk in request.stream():
            total_body += len(chunk)
            if total_body > 4096:
                raise HTTPException(
                    status_code=413,
                    detail="キャンセルリクエストのペイロードサイズが上限 (4KB) を超えています。",
                )
            body_chunks.append(chunk)
    except ClientDisconnect as exc:
        raise HTTPException(
            status_code=400,
            detail="キャンセルリクエスト送信中にクライアント接続が切断されました。",
        ) from exc
    body_bytes = b"".join(body_chunks)

    b_norm: Optional[str] = None
    explicit_cancel_all = False
    if body_bytes.strip():
        try:
            parsed = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            raise HTTPException(
                status_code=400, detail="キャンセルリクエストのJSON形式が不正です。"
            )
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=400,
                detail="キャンセルリクエストはJSONオブジェクトである必要があります。",
            )
        keys = set(parsed.keys())
        if keys == {"job_id"}:
            b_val = parsed["job_id"]
            b_norm = _normalize_job_id(b_val) if isinstance(b_val, str) else None
            if b_norm is None:
                raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")
        elif (
            keys == {"cancel_all"}
            and type(parsed["cancel_all"]) is bool
            and parsed["cancel_all"] is True
        ):
            explicit_cancel_all = True
        else:
            raise HTTPException(
                status_code=400,
                detail="キャンセルリクエストには有効な job_id または cancel_all: true を指定してください。",
            )

    if explicit_cancel_all and (has_query_param or has_header_param):
        raise HTTPException(
            status_code=400,
            detail="cancel_all 指定時に job_id クエリまたはヘッダーを併用することはできません。",
        )

    distinct_ids = {jid for jid in (q_norm, h_norm, b_norm) if jid is not None}
    if len(distinct_ids) > 1:
        raise HTTPException(
            status_code=400,
            detail="クエリ・ヘッダー・ボディ間で異なる job_id が指定されています。",
        )

    target_job_id = q_norm or h_norm or b_norm
    if target_job_id is None and not explicit_cancel_all:
        raise HTTPException(
            status_code=400,
            detail="キャンセル対象の job_id または明示的な cancel_all: true 指定が必要です。",
        )

    cancelled = model_controller.cancel_active_generation(job_id=target_job_id)
    return {
        "status": "cancelled" if cancelled else "idle",
        "cancelled": cancelled,
        "job_id": target_job_id,
    }


@app.post("/rescan_models")
async def rescan_models(request: Request):
    _rescan_limiter.check_or_raise(
        _client_key_from_request(request),
        "モデル再スキャンの実行頻度が高すぎます。しばらく待ってから再試行してください。",
    )
    _verify_handshake(request)
    models = await _run_meta(model_controller.get_model_info, True)
    return {
        "status": "rescanned",
        "user_models_dir": resolve_physical_models_dir_for_shell(model_controller.user_models_dir),
        "models": models,
    }


@app.post("/model_info")
async def get_model_info(request: Request):
    _model_info_limiter.check_or_raise(
        _client_key_from_request(request),
        "モデル情報取得リクエストの送信頻度が高すぎます。しばらく待ってから再試行してください。",
    )
    _verify_handshake(request)
    return await _run_meta(model_controller.get_model_info, False)


async def _parse_bounded_generate_form(request: Request) -> FormData:
    """
    Post-handshake, post-rate-limit bounded form parser for `POST /generate` (`[R21-HIGH-01]`).
    - Reads 0 bytes when Content-Length exceeds limits or Content-Type is missing/unsupported.
    - Enforces streaming byte caps (`34 MB` multipart / `1 MB` urlencoded) and closes all rolled
      `%TEMP%` `SpooledTemporaryFile` handles synchronously on `ClientDisconnect` or any exception.
    """
    ct_values = _extract_raw_header_values(
        request, b"content-type", invalid_detail="Content-Type ヘッダーの形式が不正です。"
    )
    if len(ct_values) > 1:
        raise HTTPException(status_code=400, detail="複数の Content-Type ヘッダーが指定されています。")
    ct_raw = ct_values[0] if len(ct_values) == 1 else (request.headers.get("content-type") or "")

    cl_values = _extract_raw_header_values(
        request, b"content-length", invalid_detail="Content-Length ヘッダーの形式が不正です。"
    )
    if len(cl_values) > 1:
        raise HTTPException(status_code=400, detail="複数の Content-Length ヘッダーが指定されています。")
    declared_len: Optional[int] = None
    if len(cl_values) == 1:
        cl_str = cl_values[0].strip()
        if not cl_str or len(cl_str) > 64 or not (cl_str.isascii() and cl_str.isdigit()):
            raise HTTPException(status_code=400, detail="Content-Length ヘッダーの形式が不正です。")
        declared_len = int(cl_str)

    if ct_raw and (
        len(ct_raw) > 2048
        or any((ord(ch) < 0x20 and ch != "\t") or ord(ch) > 0x7E for ch in ct_raw)
    ):
        raise HTTPException(status_code=400, detail="Content-Type ヘッダーの形式が不正です。")

    if not ct_raw or not ct_raw.strip():
        if declared_len is not None and declared_len > 0:
            raise HTTPException(
                status_code=400,
                detail="Content-Type は multipart/form-data または application/x-www-form-urlencoded を指定してください。",
            )
        return FormData()

    try:
        media_type, ct_params = _parse_strict_mime_header(
            ct_raw, require_slash=True, allow_utf8_quoted=False
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Content-Type ヘッダーの形式が不正です。"
        ) from exc

    norm_charset_str = "utf-8"
    if b"charset" in ct_params:
        raw_charset = ct_params.get(b"charset")
        norm_charset = (
            bytes(raw_charset).strip().lower()
            if isinstance(raw_charset, (bytes, bytearray))
            else b""
        )
        if norm_charset not in _ALLOWED_FORM_CHARSETS:
            raise HTTPException(
                status_code=400,
                detail="サポートされていない文字エンコーディングです。",
            )
        norm_charset_str = norm_charset.decode("ascii")

    if media_type == b"multipart/form-data":
        boundary = ct_params.get(b"boundary")
        if (
            not isinstance(boundary, (bytes, bytearray))
            or len(boundary) == 0
            or len(boundary) > 256
            or boundary.endswith(b" ")
            or any(b < 0x20 or b > 0x7E for b in boundary)
        ):
            raise HTTPException(
                status_code=400, detail="multipart/form-data の boundary が不正です。"
            )
        if declared_len is not None and declared_len > MAX_GENERATE_BODY_BYTES:
            raise HTTPException(
                status_code=413,
                detail="リクエスト全体のサイズが上限 (34MB) を超えています。",
            )

        boundary_bytes = bytes(boundary)
        end_marker = b"\r\n--" + boundary_bytes + b"--"
        zero_marker = b"--" + boundary_bytes + b"--"
        keep_tail = len(end_marker) - 1
        epilogue_buf = bytearray()
        mp_parser_holder: List[_BoundedMultiPartParser] = []

        async def _bounded_multipart_stream() -> AsyncGenerator[bytes, None]:
            total_bytes = 0
            prev_tail = b""
            async for chunk in request.stream():
                if chunk:
                    total_bytes += len(chunk)
                    if total_bytes > MAX_GENERATE_BODY_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="リクエスト全体のサイズが上限 (34MB) を超えています。",
                        )
                parser_ref = mp_parser_holder[0] if mp_parser_holder else None
                was_ended = bool(parser_ref and parser_ref._saw_multipart_end)
                yield chunk
                if not chunk or parser_ref is None:
                    continue
                if was_ended:
                    epilogue_buf.extend(chunk)
                elif parser_ref._saw_multipart_end:
                    if parser_ref._saw_any_part:
                        window = prev_tail + chunk
                        pos = window.find(end_marker)
                        if pos >= 0:
                            epilogue_buf.extend(window[pos + len(end_marker) :])
                    else:
                        window = prev_tail + chunk
                        pos = window.find(zero_marker)
                        if pos >= 0:
                            epilogue_buf.extend(window[pos + len(zero_marker) :])
                else:
                    if not parser_ref._saw_any_part and total_bytes > len(zero_marker) + 64:
                        raise MultiPartException("Excessive multipart preamble.")
                    combined = prev_tail + chunk
                    prev_tail = combined[-keep_tail:] if len(combined) > keep_tail else combined
                if len(epilogue_buf) > 64 or any(b not in b" \t\r\n" for b in epilogue_buf):
                    raise MultiPartException(
                        "Unexpected trailing data after closing multipart boundary."
                    )

        escaped_boundary = (
            boundary_bytes.replace(b"\\", b"\\\\").replace(b'"', b'\\"').decode("latin-1")
        )
        sanitized_mp_headers = Headers(
            {
                "content-type": (
                    f"multipart/form-data; charset={norm_charset_str}; "
                    f'boundary="{escaped_boundary}"'
                )
            }
        )
        mp_parser = _BoundedMultiPartParser(
            sanitized_mp_headers,
            _bounded_multipart_stream(),
            max_files=MAX_GENERATE_MULTIPART_FILES,
            max_fields=MAX_GENERATE_MULTIPART_FIELDS,
            max_part_size=MAX_GENERATE_PART_BYTES,
        )
        mp_parser_holder.append(mp_parser)
        try:
            parsed_form = await mp_parser.parse()
            if not mp_parser._saw_multipart_end or mp_parser._in_open_part:
                raise MultiPartException("Incomplete multipart stream missing closing boundary.")
            return parsed_form
        except BaseException as exc:
            mp_parser.close_all_files_sync()
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, MultiPartException):
                if "maximum size" in exc.message.lower() and "header" not in exc.message.lower():
                    raise HTTPException(
                        status_code=413,
                        detail="フォームパートのサイズが上限 (1MB) を超えています。",
                    ) from exc
                raise HTTPException(
                    status_code=400,
                    detail="マルチパートフォームデータの形式が不正です。",
                ) from exc
            if isinstance(exc, ClientDisconnect):
                raise HTTPException(
                    status_code=400,
                    detail="アップロード中にクライアント接続が切断されました。",
                ) from exc
            if isinstance(
                exc,
                (MultipartParseError, QuerystringParseError, FormParserError, ParseError, ValueError),
            ):
                raise HTTPException(
                    status_code=400,
                    detail="マルチパートフォームデータの解析に失敗しました。",
                ) from exc
            raise

    if media_type == b"application/x-www-form-urlencoded":
        if declared_len is not None and declared_len > MAX_GENERATE_URLENCODED_BYTES:
            raise HTTPException(
                status_code=413,
                detail="フォームデータのサイズが上限 (1MB) を超えています。",
            )

        async def _bounded_urlencoded_stream() -> AsyncGenerator[bytes, None]:
            total_bytes = 0
            sep_count = 0
            async for chunk in request.stream():
                if chunk:
                    total_bytes += len(chunk)
                    if total_bytes > MAX_GENERATE_URLENCODED_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="フォームデータのサイズが上限 (1MB) を超えています。",
                        )
                    sep_count += chunk.count(b"&") + chunk.count(b";")
                    if sep_count > 64:
                        raise HTTPException(
                            status_code=400,
                            detail="フォームフィールド数が上限を超えています。",
                        )
                yield chunk

        form_parser = FormParser(request.headers, _bounded_urlencoded_stream())
        try:
            return await form_parser.parse()
        except HTTPException:
            raise
        except ClientDisconnect as exc:
            raise HTTPException(
                status_code=400,
                detail="送信中にクライアント接続が切断されました。",
            ) from exc
        except (QuerystringParseError, FormParserError, ParseError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="フォームデータの解析に失敗しました。",
            ) from exc

    raise HTTPException(
        status_code=400,
        detail="Content-Type は multipart/form-data または application/x-www-form-urlencoded を指定してください。",
    )


@app.post("/generate")
async def _generate_music_http(request: Request):
    return await generate_music(request=request)


async def generate_music(
    request: Request,
    meta: Optional[str] = None,
    meta_json: Optional[str] = None,
    job_id: Optional[str] = None,
    past_midi: Optional[UploadFile] = None,
    conditions_midi: Optional[UploadFile] = None,
    future_midi: Optional[UploadFile] = None,
):
    direct_meta = None if isinstance(meta, FieldInfo) else meta
    direct_meta_json = None if isinstance(meta_json, FieldInfo) else meta_json
    direct_job_id = None if isinstance(job_id, FieldInfo) else job_id
    direct_past = None if isinstance(past_midi, FieldInfo) else past_midi
    direct_cond = None if isinstance(conditions_midi, FieldInfo) else conditions_midi
    direct_future = None if isinstance(future_midi, FieldInfo) else future_midi
    has_direct_args = any(
        arg is not None
        for arg in (
            direct_meta,
            direct_meta_json,
            direct_job_id,
            direct_past,
            direct_cond,
            direct_future,
        )
    )
    form_data_ref: Any = getattr(request, "_form", None)

    try:
        _generate_limiter.check_or_raise(
            _client_key_from_request(request),
            "生成リクエストの送信頻度が高すぎます。しばらく待ってから再試行してください。",
        )
        _verify_handshake(request)

        stream_fn = getattr(request, "stream", None)
        form_fn = getattr(request, "form", None)
        if not has_direct_args and (callable(stream_fn) or callable(form_fn)):
            if callable(stream_fn):
                form_data_ref = await _parse_bounded_generate_form(request)
            else:
                form_data_ref = await form_fn()
            for raw_key, _raw_val in form_data_ref.multi_items():
                if not isinstance(raw_key, str) or raw_key not in _ALLOWED_GENERATE_FORM_KEYS:
                    raise HTTPException(
                        status_code=400,
                        detail="未対応のフォームフィールドが指定されています。",
                    )
            for allowed_key in _ALLOWED_GENERATE_FORM_KEYS:
                key_items = list(form_data_ref.getlist(allowed_key))
                if len(key_items) > 1:
                    if allowed_key == "job_id":
                        raise HTTPException(
                            status_code=400, detail="複数の job_id がフォームに指定されています。"
                        )
                    raise HTTPException(
                        status_code=400,
                        detail=f"複数の {allowed_key} フィールドが指定されています。",
                    )
                if len(key_items) == 1:
                    single_item = key_items[0]
                    if allowed_key in _GENERATE_TEXT_FORM_KEYS:
                        if not isinstance(single_item, str):
                            if allowed_key == "job_id":
                                raise HTTPException(
                                    status_code=400, detail="指定された job_id の形式が不正です。"
                                )
                            raise HTTPException(
                                status_code=400,
                                detail=f"{allowed_key} フィールドの形式が不正です。",
                            )
                    elif allowed_key in _GENERATE_FILE_FORM_KEYS:
                        if not isinstance(single_item, (StarletteUploadFile, UploadFile)):
                            raise HTTPException(
                                status_code=400,
                                detail=f"{allowed_key} には MIDI ファイル (.mid / .midi) を指定してください。",
                            )
            if "meta" in form_data_ref and "meta_json" in form_data_ref:
                raise HTTPException(
                    status_code=400,
                    detail="meta と meta_json を同時に指定することはできません。",
                )
            eff_meta = form_data_ref.get("meta")
            eff_meta_json = form_data_ref.get("meta_json")
            f_list = list(form_data_ref.getlist("job_id"))
            eff_past = form_data_ref.get("past_midi")
            eff_cond = form_data_ref.get("conditions_midi")
            eff_future = form_data_ref.get("future_midi")
        else:
            if direct_meta is not None and direct_meta_json is not None:
                raise HTTPException(
                    status_code=400,
                    detail="meta と meta_json を同時に指定することはできません。",
                )
            eff_meta = direct_meta
            eff_meta_json = direct_meta_json
            f_list = []
            eff_past = direct_past
            eff_cond = direct_cond
            eff_future = direct_future

        raw_meta = eff_meta if eff_meta is not None else eff_meta_json
        if not raw_meta or not isinstance(raw_meta, str):
            raise HTTPException(status_code=422, detail="meta または meta_json フィールドが必要です。")
        if len(raw_meta) > 65536 or len(raw_meta.encode("utf-8", errors="replace")) > 65536:
            raise HTTPException(
                status_code=413,
                detail="metaデータのサイズが上限 (64KB) を超えています。",
            )
        try:
            meta_dict = json.loads(raw_meta)
            if not isinstance(meta_dict, dict):
                raise HTTPException(status_code=400, detail="metaデータはJSONオブジェクトである必要があります。")
            meta_obj = GenerateMeta(**meta_dict)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            raise HTTPException(status_code=400, detail="metaデータが正しいJSON形式ではありません。")
        except ValidationError as e:
            logger.warning("GenerateMeta schema validation rejected: %s", e)
            if "拍子" in str(e):
                raise HTTPException(
                    status_code=400,
                    detail="指定された拍子記号はモデルの1小節上限 (4/4拍子相当・96ステップ) を超えているか不正です。",
                )
            raise HTTPException(
                status_code=422,
                detail="metaパラメータの入力形式または値の範囲が不正です。",
            )
        except ValueError as e:
            logger.warning("GenerateMeta validation rejected: %s", e)
            raise HTTPException(status_code=400, detail=_classify_domain_validation_error(e))

        q_norm, h_norm, has_query_param, has_header_param = _extract_query_and_header_job_id(request)
        has_form_param = len(f_list) > 0 or direct_job_id is not None
        if len(f_list) > 1:
            raise HTTPException(
                status_code=400, detail="複数の job_id がフォームに指定されています。"
            )
        f_norm: Optional[str] = None
        if len(f_list) == 1:
            raw_f = f_list[0]
            if not isinstance(raw_f, str):
                raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")
            s_f = raw_f.strip()
            if s_f:
                f_norm = _normalize_job_id(s_f)
                if f_norm is None:
                    raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")
        elif direct_job_id is not None:
            if not isinstance(direct_job_id, str):
                raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")
            s_jf = direct_job_id.strip()
            if s_jf:
                f_norm = _normalize_job_id(s_jf)
                if f_norm is None:
                    raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")

        gen_distinct_ids = {jid for jid in (q_norm, h_norm, f_norm) if jid is not None}
        if len(gen_distinct_ids) > 1:
            raise HTTPException(
                status_code=400,
                detail="クエリ・ヘッダー・フォーム間で異なる job_id が指定されています。",
            )

        provided_job_id = q_norm or h_norm or f_norm
        if provided_job_id is None and (has_query_param or has_header_param or has_form_param):
            raise HTTPException(status_code=400, detail="指定された job_id の形式が不正です。")

        eff_job_id = provided_job_id or uuid.uuid4().hex

        if not _GEN_QUEUE_SEMAPHORE.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="現在実行中・待機中の生成ジョブが上限に達しています。完了後に再試行してください。",
                headers={"Retry-After": "5"},
            )

        temp_input_dir = ""
        cancel_event: Optional[threading.Event] = None
        lifecycle_lock = threading.Lock()
        lifecycle_owner = "outer"
        try:
            temp_input_dir = tempfile.mkdtemp(prefix="compass_in_")
            cancel_event = model_controller.register_cancel_event(job_id=eff_job_id)
            past_path = await _save_validated_midi_upload(eff_past, temp_input_dir, "past_midi")
            const_path = await _save_validated_midi_upload(
                eff_cond, temp_input_dir, "conditions_midi"
            )
            future_path = await _save_validated_midi_upload(
                eff_future, temp_input_dir, "future_midi"
            )
            await _close_spooled_uploads(form_data_ref, (direct_past, direct_cond, direct_future))

            if past_path or const_path or future_path:
                try:
                    await _run_meta(
                        AbstractModelRapper.resolve_unified_time_signature,
                        meta_obj,
                        (past_path, const_path, future_path),
                    )
                except ValueError as e:
                    logger.warning("Pre-queue MIDI meter validation rejected: %s", e)
                    raise HTTPException(status_code=400, detail=_classify_domain_validation_error(e))

            if cancel_event.is_set():
                raise HTTPException(status_code=499, detail="生成処理を中断しました。")

            def _worker_generate() -> Tuple[bytes, str, str, Optional[Any]]:
                nonlocal lifecycle_owner
                with lifecycle_lock:
                    if lifecycle_owner != "outer":
                        raise GenerationCancelledError(
                            "Generation cancelled before worker execution started."
                        )
                    lifecycle_owner = "worker"
                out_dir = ""
                try:
                    archive_path, out_media_type, out_reason = model_controller.generate(
                        past_midi_path=past_path,
                        const_midi_path=const_path,
                        future_midi_path=future_path,
                        meta=meta_obj,
                        cancel_event=cancel_event,
                        job_id=eff_job_id,
                    )
                    out_dir = os.path.dirname(archive_path)
                    out_filename = os.path.basename(archive_path)
                    with open(archive_path, "rb") as f:
                        payload = f.read()
                    if cancel_event.is_set():
                        raise GenerationCancelledError(
                            "Generation cancelled while reading output archive."
                        )
                    return payload, out_filename, out_media_type, out_reason
                finally:
                    cancel_event.set()
                    model_controller.unregister_cancel_event(cancel_event, job_id=eff_job_id)
                    _cleanup_dirs(temp_input_dir, out_dir)
                    _GEN_QUEUE_SEMAPHORE.release()

            async def _watch_client_disconnect() -> None:
                try:
                    while not cancel_event.is_set():
                        if await request.is_disconnected():
                            cancel_event.set()
                            break
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    pass

            disconnect_watcher = asyncio.create_task(_watch_client_disconnect())
            try:
                loop = asyncio.get_running_loop()
                infer_future = loop.run_in_executor(_INFER_EXECUTOR, _worker_generate)
                payload_bytes, filename, media_type, reason = await infer_future
            except GenerationCancelledError:
                raise HTTPException(status_code=499, detail="生成処理を中断しました。")
            except (EOFError, ZeroDivisionError, struct.error) as e:
                logger.warning("Corrupt MIDI stream rejected during conversion: %s", e)
                raise HTTPException(
                    status_code=400,
                    detail="入力されたMIDIデータの構造またはテンポ情報が不正です。",
                )
            except ValueError as e:
                logger.warning("Generation domain validation error: %s", e)
                raise HTTPException(status_code=400, detail=_classify_domain_validation_error(e))
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404,
                    detail="指定されたモデルの重みファイル (model.pth) が見つかりません。AppData/Roaming/COMPASS_AI/models 内に配置してください。",
                )
            except BaseException as exc:
                cancel_event.set()
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                logger.exception("Unhandled error during model generation")
                raise HTTPException(status_code=500, detail="音楽生成処理中に内部エラーが発生しました。")
            finally:
                cancel_event.set()
                disconnect_watcher.cancel()
                try:
                    await disconnect_watcher
                except BaseException:
                    pass

            response_headers: Dict[str, str] = {
                "Content-Disposition": f'attachment; filename="{filename}"',
            }
            if reason is not None:
                try:
                    encoded = urllib.parse.quote(
                        json.dumps(reason, ensure_ascii=False),
                        safe="",
                    )
                    response_headers["X-Generation-Reason"] = encoded
                except Exception:
                    pass

            return Response(
                content=payload_bytes,
                media_type=media_type,
                headers=response_headers,
            )
        finally:
            if cancel_event is not None:
                cancel_event.set()
            should_outer_clean = False
            with lifecycle_lock:
                if lifecycle_owner == "outer":
                    lifecycle_owner = "outer_cleaned"
                    should_outer_clean = True
            if should_outer_clean:
                if cancel_event is not None:
                    model_controller.unregister_cancel_event(cancel_event, job_id=eff_job_id)
                _cleanup_dirs(temp_input_dir)
                _GEN_QUEUE_SEMAPHORE.release()
    finally:
        await _close_spooled_uploads(
            form_data_ref if form_data_ref is not None else getattr(request, "_form", None),
            (direct_past, direct_cond, direct_future),
        )


_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    _ASSETS_DIR = os.path.join(_FRONTEND_DIST, "assets")
    if os.path.isdir(_ASSETS_DIR):
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="frontend_assets")

    @app.get("/")
    async def serve_frontend_index():
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn

    port = _env_int("COMPASS_PORT", 8000)
    host = os.environ.get("COMPASS_HOST", "").strip() or "127.0.0.1"
    uvicorn.run(app, host=host, port=port, proxy_headers=False)
