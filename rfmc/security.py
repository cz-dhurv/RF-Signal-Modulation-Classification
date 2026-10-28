"""Security primitives: API-key auth, rate limiting, log sanitising, model supply-chain checks."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

from .errors import RateLimitedError, SecurityError

MAX_KEY_LEN = 256
ALLOWED_MODEL_SUFFIXES = frozenset({".onnx", ".safetensors", ".pt"})

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}
HSTS = "max-age=63072000; includeSubDomains"

_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def verify_api_key(provided: object, allowed: Iterable[str]) -> bool:
    """Constant-time check. Compares against every allowed key (no early exit)."""
    if not isinstance(provided, str) or not provided or len(provided) > MAX_KEY_LEN:
        return False
    digest = hashlib.sha256(provided.encode("utf-8", "replace")).digest()
    ok = False
    for key in allowed:
        ok |= hmac.compare_digest(digest, hashlib.sha256(key.encode("utf-8")).digest())
    return ok


def key_fingerprint(key: str) -> str:
    """Non-reversible short id used for rate-limit buckets and logs. Never log the key."""
    return hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()[:12]


def sanitize_for_log(value: object, limit: int = 200) -> str:
    return _CTRL.sub("?", str(value))[:limit]


def safe_request_id(candidate: str | None) -> str:
    """Accept a client-supplied request id only if it is a plain token; else generate one."""
    if candidate is not None and _REQUEST_ID.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


class TokenBucketRateLimiter:
    """Per-key token bucket with bounded memory (LRU eviction)."""

    def __init__(
        self,
        rate_per_min: float,
        burst: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ):
        if rate_per_min <= 0 or burst < 1 or max_keys < 1:
            raise ValueError("invalid rate limiter parameters")
        self._rate = rate_per_min / 60.0
        self._burst = float(burst)
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: "OrderedDict[str, tuple[float, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._buckets)

    def check(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.pop(key, (self._burst, now))
            tokens = min(self._burst, tokens + (now - last) * self._rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                allowed, retry = True, 0
            else:
                self._buckets[key] = (tokens, now)
                allowed, retry = False, math.ceil((1.0 - tokens) / self._rate)
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
        if not allowed:
            raise RateLimitedError(retry_after=retry)


def ensure_safe_model_path(
    path: str | Path,
    allowed_dir: str | Path,
    *,
    allowed_suffixes: Iterable[str] = ALLOWED_MODEL_SUFFIXES,
) -> Path:
    """Resolve symlinks/.. and require the file to live inside allowed_dir with an allowed suffix."""
    base = Path(allowed_dir).resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise SecurityError("Model path escapes the allowed directory.") from None
    if resolved.suffix.lower() not in set(allowed_suffixes):
        raise SecurityError("Model file type is not allowed.")
    if not resolved.is_file():
        raise SecurityError("Model file does not exist.")
    return resolved


def verify_sha256(path: str | Path, expected_hex: str) -> None:
    if not isinstance(expected_hex, str) or not _SHA256.fullmatch(expected_hex):
        raise SecurityError("Expected model hash is malformed.")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if not hmac.compare_digest(h.hexdigest(), expected_hex.lower()):
        raise SecurityError("Model hash mismatch.")
