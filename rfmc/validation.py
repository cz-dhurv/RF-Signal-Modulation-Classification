"""Strict validation of untrusted I/Q input. Only ValidationError may escape."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ValidationError

MIN_LEN = 128
MAX_LEN = 4096
MAX_ABS = 1e6
MAX_SAMPLE_RATE_HZ = 1e11
ALLOWED_FIELDS = frozenset({"iq", "sample_rate_hz"})

_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _clip(value: Any, limit: int = 40) -> str:
    return _CTRL.sub("?", str(value))[:limit]


def normalize_power(x: np.ndarray) -> np.ndarray:
    """Scale a (2, N) float array to unit mean power. Rejects all-zero signals."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x)))
    if not math.isfinite(peak) or peak == 0.0:
        raise ValidationError("Signal has zero power.")
    x = x / peak  # pre-scale so tiny/huge magnitudes cannot under/overflow the power sum
    power = float(np.mean(x[0] ** 2 + x[1] ** 2))
    return x / math.sqrt(power)


def validate_iq_array(
    iq: Any,
    *,
    min_len: int = MIN_LEN,
    max_len: int = MAX_LEN,
    expected_len: int | None = None,
    max_abs: float = MAX_ABS,
) -> np.ndarray:
    """Return a power-normalized float32 array of shape (2, N) or raise ValidationError."""
    if iq is None or isinstance(iq, (str, bytes, bytearray, dict)):
        raise ValidationError("'iq' must be a 2 x N array of numbers.")
    try:
        arr = np.asarray(iq)
    except (ValueError, TypeError, MemoryError, RecursionError):
        raise ValidationError("'iq' must be a rectangular 2 x N array of numbers.") from None
    if arr.dtype.kind not in "iuf":  # rejects bool, str, complex, object (None/ragged/bigint)
        raise ValidationError("'iq' must contain real numbers only.")
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValidationError(
            "'iq' must have shape (2, N): row 0 = I, row 1 = Q.",
            details={"got_shape": [int(d) for d in arr.shape]},
        )
    n = int(arr.shape[1])
    if expected_len is not None:
        if n != expected_len:
            raise ValidationError(f"'iq' must have exactly {expected_len} samples per row.")
    elif not min_len <= n <= max_len:
        raise ValidationError(f"'iq' must have between {min_len} and {max_len} samples per row.")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValidationError("'iq' contains NaN or infinite values.")
    if float(np.max(np.abs(arr))) > max_abs:
        raise ValidationError(f"'iq' magnitudes must not exceed {max_abs:g}.")
    return normalize_power(arr).astype(np.float32)


def validate_sample_rate(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("'sample_rate_hz' must be a number.")
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0 or rate > MAX_SAMPLE_RATE_HZ:
        raise ValidationError("'sample_rate_hz' must be a positive finite number.")
    return rate


@dataclass(frozen=True)
class ClassifyRequest:
    iq: np.ndarray
    sample_rate_hz: float | None


def validate_classify_payload(
    payload: Any, *, min_len: int = MIN_LEN, max_len: int = MAX_LEN
) -> ClassifyRequest:
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")
    unknown = [k for k in payload if k not in ALLOWED_FIELDS]
    if unknown:
        shown = ", ".join(_clip(k) for k in unknown[:5])
        raise ValidationError(f"Unknown field(s): {shown}")
    if "iq" not in payload:
        raise ValidationError("Missing required field 'iq'.")
    iq = validate_iq_array(payload["iq"], min_len=min_len, max_len=max_len)
    return ClassifyRequest(iq=iq, sample_rate_hz=validate_sample_rate(payload.get("sample_rate_hz")))
