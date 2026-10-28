"""Input validation: every malformed input must raise ValidationError and nothing else."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfmc.errors import ValidationError
from rfmc.validation import (
    normalize_power,
    validate_classify_payload,
    validate_iq_array,
    validate_sample_rate,
)

N = 128


def rows(n=N, value=1.0):
    return [[value] * n, [value] * n]


def test_valid_input_is_float32_unit_power():
    rng = np.random.default_rng(1)
    out = validate_iq_array(rng.standard_normal((2, 512)) * 37.0)
    assert out.shape == (2, 512) and out.dtype == np.float32
    assert math.isclose(float(np.mean(out[0] ** 2 + out[1] ** 2)), 1.0, rel_tol=1e-4)


def test_accepts_python_lists_and_integers():
    assert validate_iq_array([[1, -1] * 64, [-1, 1] * 64]).shape == (2, 128)


def test_boundary_lengths():
    assert validate_iq_array(rows(128, 1.0), min_len=128, max_len=4096).shape[1] == 128
    assert validate_iq_array(rows(4096, 1.0), min_len=128, max_len=4096).shape[1] == 4096
    with pytest.raises(ValidationError):
        validate_iq_array(rows(127))
    with pytest.raises(ValidationError):
        validate_iq_array(rows(4097))


def test_expected_len_is_exact():
    assert validate_iq_array(rows(1024), expected_len=1024).shape[1] == 1024
    with pytest.raises(ValidationError):
        validate_iq_array(rows(128), expected_len=1024)


def test_tiny_and_large_magnitudes_normalize_safely():
    tiny = validate_iq_array(rows(N, 1e-200))
    assert math.isclose(float(np.mean(tiny[0] ** 2 + tiny[1] ** 2)), 1.0, rel_tol=1e-4)
    with pytest.raises(ValidationError):
        validate_iq_array(rows(N, 1e300))  # above max_abs


BAD_INPUTS = {
    "none": None,
    "string": "1,2,3",
    "bytes": b"\x00\x01",
    "dict": {"i": [1], "q": [2]},
    "scalar": 5,
    "empty list": [],
    "empty rows": [[], []],
    "one row": [[1.0] * N],
    "three rows": [[1.0] * N] * 3,
    "transposed (N,2)": [[1.0, 2.0]] * N,
    "ragged": [[1.0] * N, [1.0] * (N - 1)],
    "too deep": [[[1.0] * N, [1.0] * N]],
    "nan": [[float("nan")] * N, [1.0] * N],
    "inf": [[float("inf")] * N, [1.0] * N],
    "neg inf": [[1.0] * N, [float("-inf")] * N],
    "numeric strings": [["1"] * N, ["2"] * N],
    "mixed str": [[1.0] * (N - 1) + ["x"], [1.0] * N],
    "bools": [[True] * N, [False] * N],
    "none inside": [[None] * N, [1.0] * N],
    "complex": [[1 + 2j] * N, [1 + 2j] * N],
    "bigint": [[10**40] * N, [1] * N],
    "all zero": [[0.0] * N, [0.0] * N],
    "over max_abs": [[1e7] * N, [1.0] * N],
}


@pytest.mark.parametrize("bad", BAD_INPUTS.values(), ids=BAD_INPUTS.keys())
def test_rejects_malformed_iq(bad):
    with pytest.raises(ValidationError):
        validate_iq_array(bad)


def test_fuzz_only_validation_error_escapes():
    rng = np.random.default_rng(7)
    junk = [object(), set(), (1, 2), 1e308, -0.0, "", [[]], [[[]]], [None], [[1], [2, 3]],
            [[float("nan")]], np.nan, {1: 2}, [[1.0] * 200, "ab"], range(5), lambda: 0]
    for _ in range(300):
        pick = junk[int(rng.integers(len(junk)))]
        if rng.random() < 0.5:
            pick = [pick, pick]
        try:
            validate_iq_array(pick)
        except ValidationError:
            pass  # any other exception type fails the test


@pytest.mark.parametrize("ok", [None, 1, 2.4e6, 61.44e6, 1e11])
def test_sample_rate_ok(ok):
    assert validate_sample_rate(ok) == (None if ok is None else float(ok))


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), 1e12, True, "2e6", [1], {}])
def test_sample_rate_rejected(bad):
    with pytest.raises(ValidationError):
        validate_sample_rate(bad)


def test_normalize_power_rejects_zero():
    with pytest.raises(ValidationError):
        normalize_power(np.zeros((2, 64)))


def test_normalize_does_not_mutate_input():
    x = np.random.default_rng(0).standard_normal((2, 256))
    before = x.copy()
    validate_iq_array(x)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("payload", [[], "x", 3, None, [[1.0] * N] * 2])
def test_payload_must_be_object(payload):
    with pytest.raises(ValidationError):
        validate_classify_payload(payload)


def test_payload_requires_iq():
    with pytest.raises(ValidationError, match="iq"):
        validate_classify_payload({"sample_rate_hz": 1e6})


def test_payload_rejects_unknown_fields_and_sanitizes_names():
    with pytest.raises(ValidationError) as ei:
        validate_classify_payload({"iq": rows(), "evil\r\nkey" + "x" * 5000: 1})
    msg = str(ei.value)
    assert "\n" not in msg and "\r" not in msg and len(msg) < 300


def test_payload_happy_path():
    req = validate_classify_payload({"iq": rows(256), "sample_rate_hz": 2e6})
    assert req.iq.shape == (2, 256) and req.sample_rate_hz == 2e6
