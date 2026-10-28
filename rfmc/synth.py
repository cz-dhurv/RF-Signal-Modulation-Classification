"""Tiny deterministic baseband signal generator for tests and augmentation checks."""

from __future__ import annotations

import numpy as np

SUPPORTED = ("BPSK", "QPSK", "8PSK", "QAM16")


def clean_symbols(mod: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Complex baseband symbols, 1 sample/symbol, unit average power."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if mod == "BPSK":
        return rng.choice([-1.0, 1.0], n).astype(np.complex128)
    if mod == "QPSK":
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n)))
    if mod == "8PSK":
        return np.exp(2j * np.pi * rng.integers(0, 8, n) / 8)
    if mod == "QAM16":
        lv = np.array([-3.0, -1.0, 1.0, 3.0])
        return (rng.choice(lv, n) + 1j * rng.choice(lv, n)) / np.sqrt(10.0)
    raise ValueError(f"unsupported modulation: {mod!r}; choose from {SUPPORTED}")


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    p_sig = float(np.mean(np.abs(x) ** 2))
    p_noise = p_sig / 10.0 ** (snr_db / 10.0)
    noise = np.sqrt(p_noise / 2.0) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + noise


def generate_signal(mod: str, n: int, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a (2, n) float32 I/Q array with AWGN at the requested SNR and a random carrier phase."""
    rng = rng or np.random.default_rng()
    x = clean_symbols(mod, n, rng) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    y = add_awgn(x, snr_db, rng)
    return np.stack([y.real, y.imag]).astype(np.float32)
