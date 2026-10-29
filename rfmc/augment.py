"""Data augmentation for I/Q signals: channel impairments and transforms.

All augmentations operate on (2, N) float32 arrays (I, Q rows) and return the
same shape.  They are composable: chain them in any order.

These are training-time transforms only. The serving path (rfmc/) does not
import this module — only the training scripts in scripts/ use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _to_complex(iq: np.ndarray) -> np.ndarray:
    """(2, N) float -> (N,) complex128."""
    return iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)


def _to_iq(c: np.ndarray) -> np.ndarray:
    """(N,) complex128 -> (2, N) float32."""
    return np.stack([c.real, c.imag]).astype(np.float32)


# ─── Individual augmentations ────────────────────────────────────────


def random_phase_offset(iq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random constant carrier phase rotation ∈ [0, 2π)."""
    phi = rng.uniform(0, 2 * math.pi)
    return _to_iq(_to_complex(iq) * np.exp(1j * phi))


def random_frequency_offset(
    iq: np.ndarray, rng: np.random.Generator, *, max_offset: float = 0.01
) -> np.ndarray:
    """Apply a random linear frequency offset (normalised to sample rate).

    max_offset: maximum offset as a fraction of the sample rate.
    At 0.01, a 1 MHz sample rate gives up to ±10 kHz offset.
    """
    delta_f = rng.uniform(-max_offset, max_offset)
    n = iq.shape[1]
    t = np.arange(n, dtype=np.float64)
    return _to_iq(_to_complex(iq) * np.exp(2j * math.pi * delta_f * t))


def random_time_shift(
    iq: np.ndarray, rng: np.random.Generator, *, max_shift: int = 16
) -> np.ndarray:
    """Circular shift the signal by a random number of samples."""
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(iq, shift, axis=1).astype(np.float32)


def random_amplitude_scale(
    iq: np.ndarray, rng: np.random.Generator, *, low: float = 0.8, high: float = 1.2
) -> np.ndarray:
    """Scale the amplitude by a random factor."""
    return (iq * rng.uniform(low, high)).astype(np.float32)


def add_awgn(
    iq: np.ndarray, rng: np.random.Generator, *, snr_range: tuple[float, float] = (-4.0, 20.0)
) -> np.ndarray:
    """Add white Gaussian noise at a random SNR within the range (dB)."""
    snr_db = rng.uniform(snr_range[0], snr_range[1])
    c = _to_complex(iq)
    p_sig = float(np.mean(np.abs(c) ** 2))
    if p_sig == 0:
        return iq
    p_noise = p_sig / 10.0 ** (snr_db / 10.0)
    noise = np.sqrt(p_noise / 2.0) * (
        rng.standard_normal(len(c)) + 1j * rng.standard_normal(len(c))
    )
    return _to_iq(c + noise)


def rayleigh_fading(
    iq: np.ndarray, rng: np.random.Generator, *, n_taps: int = 4, max_delay: int = 8
) -> np.ndarray:
    """Apply a random Rayleigh fading channel with multipath.

    Creates n_taps with exponentially decaying power and random delays.
    """
    c = _to_complex(iq)
    n = len(c)
    out = np.zeros(n, dtype=np.complex128)

    # Generate tap gains (Rayleigh = magnitude of complex Gaussian)
    for k in range(n_taps):
        delay = min(int(rng.integers(0, max_delay + 1)), n - 1)
        # Exponential decay profile
        power = math.exp(-0.5 * k)
        gain = math.sqrt(power / 2.0) * (rng.standard_normal() + 1j * rng.standard_normal())
        out += gain * np.roll(c, delay)

    # Normalise to preserve signal power
    scale = math.sqrt(float(np.mean(np.abs(c) ** 2)) / max(float(np.mean(np.abs(out) ** 2)), 1e-12))
    return _to_iq(out * scale)


def rician_fading(
    iq: np.ndarray, rng: np.random.Generator, *, k_factor: float = 4.0, n_taps: int = 3
) -> np.ndarray:
    """Apply Rician fading: dominant LOS component + scattered multipath.

    k_factor: ratio of LOS power to scattered power (linear, not dB).
    """
    c = _to_complex(iq)
    n = len(c)

    # LOS component with random phase
    los_power = k_factor / (k_factor + 1.0)
    scat_power = 1.0 / (k_factor + 1.0)
    los_phase = rng.uniform(0, 2 * math.pi)
    los = math.sqrt(los_power) * np.exp(1j * los_phase) * c

    # Scattered component (Rayleigh)
    scattered = np.zeros(n, dtype=np.complex128)
    for k in range(n_taps):
        delay = int(rng.integers(0, 5))
        gain = math.sqrt(scat_power / (2.0 * n_taps)) * (
            rng.standard_normal() + 1j * rng.standard_normal()
        )
        scattered += gain * np.roll(c, delay)

    result = los + scattered
    # Normalise
    scale = math.sqrt(float(np.mean(np.abs(c) ** 2)) / max(float(np.mean(np.abs(result) ** 2)), 1e-12))
    return _to_iq(result * scale)


def iq_imbalance(
    iq: np.ndarray, rng: np.random.Generator,
    *, max_gain_db: float = 1.0, max_phase_deg: float = 5.0
) -> np.ndarray:
    """Simulate I/Q gain and phase imbalance."""
    gain_db = rng.uniform(-max_gain_db, max_gain_db)
    phase_deg = rng.uniform(-max_phase_deg, max_phase_deg)
    gain = 10.0 ** (gain_db / 20.0)
    phase_rad = math.radians(phase_deg)
    result = iq.copy().astype(np.float64)
    result[0] = iq[0] * gain
    result[1] = iq[0] * math.sin(phase_rad) + iq[1] * math.cos(phase_rad)
    return result.astype(np.float32)


# ─── Augmentation pipeline ───────────────────────────────────────────


@dataclass
class AugmentationConfig:
    """Configuration for the augmentation pipeline.

    Each probability controls how often that augmentation is applied.
    Set to 0.0 to disable, 1.0 to always apply.
    """
    phase_offset_prob: float = 0.5
    freq_offset_prob: float = 0.3
    freq_offset_max: float = 0.01
    time_shift_prob: float = 0.3
    time_shift_max: int = 16
    amplitude_scale_prob: float = 0.3
    awgn_prob: float = 0.3
    awgn_snr_range: tuple[float, float] = (-4.0, 20.0)
    rayleigh_prob: float = 0.2
    rayleigh_taps: int = 4
    rician_prob: float = 0.2
    rician_k_factor: float = 4.0
    iq_imbalance_prob: float = 0.2


def augment(iq: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig | None = None) -> np.ndarray:
    """Apply a random subset of augmentations to a (2, N) I/Q array.

    Each augmentation is applied independently with its configured probability.
    """
    cfg = cfg or AugmentationConfig()

    if rng.random() < cfg.phase_offset_prob:
        iq = random_phase_offset(iq, rng)
    if rng.random() < cfg.freq_offset_prob:
        iq = random_frequency_offset(iq, rng, max_offset=cfg.freq_offset_max)
    if rng.random() < cfg.time_shift_prob:
        iq = random_time_shift(iq, rng, max_shift=cfg.time_shift_max)
    if rng.random() < cfg.amplitude_scale_prob:
        iq = random_amplitude_scale(iq, rng)
    if rng.random() < cfg.awgn_prob:
        iq = add_awgn(iq, rng, snr_range=cfg.awgn_snr_range)
    if rng.random() < cfg.rayleigh_prob:
        iq = rayleigh_fading(iq, rng, n_taps=cfg.rayleigh_taps)
    if rng.random() < cfg.rician_prob:
        iq = rician_fading(iq, rng, k_factor=cfg.rician_k_factor)
    if rng.random() < cfg.iq_imbalance_prob:
        iq = iq_imbalance(iq, rng)

    return iq
