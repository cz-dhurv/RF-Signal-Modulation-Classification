"""Advanced signal simulator: realistic channel impairments for Phase 4 robustness testing.

Goes beyond the basic synth.py (1 sample/symbol, AWGN only) by adding:
- Pulse shaping (raised-cosine filter)
- Rayleigh / Rician fading channels
- Carrier frequency and sample clock offsets
- Multipath propagation
- Held-out unknown modulations for open-set evaluation

Output format: (2, N) float32 I/Q arrays, same as the rest of the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ─── Modulation pool ─────────────────────────────────────────────────

# Known modulations (shared with synth.py / training)
KNOWN_MODS = ("BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "QAM256")

# Unknown modulations for open-set evaluation — these should NOT appear in training
UNKNOWN_MODS = ("OOK", "GMSK", "OFDM_QPSK", "CHIRP", "NOISE")


def _qam_constellation(order: int) -> np.ndarray:
    """Generate a square QAM constellation with unit average power."""
    k = int(math.sqrt(order))
    assert k * k == order, f"QAM-{order} requires a perfect square"
    levels = np.arange(k) - (k - 1) / 2.0
    grid = np.array([(i + 1j * q) for i in levels for q in levels])
    return grid / math.sqrt(float(np.mean(np.abs(grid) ** 2)))


# Pre-compute constellations
_CONSTELLATIONS = {
    "BPSK": np.array([-1.0 + 0j, 1.0 + 0j]),
    "QPSK": np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4))),
    "8PSK": np.exp(2j * np.pi * np.arange(8) / 8),
    "QAM16": _qam_constellation(16),
    "QAM64": _qam_constellation(64),
    "QAM256": _qam_constellation(256),
}


def _generate_symbols(mod: str, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate complex baseband symbols for a given modulation."""
    if mod in _CONSTELLATIONS:
        return rng.choice(_CONSTELLATIONS[mod], n_symbols)
    if mod == "OOK":
        return rng.choice([0.0 + 0j, 1.0 + 0j], n_symbols)
    if mod == "GMSK":
        # Approximate GMSK as filtered MSK (simplified)
        bits = rng.choice([-1.0, 1.0], n_symbols)
        phase = np.cumsum(bits) * (np.pi / 2)
        return np.exp(1j * phase)
    if mod == "OFDM_QPSK":
        # Simplified OFDM: N-point IFFT of QPSK symbols
        n_carriers = min(64, n_symbols)
        qpsk = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n_carriers)))
        time_domain = np.fft.ifft(qpsk, n=n_carriers) * math.sqrt(n_carriers)
        # Repeat to fill requested length
        reps = int(math.ceil(n_symbols / n_carriers))
        return np.tile(time_domain, reps)[:n_symbols]
    if mod == "CHIRP":
        # Linear frequency-modulated chirp
        t = np.linspace(0, 1, n_symbols)
        return np.exp(2j * np.pi * (0.5 * rng.uniform(0.1, 0.5) * t ** 2))
    if mod == "NOISE":
        # Pure noise — should always be detected as unknown
        return (rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)) / math.sqrt(2)
    raise ValueError(f"Unknown modulation: {mod}")


# ─── Pulse shaping ───────────────────────────────────────────────────


def raised_cosine_filter(n_taps: int = 65, beta: float = 0.35, sps: int = 4) -> np.ndarray:
    """Design a raised-cosine FIR filter."""
    t = np.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0
        elif abs(abs(2 * beta * ti) - 1.0) < 1e-12:
            h[i] = (np.pi / 4) * np.sinc(1.0 / (2 * beta))
        else:
            h[i] = np.sinc(ti) * np.cos(np.pi * beta * ti) / (1 - (2 * beta * ti) ** 2)
    return h / np.sqrt(np.sum(h ** 2))


def upsample_and_filter(symbols: np.ndarray, sps: int = 4, beta: float = 0.35) -> np.ndarray:
    """Upsample symbols and apply raised-cosine pulse shaping."""
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    h = raised_cosine_filter(sps=sps, beta=beta)
    return np.convolve(upsampled, h, mode="same")


# ─── Channel impairments ─────────────────────────────────────────────


def apply_frequency_offset(signal: np.ndarray, offset: float) -> np.ndarray:
    """Apply a constant frequency offset (normalised to sample rate)."""
    t = np.arange(len(signal), dtype=np.float64)
    return signal * np.exp(2j * np.pi * offset * t)


def apply_clock_offset(signal: np.ndarray, ppm: float) -> np.ndarray:
    """Apply a sample clock offset (resample via linear interpolation).

    ppm: parts per million offset.
    """
    if abs(ppm) < 0.01:
        return signal
    n = len(signal)
    rate = 1.0 + ppm * 1e-6
    new_indices = np.arange(n, dtype=np.float64) * rate
    valid = new_indices < n
    new_indices = new_indices[valid]
    idx = np.floor(new_indices).astype(int)
    frac = new_indices - idx
    idx = np.minimum(idx, n - 2)
    return signal[idx] * (1 - frac) + signal[idx + 1] * frac


def apply_multipath(
    signal: np.ndarray, rng: np.random.Generator, n_paths: int = 3, max_delay: int = 10
) -> np.ndarray:
    """Apply a random multipath channel."""
    out = signal.copy()
    for _ in range(n_paths - 1):
        delay = int(rng.integers(1, max_delay + 1))
        # Exponentially decaying path gain
        attn = rng.exponential(0.3)
        phase = rng.uniform(0, 2 * np.pi)
        out += attn * np.exp(1j * phase) * np.roll(signal, delay)
    # Normalise power
    scale = np.sqrt(np.mean(np.abs(signal) ** 2) / max(np.mean(np.abs(out) ** 2), 1e-12))
    return out * scale


# ─── Complete signal generator ───────────────────────────────────────


@dataclass
class SimulatorConfig:
    """Configuration for the advanced simulator."""
    n_samples: int = 1024       # output length in samples
    sps: int = 4                # samples per symbol for pulse shaping
    beta: float = 0.35          # raised-cosine rolloff factor
    use_pulse_shaping: bool = True
    snr_range: tuple[float, float] = (-6.0, 20.0)  # dB
    freq_offset_range: tuple[float, float] = (-0.005, 0.005)
    clock_offset_ppm_range: tuple[float, float] = (-50.0, 50.0)
    multipath_prob: float = 0.5
    multipath_n_paths: int = 3
    rayleigh_prob: float = 0.3
    rayleigh_taps: int = 4
    rician_prob: float = 0.2
    rician_k_factor: float = 4.0


def generate_advanced_signal(
    mod: str,
    rng: np.random.Generator,
    cfg: SimulatorConfig | None = None,
) -> np.ndarray:
    """Generate a realistic I/Q signal with channel impairments.

    Returns (2, N) float32 array, power-normalised.
    """
    cfg = cfg or SimulatorConfig()

    # Number of symbols needed
    n_sym = cfg.n_samples // cfg.sps if cfg.use_pulse_shaping else cfg.n_samples
    n_sym = max(n_sym + 20, 32)  # extra for filter settling

    # Generate symbols
    symbols = _generate_symbols(mod, n_sym, rng)

    # Pulse shaping
    if cfg.use_pulse_shaping and mod not in ("OFDM_QPSK", "NOISE"):
        signal = upsample_and_filter(symbols, sps=cfg.sps, beta=cfg.beta)
    else:
        signal = symbols.astype(np.complex128)

    # Random carrier phase
    signal = signal * np.exp(1j * rng.uniform(0, 2 * np.pi))

    # Frequency offset
    freq_offset = rng.uniform(*cfg.freq_offset_range)
    signal = apply_frequency_offset(signal, freq_offset)

    # Clock offset
    clock_ppm = rng.uniform(*cfg.clock_offset_ppm_range)
    signal = apply_clock_offset(signal, clock_ppm)

    # Multipath
    if rng.random() < cfg.multipath_prob:
        signal = apply_multipath(signal, rng, cfg.multipath_n_paths)

    # Rayleigh fading
    if rng.random() < cfg.rayleigh_prob:
        from .augment import rayleigh_fading
        iq_temp = np.stack([signal.real, signal.imag]).astype(np.float32)
        iq_temp = rayleigh_fading(iq_temp, rng, n_taps=cfg.rayleigh_taps)
        signal = iq_temp[0].astype(np.float64) + 1j * iq_temp[1].astype(np.float64)

    # Rician fading
    if rng.random() < cfg.rician_prob:
        from .augment import rician_fading
        iq_temp = np.stack([signal.real, signal.imag]).astype(np.float32)
        iq_temp = rician_fading(iq_temp, rng, k_factor=cfg.rician_k_factor)
        signal = iq_temp[0].astype(np.float64) + 1j * iq_temp[1].astype(np.float64)

    # AWGN
    snr_db = rng.uniform(*cfg.snr_range)
    p_sig = float(np.mean(np.abs(signal) ** 2))
    if p_sig > 0:
        p_noise = p_sig / 10.0 ** (snr_db / 10.0)
        noise = np.sqrt(p_noise / 2.0) * (
            rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
        )
        signal = signal + noise

    # Trim/pad to exact output length
    if len(signal) >= cfg.n_samples:
        start = int(rng.integers(0, max(len(signal) - cfg.n_samples, 1)))
        signal = signal[start : start + cfg.n_samples]
    else:
        signal = np.pad(signal, (0, cfg.n_samples - len(signal)))

    # To I/Q float32
    iq = np.stack([signal.real, signal.imag]).astype(np.float32)

    # Power normalise
    peak = float(np.max(np.abs(iq)))
    if peak > 0:
        iq = iq / peak
        power = float(np.mean(iq[0] ** 2 + iq[1] ** 2))
        if power > 0:
            iq = (iq / math.sqrt(power)).astype(np.float32)

    return iq


def generate_dataset(
    mods: list[str],
    n_per_class_per_snr: int = 100,
    snr_values: list[int] | None = None,
    rng: np.random.Generator | None = None,
    cfg: SimulatorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate a complete simulated dataset.

    Returns (X, y, snr, class_names) matching the .npz format.
    """
    snr_values = snr_values or list(range(-6, 22, 2))
    rng = rng or np.random.default_rng()
    cfg = cfg or SimulatorConfig()

    all_x, all_y, all_snr = [], [], []
    for snr_db in snr_values:
        sim_cfg = SimulatorConfig(
            n_samples=cfg.n_samples,
            sps=cfg.sps,
            beta=cfg.beta,
            use_pulse_shaping=cfg.use_pulse_shaping,
            snr_range=(snr_db, snr_db),  # fixed SNR
            freq_offset_range=cfg.freq_offset_range,
            clock_offset_ppm_range=cfg.clock_offset_ppm_range,
            multipath_prob=cfg.multipath_prob,
            multipath_n_paths=cfg.multipath_n_paths,
            rayleigh_prob=cfg.rayleigh_prob,
            rayleigh_taps=cfg.rayleigh_taps,
            rician_prob=cfg.rician_prob,
            rician_k_factor=cfg.rician_k_factor,
        )
        for class_idx, mod in enumerate(mods):
            for _ in range(n_per_class_per_snr):
                x = generate_advanced_signal(mod, rng, sim_cfg)
                all_x.append(x)
                all_y.append(class_idx)
                all_snr.append(snr_db)

    X = np.stack(all_x)
    y = np.array(all_y, dtype=np.int32)
    snr = np.array(all_snr, dtype=np.int32)
    return X, y, snr, list(mods)
