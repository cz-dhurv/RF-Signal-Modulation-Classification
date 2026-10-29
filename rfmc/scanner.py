"""Wideband spectrum scanner: energy detection, channelisation, and cognitive radio (Problem C).

Takes wideband I/Q captures (e.g. 10–50 MHz bandwidth), detects active emitters
above an adaptive noise floor, channelises and downconverts each slice to baseband,
classifies each channel via ClassifierService, and identifies free spectrum bands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .errors import ValidationError
from .validation import normalize_power

if TYPE_CHECKING:
    from .model import ClassifierService, Prediction


@dataclass(frozen=True)
class ChannelDetection:
    """An active channel detected in the wideband spectrum."""
    channel_id: int
    center_freq_hz: float     # relative to carrier (0 Hz = center of capture)
    bandwidth_hz: float
    power_db: float
    snr_db: float             # estimated SNR above local noise floor
    occupied: bool
    prediction: Prediction | None = None


@dataclass(frozen=True)
class FreeBand:
    """An unoccupied frequency range suitable for cognitive radio transmission."""
    start_freq_hz: float
    stop_freq_hz: float
    center_freq_hz: float
    bandwidth_hz: float
    noise_power_db: float


@dataclass(frozen=True)
class ScanResult:
    """Full wideband spectrum scan report."""
    sample_rate_hz: float
    noise_floor_db: float
    total_bandwidth_hz: float
    occupied_bandwidth_hz: float
    spectral_occupancy_ratio: float
    channels: list[ChannelDetection] = field(default_factory=list)
    free_bands: list[FreeBand] = field(default_factory=list)
    recommended_free_band: FreeBand | None = None


def compute_psd(
    signal: np.ndarray,
    n_fft: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute average power spectral density (PSD) using Welch's method (pure NumPy).

    signal: 1D complex array.
    Returns (freqs_norm, psd_db) where freqs_norm is in [-0.5, 0.5).
    """
    n = len(signal)
    if n < n_fft:
        n_fft = max(128, 2 ** int(math.log2(n)))
    hop = n_fft // 2
    window = np.hanning(n_fft)
    win_power = float(np.sum(window ** 2))

    psd_accum = np.zeros(n_fft, dtype=np.float64)
    count = 0
    for start in range(0, n - n_fft + 1, hop):
        segment = signal[start : start + n_fft] * window
        fft_val = np.fft.fftshift(np.fft.fft(segment, n=n_fft))
        psd_accum += (np.abs(fft_val) ** 2) / max(win_power, 1e-12)
        count += 1

    if count == 0:
        segment = np.pad(signal, (0, n_fft - n)) * window
        fft_val = np.fft.fftshift(np.fft.fft(segment, n=n_fft))
        psd_accum = (np.abs(fft_val) ** 2) / max(win_power, 1e-12)
        count = 1

    psd_avg = psd_accum / count
    freqs = np.linspace(-0.5, 0.5, n_fft, endpoint=False)
    psd_db = 10.0 * np.log10(np.maximum(psd_avg, 1e-15))
    return freqs, psd_db


def estimate_noise_floor_db(psd_db: np.ndarray) -> float:
    """Robust noise floor estimation using bottom 40th percentile trimmed mean."""
    sorted_psd = np.sort(psd_db)
    trim_idx = int(len(sorted_psd) * 0.40)
    return float(np.mean(sorted_psd[: max(trim_idx, 1)]))


def detect_active_bands(
    psd_db: np.ndarray,
    freqs_norm: np.ndarray,
    sample_rate_hz: float,
    threshold_db: float = 6.0,
    min_bins: int = 8,
) -> tuple[list[dict], float]:
    """Detect contiguous spectral regions rising above noise_floor + threshold_db."""
    noise_floor_db = estimate_noise_floor_db(psd_db)
    thresh = noise_floor_db + threshold_db
    is_active = psd_db > thresh

    # Find contiguous active runs
    runs = []
    in_run = False
    start_idx = 0
    for i, active in enumerate(is_active):
        if active and not in_run:
            in_run = True
            start_idx = i
        elif not active and in_run:
            in_run = False
            if (i - start_idx) >= min_bins:
                runs.append((start_idx, i))
    if in_run and (len(is_active) - start_idx) >= min_bins:
        runs.append((start_idx, len(is_active)))

    bin_width_hz = sample_rate_hz / len(psd_db)
    detections = []
    for cid, (s, e) in enumerate(runs):
        band_psd = psd_db[s:e]
        peak_snr = float(np.max(band_psd) - noise_floor_db)
        avg_pwr = float(10.0 * np.log10(np.mean(10.0 ** (band_psd / 10.0))))
        f_start = float(freqs_norm[s] * sample_rate_hz)
        f_stop = float(freqs_norm[e - 1] * sample_rate_hz + bin_width_hz)
        bw = max(f_stop - f_start, bin_width_hz)
        f_center = (f_start + f_stop) / 2.0

        detections.append({
            "channel_id": cid,
            "start_freq_hz": f_start,
            "stop_freq_hz": f_stop,
            "center_freq_hz": f_center,
            "bandwidth_hz": bw,
            "power_db": avg_pwr,
            "snr_db": peak_snr,
        })

    return detections, noise_floor_db


def downconvert_and_resample(
    signal: np.ndarray,
    center_freq_hz: float,
    bandwidth_hz: float,
    sample_rate_hz: float,
    target_len: int = 1024,
) -> np.ndarray:
    """Digital Down-Conversion (DDC) and bandpass extraction to baseband (2, target_len)."""
    n = len(signal)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    # 1. Frequency shift to center baseband (0 Hz)
    mixer = np.exp(-2j * math.pi * center_freq_hz * t)
    baseband = signal * mixer

    # 2. Windowed sinc low-pass filter with cutoff at bandwidth/2
    cutoff = min(bandwidth_hz / 2.0, sample_rate_hz * 0.45)
    f_norm = cutoff / sample_rate_hz
    n_taps = 65
    t_filter = np.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=np.float64)
    h = 2 * f_norm * np.sinc(2 * f_norm * t_filter) * np.hamming(n_taps)
    h = h / max(np.sum(h), 1e-12)

    filtered = np.convolve(baseband, h, mode="same")

    # 3. Resample / decimate to target_len
    if len(filtered) >= target_len:
        # Uniform sampling across the window
        indices = np.linspace(0, len(filtered) - 1, target_len)
        real_part = np.interp(indices, np.arange(len(filtered)), filtered.real)
        imag_part = np.interp(indices, np.arange(len(filtered)), filtered.imag)
        out = real_part + 1j * imag_part
    else:
        out = np.pad(filtered, (0, target_len - len(filtered)))

    # 4. Return power-normalized (2, target_len) float32 array
    iq = np.stack([out.real, out.imag]).astype(np.float32)
    return normalize_power(iq).astype(np.float32)


def find_free_spectrum(
    occupied_intervals: list[tuple[float, float]],
    sample_rate_hz: float,
    min_bandwidth_hz: float = 1_000_000.0,
    noise_floor_db: float = -80.0,
) -> list[FreeBand]:
    """Identify contiguous free spectral bands not occupied by active channels."""
    f_min = -sample_rate_hz / 2.0
    f_max = sample_rate_hz / 2.0

    # Sort and merge occupied intervals
    if not occupied_intervals:
        return [
            FreeBand(
                start_freq_hz=f_min,
                stop_freq_hz=f_max,
                center_freq_hz=0.0,
                bandwidth_hz=sample_rate_hz,
                noise_power_db=noise_floor_db,
            )
        ]

    sorted_ivs = sorted(occupied_intervals, key=lambda x: x[0])
    merged = []
    curr_s, curr_e = sorted_ivs[0]
    for s, e in sorted_ivs[1:]:
        if s <= curr_e:
            curr_e = max(curr_e, e)
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = s, e
    merged.append((curr_s, curr_e))

    # Invert merged intervals
    free_bands = []
    prev_end = f_min
    for s, e in merged:
        gap = s - prev_end
        if gap >= min_bandwidth_hz:
            free_bands.append(
                FreeBand(
                    start_freq_hz=prev_end,
                    stop_freq_hz=s,
                    center_freq_hz=(prev_end + s) / 2.0,
                    bandwidth_hz=gap,
                    noise_power_db=noise_floor_db,
                )
            )
        prev_end = max(prev_end, e)

    if (f_max - prev_end) >= min_bandwidth_hz:
        free_bands.append(
            FreeBand(
                start_freq_hz=prev_end,
                stop_freq_hz=f_max,
                center_freq_hz=(prev_end + f_max) / 2.0,
                bandwidth_hz=f_max - prev_end,
                noise_power_db=noise_floor_db,
            )
        )

    # Sort by widest bandwidth descending
    return sorted(free_bands, key=lambda b: -b.bandwidth_hz)


class WidebandScanner:
    """Wideband spectrum scanner and cognitive radio channeliser."""

    def __init__(
        self,
        classifier: ClassifierService | None = None,
        default_sample_rate_hz: float = 20_000_000.0,
        energy_threshold_db: float = 6.0,
        target_window_len: int = 1024,
    ):
        self.classifier = classifier
        self.default_sample_rate_hz = default_sample_rate_hz
        self.energy_threshold_db = energy_threshold_db
        self.target_window_len = target_window_len

    def scan(
        self,
        wideband_iq: np.ndarray,
        sample_rate_hz: float | None = None,
        threshold_db: float | None = None,
        min_free_bandwidth_hz: float = 1_000_000.0,
    ) -> ScanResult:
        """Scan a wideband (2, N) I/Q array, detect emitters, classify, and find free spectrum."""
        wideband_iq = np.asarray(wideband_iq)
        if wideband_iq.ndim != 2 or wideband_iq.shape[0] != 2:
            raise ValidationError("wideband_iq must have shape (2, N)")

        sample_rate = sample_rate_hz or self.default_sample_rate_hz
        thresh_db = threshold_db if threshold_db is not None else self.energy_threshold_db

        # Complex baseband signal
        sig_c = wideband_iq[0].astype(np.float64) + 1j * wideband_iq[1].astype(np.float64)

        # 1. Compute PSD
        freqs_norm, psd_db = compute_psd(sig_c, n_fft=min(2048, len(sig_c)))

        # 2. Detect active bands
        active_raw, noise_floor_db = detect_active_bands(
            psd_db, freqs_norm, sample_rate, threshold_db=thresh_db
        )

        channels: list[ChannelDetection] = []
        occupied_intervals = []
        total_occupied_bw = 0.0

        for item in active_raw:
            f_center = item["center_freq_hz"]
            bw = item["bandwidth_hz"]
            occupied_intervals.append((item["start_freq_hz"], item["stop_freq_hz"]))
            total_occupied_bw += bw

            # Channelise & classify if classifier is provided
            pred = None
            if self.classifier is not None:
                channel_iq = downconvert_and_resample(
                    sig_c,
                    center_freq_hz=f_center,
                    bandwidth_hz=bw,
                    sample_rate_hz=sample_rate,
                    target_len=self.target_window_len,
                )
                try:
                    pred = self.classifier.classify(channel_iq)
                except Exception:
                    pred = None

            channels.append(
                ChannelDetection(
                    channel_id=item["channel_id"],
                    center_freq_hz=f_center,
                    bandwidth_hz=bw,
                    power_db=item["power_db"],
                    snr_db=item["snr_db"],
                    occupied=True,
                    prediction=pred,
                )
            )

        # 3. Find free spectrum for cognitive radio
        free_bands = find_free_spectrum(
            occupied_intervals,
            sample_rate_hz=sample_rate,
            min_bandwidth_hz=min_free_bandwidth_hz,
            noise_floor_db=noise_floor_db,
        )

        rec_band = free_bands[0] if free_bands else None
        ratio = total_occupied_bw / sample_rate

        return ScanResult(
            sample_rate_hz=sample_rate,
            noise_floor_db=noise_floor_db,
            total_bandwidth_hz=sample_rate,
            occupied_bandwidth_hz=total_occupied_bw,
            spectral_occupancy_ratio=ratio,
            channels=channels,
            free_bands=free_bands,
            recommended_free_band=rec_band,
        )

