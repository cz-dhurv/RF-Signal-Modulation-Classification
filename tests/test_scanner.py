"""Tests for rfmc.scanner: wideband spectrum scanning, channelisation, and cognitive radio (Problem C)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from starlette.testclient import TestClient

from rfmc.api import create_app
from rfmc.errors import ValidationError
from rfmc.model import ClassifierService
from rfmc.openset import OpenSetDetector
from rfmc.scanner import (
    ChannelDetection,
    FreeBand,
    ScanResult,
    WidebandScanner,
    compute_psd,
    detect_active_bands,
    downconvert_and_resample,
    estimate_noise_floor_db,
    find_free_spectrum,
)
from rfmc.validation import validate_scan_payload
from tests.conftest import make_service


def _make_wideband_signal(
    sample_rate_hz: float = 20_000_000.0,
    n_samples: int = 4096,
    active_channels: list[tuple[float, float, str]] | None = None,
    snr_db: float = 15.0,
    noise_power: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate wideband I/Q signal with noise floor and optional carriers.

    active_channels: list of (center_freq_hz, bandwidth_hz, mod)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz

    # Background AWGN
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    sig = noise.copy()

    if active_channels:
        for f_center, bw, mod in active_channels:
            # Generate modulated baseband symbol stream
            n_sym = max(16, int(n_samples * (bw / sample_rate_hz)))
            if mod == "BPSK":
                syms = rng.choice([-1.0, 1.0], n_sym).astype(np.complex128)
            else:
                syms = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n_sym)))

            # Interpolate to signal length
            indices = np.linspace(0, n_sym - 1, n_samples)
            bb_real = np.interp(indices, np.arange(n_sym), syms.real)
            bb_imag = np.interp(indices, np.arange(n_sym), syms.imag)
            bb = bb_real + 1j * bb_imag

            # Scale for SNR
            sig_pwr = noise_power * (10.0 ** (snr_db / 10.0))
            bb = bb * np.sqrt(sig_pwr / max(float(np.mean(np.abs(bb) ** 2)), 1e-12))

            # Mix to center frequency
            mixer = np.exp(2j * math.pi * f_center * t)
            sig += bb * mixer

    iq = np.stack([sig.real, sig.imag]).astype(np.float32)
    return iq


# ─── Unit tests: PSD & Noise Floor ───────────────────────────────────


class TestPSDAndNoiseFloor:
    def test_compute_psd_shape_and_range(self):
        rng = np.random.default_rng(1)
        sig = rng.standard_normal(2048) + 1j * rng.standard_normal(2048)
        freqs, psd = compute_psd(sig, n_fft=512)
        assert len(freqs) == 512
        assert len(psd) == 512
        assert freqs[0] == -0.5
        assert np.isfinite(psd).all()

    def test_estimate_noise_floor_finite(self):
        rng = np.random.default_rng(2)
        psd = -50.0 + 2.0 * rng.standard_normal(512)
        nf = estimate_noise_floor_db(psd)
        assert math.isfinite(nf)
        assert abs(nf - (-50.0)) < 3.0


# ─── Unit tests: Active Band Detection ───────────────────────────────


class TestBandDetection:
    def test_pure_noise_detects_no_carriers(self):
        wideband = _make_wideband_signal(active_channels=None, n_samples=4096)
        sig_c = wideband[0] + 1j * wideband[1]
        freqs, psd = compute_psd(sig_c, n_fft=512)
        detections, nf = detect_active_bands(psd, freqs, sample_rate_hz=20_000_000.0, threshold_db=15.0)
        assert len(detections) == 0

    def test_detects_single_carrier(self):
        # Place carrier at +3 MHz with 2 MHz bandwidth in a 20 MHz band
        wideband = _make_wideband_signal(
            sample_rate_hz=20_000_000.0,
            n_samples=4096,
            active_channels=[(3_000_000.0, 2_000_000.0, "QPSK")],
            snr_db=25.0,
        )
        sig_c = wideband[0] + 1j * wideband[1]
        freqs, psd = compute_psd(sig_c, n_fft=1024)
        detections, nf = detect_active_bands(psd, freqs, sample_rate_hz=20_000_000.0, threshold_db=6.0)
        assert len(detections) >= 1
        # The strongest detection should be near 3 MHz
        strongest = max(detections, key=lambda d: d["power_db"])
        assert abs(strongest["center_freq_hz"] - 3_000_000.0) < 1_500_000.0


# ─── Unit tests: Channelisation & Downconversion ─────────────────────


class TestChannelisation:
    def test_downconvert_and_resample_shape(self):
        wideband = _make_wideband_signal(n_samples=2048)
        sig_c = wideband[0] + 1j * wideband[1]
        ch_iq = downconvert_and_resample(
            sig_c, center_freq_hz=2_000_000.0, bandwidth_hz=1_500_000.0,
            sample_rate_hz=20_000_000.0, target_len=1024
        )
        assert ch_iq.shape == (2, 1024)
        assert ch_iq.dtype == np.float32
        assert np.isfinite(ch_iq).all()
        # Unit power check
        power = float(np.mean(ch_iq[0] ** 2 + ch_iq[1] ** 2))
        assert abs(power - 1.0) < 0.05


# ─── Unit tests: Free Spectrum Identification ────────────────────────


class TestFreeSpectrum:
    def test_no_occupied_bands_returns_full_bandwidth(self):
        free = find_free_spectrum([], sample_rate_hz=20_000_000.0, min_bandwidth_hz=1_000_000.0)
        assert len(free) == 1
        assert free[0].bandwidth_hz == 20_000_000.0

    def test_channel_in_middle_splits_free_bands(self):
        # Occupied from -2 MHz to +2 MHz (4 MHz wide) in 20 MHz spectrum (-10 to +10 MHz)
        occupied = [(-2_000_000.0, 2_000_000.0)]
        free = find_free_spectrum(occupied, sample_rate_hz=20_000_000.0, min_bandwidth_hz=1_000_000.0)
        assert len(free) == 2
        # Left gap: -10 to -2 MHz = 8 MHz
        # Right gap: +2 to +10 MHz = 8 MHz
        for b in free:
            assert abs(b.bandwidth_hz - 8_000_000.0) < 100.0


# ─── Integration tests: WidebandScanner ──────────────────────────────


class TestWidebandScannerService:
    def test_scanner_end_to_end_with_classifier(self):
        svc = make_service()
        scanner = WidebandScanner(classifier=svc, default_sample_rate_hz=20_000_000.0)
        wideband = _make_wideband_signal(
            sample_rate_hz=20_000_000.0,
            n_samples=4096,
            active_channels=[(-4_000_000.0, 2_000_000.0, "BPSK"), (4_000_000.0, 2_000_000.0, "QPSK")],
            snr_db=20.0,
        )
        result = scanner.scan(wideband, sample_rate_hz=20_000_000.0, threshold_db=6.0)

        assert isinstance(result, ScanResult)
        assert result.sample_rate_hz == 20_000_000.0
        assert math.isfinite(result.noise_floor_db)
        assert len(result.channels) >= 1
        for ch in result.channels:
            assert ch.prediction is not None
            assert ch.prediction.label in svc.class_names or ch.prediction.label == "unknown"
        assert len(result.free_bands) >= 1
        assert result.recommended_free_band is not None

    def test_rejects_bad_input_shape(self):
        scanner = WidebandScanner()
        with pytest.raises(ValidationError):
            scanner.scan(np.zeros((3, 512)))


# ─── API endpoint tests: POST /v1/scan ───────────────────────────────


class TestScanAPI:
    def test_scan_happy_path(self, client, auth):
        iq = _make_wideband_signal(n_samples=1024).tolist()
        payload = {
            "iq": iq,
            "sample_rate_hz": 20_000_000.0,
            "threshold_db": 8.0,
            "min_free_bandwidth_hz": 1_000_000.0,
        }
        resp = client.post("/v1/scan", json=payload, headers=auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert "channels" in data
        assert "free_bands" in data
        assert "noise_floor_db" in data
        assert "spectral_occupancy_ratio" in data

    def test_scan_requires_auth(self, client):
        resp = client.post("/v1/scan", json={"iq": [[0.0] * 512, [0.0] * 512]})
        assert resp.status_code == 401

    def test_scan_rejects_unknown_fields(self, client, auth):
        resp = client.post(
            "/v1/scan",
            json={"iq": [[1.0] * 512, [0.5] * 512], "hacker_field": 123},
            headers=auth,
        )
        assert resp.status_code == 422
