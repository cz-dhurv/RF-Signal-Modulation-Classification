"""Tests for rfmc.simulator and rfmc.augment — Phase 4 signal generation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfmc.augment import (
    AugmentationConfig,
    add_awgn,
    augment,
    iq_imbalance,
    random_amplitude_scale,
    random_frequency_offset,
    random_phase_offset,
    random_time_shift,
    rayleigh_fading,
    rician_fading,
)
from rfmc.simulator import (
    KNOWN_MODS,
    UNKNOWN_MODS,
    SimulatorConfig,
    _generate_symbols,
    _qam_constellation,
    apply_clock_offset,
    apply_frequency_offset,
    apply_multipath,
    generate_advanced_signal,
    generate_dataset,
    raised_cosine_filter,
    upsample_and_filter,
)


RNG = np.random.default_rng(42)


def _fresh_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_iq(n: int = 256, seed: int = 0) -> np.ndarray:
    """Generate a simple (2, N) I/Q test signal."""
    rng = _fresh_rng(seed)
    iq = rng.standard_normal((2, n)).astype(np.float32)
    # Normalise
    power = float(np.mean(iq[0] ** 2 + iq[1] ** 2))
    if power > 0:
        iq = iq / math.sqrt(power)
    return iq.astype(np.float32)


# ─── Simulator: constellations ────────────────────────────────────────


class TestConstellations:
    def test_qam16_size(self):
        c = _qam_constellation(16)
        assert len(c) == 16

    def test_qam64_size(self):
        c = _qam_constellation(64)
        assert len(c) == 64

    def test_qam256_size(self):
        c = _qam_constellation(256)
        assert len(c) == 256

    def test_qam_unit_power(self):
        for order in [16, 64, 256]:
            c = _qam_constellation(order)
            power = float(np.mean(np.abs(c) ** 2))
            assert abs(power - 1.0) < 0.05, f"QAM-{order} power={power}"

    def test_non_square_raises(self):
        with pytest.raises(AssertionError):
            _qam_constellation(12)


# ─── Simulator: symbol generation ────────────────────────────────────


class TestSymbolGeneration:
    @pytest.mark.parametrize("mod", list(KNOWN_MODS) + list(UNKNOWN_MODS))
    def test_generate_correct_length(self, mod):
        symbols = _generate_symbols(mod, 128, _fresh_rng())
        assert len(symbols) == 128

    @pytest.mark.parametrize("mod", list(KNOWN_MODS))
    def test_known_symbols_finite(self, mod):
        symbols = _generate_symbols(mod, 256, _fresh_rng())
        assert np.isfinite(symbols).all()

    @pytest.mark.parametrize("mod", list(UNKNOWN_MODS))
    def test_unknown_symbols_finite(self, mod):
        symbols = _generate_symbols(mod, 256, _fresh_rng())
        assert np.isfinite(symbols).all()

    def test_unknown_modulation_raises(self):
        with pytest.raises(ValueError, match="Unknown modulation"):
            _generate_symbols("NOTAMOD", 128, _fresh_rng())


# ─── Simulator: pulse shaping ────────────────────────────────────────


class TestPulseShaping:
    def test_raised_cosine_shape(self):
        h = raised_cosine_filter(n_taps=65, beta=0.35, sps=4)
        assert len(h) == 65

    def test_raised_cosine_normalised(self):
        h = raised_cosine_filter()
        energy = float(np.sum(h ** 2))
        assert abs(energy - 1.0) < 0.01

    def test_upsample_length(self):
        symbols = np.ones(32, dtype=np.complex128)
        out = upsample_and_filter(symbols, sps=4)
        assert len(out) == 128

    def test_upsample_finite(self):
        rng = _fresh_rng()
        symbols = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        out = upsample_and_filter(symbols)
        assert np.isfinite(out).all()


# ─── Simulator: channel impairments ──────────────────────────────────


class TestChannelImpairments:
    def test_frequency_offset_preserves_length(self):
        signal = np.ones(128, dtype=np.complex128)
        out = apply_frequency_offset(signal, 0.01)
        assert len(out) == 128
        assert np.isfinite(out).all()

    def test_clock_offset_small_ppm_noop(self):
        signal = np.ones(128, dtype=np.complex128)
        out = apply_clock_offset(signal, 0.001)
        np.testing.assert_array_equal(out, signal)

    def test_clock_offset_resamples(self):
        rng = _fresh_rng()
        signal = rng.standard_normal(256) + 1j * rng.standard_normal(256)
        out = apply_clock_offset(signal, 50.0)
        assert len(out) > 0
        assert np.isfinite(out).all()

    def test_multipath_preserves_power(self):
        rng = _fresh_rng()
        signal = rng.standard_normal(256) + 1j * rng.standard_normal(256)
        p_in = float(np.mean(np.abs(signal) ** 2))
        out = apply_multipath(signal, rng, n_paths=3)
        p_out = float(np.mean(np.abs(out) ** 2))
        # Power should be roughly preserved
        assert abs(p_out - p_in) / p_in < 0.3


# ─── Simulator: full signal generation ───────────────────────────────


class TestAdvancedSignalGeneration:
    @pytest.mark.parametrize("mod", list(KNOWN_MODS) + list(UNKNOWN_MODS))
    def test_output_shape(self, mod):
        iq = generate_advanced_signal(mod, _fresh_rng())
        assert iq.shape == (2, 1024)
        assert iq.dtype == np.float32

    @pytest.mark.parametrize("mod", list(KNOWN_MODS))
    def test_output_finite(self, mod):
        iq = generate_advanced_signal(mod, _fresh_rng())
        assert np.isfinite(iq).all()

    def test_custom_config(self):
        cfg = SimulatorConfig(n_samples=512, sps=2, multipath_prob=1.0)
        iq = generate_advanced_signal("QPSK", _fresh_rng(), cfg)
        assert iq.shape == (2, 512)

    def test_no_pulse_shaping(self):
        cfg = SimulatorConfig(use_pulse_shaping=False, n_samples=256)
        iq = generate_advanced_signal("BPSK", _fresh_rng(), cfg)
        assert iq.shape == (2, 256)

    def test_deterministic_with_seed(self):
        iq1 = generate_advanced_signal("QPSK", _fresh_rng(123))
        iq2 = generate_advanced_signal("QPSK", _fresh_rng(123))
        np.testing.assert_array_equal(iq1, iq2)

    def test_different_seeds_differ(self):
        iq1 = generate_advanced_signal("QPSK", _fresh_rng(1))
        iq2 = generate_advanced_signal("QPSK", _fresh_rng(2))
        assert not np.allclose(iq1, iq2)


# ─── Simulator: dataset generation ───────────────────────────────────


class TestDatasetGeneration:
    def test_dataset_shapes(self):
        X, y, snr, names = generate_dataset(
            mods=["BPSK", "QPSK"], n_per_class_per_snr=5,
            snr_values=[0, 10], rng=_fresh_rng(),
        )
        assert X.shape == (20, 2, 1024)  # 2 mods × 2 SNRs × 5 each
        assert y.shape == (20,)
        assert snr.shape == (20,)
        assert names == ["BPSK", "QPSK"]

    def test_dataset_class_labels(self):
        X, y, snr, names = generate_dataset(
            mods=["BPSK", "QPSK", "8PSK"], n_per_class_per_snr=3,
            snr_values=[0], rng=_fresh_rng(),
        )
        assert set(y.tolist()) == {0, 1, 2}

    def test_dataset_snr_labels(self):
        X, y, snr, names = generate_dataset(
            mods=["BPSK"], n_per_class_per_snr=2,
            snr_values=[-6, 0, 10, 20], rng=_fresh_rng(),
        )
        assert set(snr.tolist()) == {-6, 0, 10, 20}


# ─── Augmentation: individual transforms ─────────────────────────────


class TestAugmentations:
    def test_phase_offset_preserves_shape(self):
        iq = _make_iq()
        out = random_phase_offset(iq, _fresh_rng())
        assert out.shape == iq.shape
        assert out.dtype == np.float32

    def test_phase_offset_preserves_power(self):
        iq = _make_iq()
        p_in = float(np.mean(iq ** 2))
        out = random_phase_offset(iq, _fresh_rng())
        p_out = float(np.mean(out ** 2))
        assert abs(p_out - p_in) / max(p_in, 1e-12) < 0.01

    def test_frequency_offset_shape(self):
        iq = _make_iq()
        out = random_frequency_offset(iq, _fresh_rng(), max_offset=0.01)
        assert out.shape == iq.shape
        assert np.isfinite(out).all()

    def test_time_shift_circular(self):
        iq = _make_iq()
        out = random_time_shift(iq, _fresh_rng(), max_shift=16)
        assert out.shape == iq.shape

    def test_amplitude_scale_range(self):
        iq = _make_iq()
        rng = _fresh_rng()
        out = random_amplitude_scale(iq, rng, low=0.5, high=2.0)
        assert out.shape == iq.shape

    def test_awgn_preserves_shape(self):
        iq = _make_iq()
        out = add_awgn(iq, _fresh_rng())
        assert out.shape == iq.shape

    def test_awgn_zero_power_noop(self):
        iq = np.zeros((2, 128), dtype=np.float32)
        out = add_awgn(iq, _fresh_rng())
        np.testing.assert_array_equal(out, iq)

    def test_rayleigh_shape(self):
        iq = _make_iq()
        out = rayleigh_fading(iq, _fresh_rng())
        assert out.shape == iq.shape
        assert out.dtype == np.float32

    def test_rician_shape(self):
        iq = _make_iq()
        out = rician_fading(iq, _fresh_rng())
        assert out.shape == iq.shape

    def test_iq_imbalance_shape(self):
        iq = _make_iq()
        out = iq_imbalance(iq, _fresh_rng())
        assert out.shape == iq.shape
        assert out.dtype == np.float32


# ─── Augmentation: pipeline ──────────────────────────────────────────


class TestAugmentPipeline:
    def test_augment_preserves_shape(self):
        iq = _make_iq(512)
        out = augment(iq, _fresh_rng())
        assert out.shape == iq.shape

    def test_augment_all_enabled(self):
        cfg = AugmentationConfig(
            phase_offset_prob=1.0,
            freq_offset_prob=1.0,
            time_shift_prob=1.0,
            amplitude_scale_prob=1.0,
            awgn_prob=1.0,
            rayleigh_prob=1.0,
            rician_prob=1.0,
            iq_imbalance_prob=1.0,
        )
        iq = _make_iq(512)
        out = augment(iq, _fresh_rng(), cfg)
        assert out.shape == iq.shape
        assert np.isfinite(out).all()

    def test_augment_all_disabled(self):
        cfg = AugmentationConfig(
            phase_offset_prob=0.0,
            freq_offset_prob=0.0,
            time_shift_prob=0.0,
            amplitude_scale_prob=0.0,
            awgn_prob=0.0,
            rayleigh_prob=0.0,
            rician_prob=0.0,
            iq_imbalance_prob=0.0,
        )
        iq = _make_iq()
        out = augment(iq, _fresh_rng(), cfg)
        np.testing.assert_array_equal(out, iq)

    def test_augment_deterministic(self):
        iq = _make_iq()
        out1 = augment(iq.copy(), _fresh_rng(99))
        out2 = augment(iq.copy(), _fresh_rng(99))
        np.testing.assert_array_equal(out1, out2)

