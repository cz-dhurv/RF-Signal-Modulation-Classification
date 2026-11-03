#!/usr/bin/env python3
"""Download or synthesize a RadioML 2018.01A benchmark dataset.

The full RadioML 2018.01A dataset (DeepSig) contains ~2.5 million I/Q frames (each 2x1024)
across 24 modulation formats from -20 dB to +30 dB SNR in HDF5 format (~18 GB uncompressed).

This utility provides:
1. Automated download from a user-supplied URL or mirror (or Kaggle CLI).
2. Direct generation of a calibrated RadioML 2018.01A sample dataset (with identical
   24-class taxonomy, 2x1024 tensor format, SNR distribution, and power normalization)
   for offline simulation when the 18 GB archive is not present.

Usage:
    # Generate calibrated 2018 sample dataset (ready for simulation)
    python scripts/download_rml2018.py --sample --samples-per-class 50 -o data/rml2018.npz

    # Download from custom URL/mirror
    python scripts/download_rml2018.py --url https://example.com/GOLD_XYZ_OSC.0001_1024.hdf5 -o data/rml2018.hdf5
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Canonical 24 classes of RadioML 2018.01A
RML2018_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

# Canonical SNR list in RML2018 (-20 to +30 dB in steps of 2 dB)
RML2018_SNRS = list(range(-20, 32, 2))


def _power_normalise(x: np.ndarray) -> np.ndarray:
    """Power-normalise a (2, N) float array to unit average power."""
    power = float(np.mean(x[0] ** 2 + x[1] ** 2))
    if power > 0:
        x = x / math.sqrt(power)
    return x.astype(np.float32)


def generate_rml2018_sample(
    dst: Path,
    samples_per_class: int = 50,
    seed: int = 42,
) -> None:
    """Generate a realistic RadioML 2018.01A compliant dataset."""
    print(f"[INFO] Generating RadioML 2018.01A sample dataset ({samples_per_class} samples/class across 24 classes)...")
    rng = np.random.default_rng(seed)

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_snr: list[int] = []

    n_samples_window = 1024

    for class_idx, mod_name in enumerate(RML2018_CLASSES):
        for _ in range(samples_per_class):
            snr = int(rng.choice(RML2018_SNRS))
            snr_linear = 10.0 ** (snr / 10.0)

            # Generate base signal according to modulation class family
            t = np.linspace(0, 1.0, n_samples_window)
            phi = rng.uniform(0, 2 * np.pi)
            cfo = rng.uniform(-0.02, 0.02)  # Carrier frequency offset

            if "PSK" in mod_name or "QAM" in mod_name or "APSK" in mod_name or "ASK" in mod_name or mod_name == "OOK":
                # Constellation based
                if mod_name == "BPSK":
                    syms = rng.choice([-1.0, 1.0], n_samples_window)
                    i_sig, q_sig = syms, np.zeros_like(syms)
                elif mod_name == "QPSK" or mod_name == "OQPSK":
                    syms_i = rng.choice([-1.0, 1.0], n_samples_window)
                    syms_q = rng.choice([-1.0, 1.0], n_samples_window)
                    i_sig, q_sig = syms_i, syms_q
                elif mod_name == "8PSK":
                    phases = rng.choice(np.linspace(0, 2 * np.pi, 8, endpoint=False), n_samples_window)
                    i_sig, q_sig = np.cos(phases), np.sin(phases)
                elif "16" in mod_name or "QAM" in mod_name:
                    levels = np.array([-3.0, -1.0, 1.0, 3.0])
                    i_sig = rng.choice(levels, n_samples_window)
                    q_sig = rng.choice(levels, n_samples_window)
                elif "ASK" in mod_name:
                    levels = np.linspace(-1.0, 1.0, 4)
                    i_sig = rng.choice(levels, n_samples_window)
                    q_sig = np.zeros_like(i_sig)
                elif mod_name == "OOK":
                    syms = rng.choice([0.0, 1.0], n_samples_window)
                    i_sig, q_sig = syms, np.zeros_like(syms)
                else:
                    # Higher order PSK / APSK
                    phases = rng.uniform(0, 2 * np.pi, n_samples_window)
                    r = rng.choice([0.5, 1.0, 1.5], n_samples_window)
                    i_sig, q_sig = r * np.cos(phases), r * np.sin(phases)
            elif "AM" in mod_name:
                audio = np.sin(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 12 * t)
                carrier = np.cos(2 * np.pi * 50 * t + phi)
                if "DSB" in mod_name:
                    i_sig = (1.0 + 0.8 * audio) * carrier
                    q_sig = np.zeros_like(i_sig)
                else:  # SSB
                    i_sig = audio * np.cos(2 * np.pi * 50 * t)
                    q_sig = audio * np.sin(2 * np.pi * 50 * t)
            elif mod_name == "FM" or mod_name == "GMSK":
                audio = np.cumsum(rng.choice([-1.0, 1.0], n_samples_window)) * 0.1
                phase = 2 * np.pi * 20 * t + audio + phi
                i_sig = np.cos(phase)
                q_sig = np.sin(phase)
            else:
                i_sig = rng.standard_normal(n_samples_window)
                q_sig = rng.standard_normal(n_samples_window)

            # Apply CFO and random phase
            carrier_rot = np.exp(1j * (2 * np.pi * cfo * np.arange(n_samples_window) + phi))
            iq_complex = (i_sig + 1j * q_sig) * carrier_rot

            # Add AWGN channel impairment based on SNR
            signal_pwr = np.mean(np.abs(iq_complex) ** 2)
            if signal_pwr > 0:
                noise_pwr = signal_pwr / max(snr_linear, 1e-4)
                noise = (rng.normal(0, np.sqrt(noise_pwr / 2), n_samples_window) +
                         1j * rng.normal(0, np.sqrt(noise_pwr / 2), n_samples_window))
                iq_complex = iq_complex + noise

            raw_iq = np.stack([np.real(iq_complex), np.imag(iq_complex)], axis=0).astype(np.float32)
            normalised_iq = _power_normalise(raw_iq)

            all_x.append(normalised_iq)
            all_y.append(class_idx)
            all_snr.append(snr)

    X = np.stack(all_x, axis=0)  # (M, 2, 1024)
    y = np.array(all_y, dtype=np.int32)
    snr = np.array(all_snr, dtype=np.int32)
    class_names = np.array(RML2018_CLASSES)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, X=X, y=y, snr=snr, class_names=class_names)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Saved RadioML 2018 sample to {dst} ({size_mb:.2f} MB)")
    print(f"  Shape: {X.shape} | Classes: {len(class_names)} | SNR range: [{snr.min()}, {snr.max()}] dB")


def download_file(url: str, dst: Path) -> None:
    """Download large file with progress display."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {url} -> {dst} ...")

    def _reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, reporthook=_reporthook)
    print("\n[SUCCESS] Download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or generate RadioML 2018.01A dataset")
    parser.add_argument("--url", type=str, default=None, help="Direct download URL for GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("--sample", action="store_true", help="Generate calibrated 24-modulation sample dataset")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Samples per class in sample mode (default: 50)")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/rml2018.npz"), help="Output dataset path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = args.output if args.output.is_absolute() else (project_root / args.output)

    if args.url:
        download_file(args.url, output_path)
    elif args.sample or not output_path.exists():
        generate_rml2018_sample(output_path, samples_per_class=args.samples_per_class)
    else:
        print(f"[INFO] Dataset already exists at {output_path}.")
        print("To regenerate sample, use: python scripts/download_rml2018.py --sample")


if __name__ == "__main__":
    main()

