#!/usr/bin/env python3
"""Convert RadioML datasets to safe .npz format.

**Run this script OUTSIDE the rfmc package.** The production security gate
(AST scan) forbids pickle inside rfmc/. This script is intentionally in
scripts/ which is not scanned.

Usage:
    # RadioML 2016.10A (.pkl)
    python scripts/convert_radioml.py rml2016 /path/to/RML2016.10a_dict.pkl -o data/rml2016.npz

    # RadioML 2018.01A (.hdf5)
    python scripts/convert_radioml.py rml2018 /path/to/GOLD_XYZ_OSC.0001_1024.hdf5 -o data/rml2018.npz

Output .npz contains:
    X          (M, 2, N) float32   — I/Q samples, power-normalised
    y          (M,) int32          — class index (0..K-1)
    snr        (M,) int32          — SNR in dB
    class_names (K,) str           — ordered class names
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle  # noqa: S403 — intentional, isolated from production code
import sys
from pathlib import Path

import numpy as np

# Known file hashes for integrity verification (add your own if needed)
KNOWN_HASHES: dict[str, str] = {
    # Add SHA-256 hashes here after verifying your downloads:
    # "RML2016.10a_dict.pkl": "abc123...",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise(x: np.ndarray) -> np.ndarray:
    """Power-normalise a (2, N) array to unit mean power."""
    x = x.astype(np.float64)
    peak = float(np.max(np.abs(x)))
    if not math.isfinite(peak) or peak == 0.0:
        return x  # skip zero-power signals (they'll be filtered later)
    x = x / peak
    power = float(np.mean(x[0] ** 2 + x[1] ** 2))
    if power > 0:
        x = x / math.sqrt(power)
    return x.astype(np.float32)


def convert_rml2016(src: Path, dst: Path) -> None:
    """Convert RadioML 2016.10A pickle to .npz."""
    print(f"Loading {src} ...")
    known = KNOWN_HASHES.get(src.name)
    if known:
        actual = _sha256(src)
        if actual != known:
            print(f"ERROR: hash mismatch. Expected {known}, got {actual}", file=sys.stderr)
            sys.exit(1)

    with open(src, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")  # noqa: S301

    # data is dict[(mod_name, snr_db)] -> np.ndarray (N_per_class, 2, 128)
    all_x, all_y, all_snr = [], [], []
    class_set: set[str] = set()

    for (mod, snr), samples in sorted(data.items()):
        class_set.add(mod)

    class_names = sorted(class_set)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for (mod, snr), samples in sorted(data.items()):
        samples = np.asarray(samples, dtype=np.float32)
        for i in range(len(samples)):
            x = _normalise(samples[i])
            all_x.append(x)
            all_y.append(class_to_idx[mod])
            all_snr.append(int(snr))

    X = np.stack(all_x)  # (M, 2, 128)
    y = np.array(all_y, dtype=np.int32)
    snr = np.array(all_snr, dtype=np.int32)

    print(f"  Samples: {len(X)}, Classes: {len(class_names)}, SNR range: [{snr.min()}, {snr.max()}] dB")
    print(f"  Classes: {class_names}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, X=X, y=y, snr=snr, class_names=np.array(class_names))
    print(f"  Saved to {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def convert_rml2018(src: Path, dst: Path) -> None:
    """Convert RadioML 2018.01A HDF5 to .npz."""
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py is required for RML2018. Install with: pip install h5py", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {src} ... (this may take a while, the file is ~4 GB)")
    with h5py.File(src, "r") as f:
        X_raw = f["X"][:]       # (M, 1024, 2) — note: samples × 2, transposed from RML2016
        Y_raw = f["Y"][:]       # (M, 24) one-hot
        Z_raw = f["Z"][:]       # (M, 1) SNR in dB

    # RML2018 class names in canonical order (matches one-hot columns)
    class_names = [
        "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
        "16APSK", "32APSK", "64APSK", "128APSK",
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
        "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
        "FM", "GMSK", "OQPSK",
    ]

    y = np.argmax(Y_raw, axis=1).astype(np.int32)
    snr = Z_raw.ravel().astype(np.int32)

    # Transpose from (M, 1024, 2) to (M, 2, 1024)
    X = np.transpose(X_raw, (0, 2, 1)).astype(np.float32)

    # Power-normalise each sample
    print("  Power-normalising ...")
    for i in range(len(X)):
        X[i] = _normalise(X[i])

    print(f"  Samples: {len(X)}, Classes: {len(class_names)}, SNR range: [{snr.min()}, {snr.max()}] dB")
    print(f"  Classes: {class_names}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, X=X, y=y, snr=snr, class_names=np.array(class_names))
    print(f"  Saved to {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RadioML datasets to .npz")
    parser.add_argument("format", choices=["rml2016", "rml2018"], help="Dataset format")
    parser.add_argument("input", type=Path, help="Input file (.pkl or .hdf5)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .npz path")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: {args.input} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.format == "rml2016":
        convert_rml2016(args.input, args.output)
    else:
        convert_rml2018(args.input, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
