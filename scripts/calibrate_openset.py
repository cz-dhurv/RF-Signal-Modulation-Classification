#!/usr/bin/env python3
"""Phase 3 open-set calibration and evaluation: Problem B.

Calibrates open-set thresholds (MSP, Energy, Mahalanobis) on known validation signals
at a target true-positive rate (TPR), then measures AUROC and false-reject rate
against held-out unknown signals across SNR bands.

Usage:
    # Calibrate using simulator-generated known and held-out unknown waveforms
    python scripts/calibrate_openset.py --model models/rfnet_best.pt --model-card models/model_card.json --simulate --out models/openset

    # Calibrate using custom datasets
    python scripts/calibrate_openset.py --model models/rfnet_best.pt --known data/val_known.npz --unknown data/test_unknown.npz --out models/openset

Outputs:
    openset_calibration.json     — Calibrated thresholds, AUROC, and release metrics
    auroc_vs_snr.png             — Open-set AUROC vs SNR curve
    score_distributions.png      — Histograms of known vs unknown scores
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfmc.openset import (
    METHODS,
    MahalanobisScorer,
    OpenSetDetector,
    auroc,
    energy_score,
    msp_score,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pad_or_trim(X: np.ndarray, target_len: int) -> np.ndarray:
    n = X.shape[2]
    if n == target_len:
        return X
    if n > target_len:
        start = (n - target_len) // 2
        return X[:, :, start : start + target_len]
    return np.pad(X, ((0, 0), (0, 0), (0, target_len - n)))


def extract_features(
    model, X: np.ndarray, batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference to get logits and embeddings."""
    import torch

    all_logits = []
    all_embeddings = []
    n = len(X)
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch = torch.from_numpy(np.ascontiguousarray(X[i:end], dtype=np.float32)).to(device)
            logits, emb = model(batch)
            all_logits.append(logits.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_embeddings)


def plot_auroc_vs_snr(
    snr_aurocs: dict[str, dict[int, float]], out_path: Path
) -> None:
    """Plot AUROC vs SNR for all open-set methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"msp": "o", "energy": "s", "mahalanobis": "^"}

    for method, curve in snr_aurocs.items():
        if not curve:
            continue
        snrs = sorted(curve.keys())
        values = [curve[s] for s in snrs]
        ax.plot(
            snrs,
            values,
            f"-{markers.get(method, 'd')}",
            label=f"{method.upper()}",
            linewidth=2,
            markersize=6,
        )

    ax.axhline(0.90, color="red", linestyle="--", alpha=0.7, label="PRD Target (≥ 0.90)")
    ax.axhline(0.50, color="gray", linestyle=":", alpha=0.5, label="Random (0.50)")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Open-Set AUROC", fontsize=12)
    ax.set_title("Open-Set AUROC vs SNR (Known vs Unknown)", fontsize=14)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved AUROC vs SNR plot to {out_path}")


def plot_score_distributions(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
    method_name: str,
    out_path: Path,
) -> None:
    """Plot score histograms of known vs unknown signals with the calibrated threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(known_scores.min(), unknown_scores.min()),
        max(known_scores.max(), unknown_scores.max()),
        50,
    )

    ax.hist(known_scores, bins=bins, alpha=0.6, density=True, label="Known Signals", color="dodgerblue")
    ax.hist(unknown_scores, bins=bins, alpha=0.6, density=True, label="Unknown Signals", color="crimson")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel(f"{method_name.upper()} Score (higher = more known)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Open-Set Score Distributions ({method_name.upper()})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved score distribution plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate open-set recognition (Problem B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--model-card", type=str, default=None, help="Path to model_card.json")
    parser.add_argument("--arch", choices=["auto", "rfnet", "cldnn", "transformer"], default="auto")
    parser.add_argument("--simulate", action="store_true", help="Generate known and unknown data via simulator")
    parser.add_argument("--sim-samples", type=int, default=100, help="Simulated samples per class per SNR")
    parser.add_argument("--known", type=str, default=None, help="Path to known validation .npz")
    parser.add_argument("--unknown", type=str, default=None, help="Path to unknown evaluation .npz")
    parser.add_argument("--tpr", type=float, default=0.95, help="Target TPR on known validation data")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for energy scoring")
    parser.add_argument("--out", type=str, default="models/openset", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    import torch
    from rfmc.model_torch import build_model

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve classes and architecture ──
    class_names = None
    arch = "rfnet" if args.arch == "auto" else args.arch
    if args.model_card and Path(args.model_card).exists():
        with open(args.model_card) as f:
            card = json.load(f)
        class_names = card.get("class_names")
        if args.arch == "auto":
            arch = card.get("arch", "rfnet")

    # ── Load or generate known and unknown datasets ──
    print("\n=== Preparing Data ===")
    if args.simulate:
        from rfmc.simulator import KNOWN_MODS, UNKNOWN_MODS, SimulatorConfig, generate_dataset

        sim_cfg = SimulatorConfig(n_samples=args.input_length)
        known_mods = list(class_names) if class_names else list(KNOWN_MODS)
        unknown_mods = list(UNKNOWN_MODS)

        print(f"Generating known validation signals ({len(known_mods)} classes: {', '.join(known_mods)}) ...")
        X_known, y_known, snr_known, _ = generate_dataset(
            mods=known_mods,
            n_per_class_per_snr=args.sim_samples,
            rng=np.random.default_rng(args.seed),
            cfg=sim_cfg,
        )

        print(f"Generating unknown signals ({len(unknown_mods)} classes: {', '.join(unknown_mods)}) ...")
        X_unknown, y_unknown, snr_unknown, _ = generate_dataset(
            mods=unknown_mods,
            n_per_class_per_snr=args.sim_samples,
            rng=np.random.default_rng(args.seed + 5000),
            cfg=sim_cfg,
        )
        if class_names is None:
            class_names = known_mods
    else:
        if not args.known or not args.unknown:
            parser.error("Either provide --simulate or provide both --known and --unknown paths")
        d_k = np.load(args.known, allow_pickle=False)
        X_known = d_k["X"].astype(np.float32)
        y_known = d_k["y"].astype(np.int32)
        snr_known = d_k["snr"].astype(np.int32)
        if class_names is None and "class_names" in d_k:
            class_names = [str(c) for c in d_k["class_names"]]

        d_u = np.load(args.unknown, allow_pickle=False)
        X_unknown = d_u["X"].astype(np.float32)
        y_unknown = d_u["y"].astype(np.int32)
        snr_unknown = d_u["snr"].astype(np.int32)

    X_known = pad_or_trim(X_known, args.input_length)
    X_unknown = pad_or_trim(X_unknown, args.input_length)
    n_classes = len(class_names)

    print(f"  Known:   {len(X_known)} samples, {n_classes} classes, SNR [{snr_known.min()}, {snr_known.max()}] dB")
    print(f"  Unknown: {len(X_unknown)} samples, SNR [{snr_unknown.min()}, {snr_unknown.max()}] dB")

    # ── Load model ──
    print(f"\n=== Loading Model ({arch.upper()}, n_classes={n_classes}) ===")
    model = build_model(arch, n_classes)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    model_sha256 = sha256_file(Path(args.model))
    print(f"  Loaded weights from {args.model}")
    print(f"  Model SHA-256: {model_sha256}")

    # ── Extract logits and embeddings ──
    print("\n=== Extracting Features ===")
    k_logits, k_emb = extract_features(model, X_known, args.batch_size, device)
    u_logits, u_emb = extract_features(model, X_unknown, args.batch_size, device)

    # ── Calibrate and evaluate each method ──
    print("\n=== Open-Set Calibration & Evaluation ===")
    results = {}
    snr_aurocs = {}
    unique_snrs = sorted(set(snr_known.tolist()) & set(snr_unknown.tolist()))

    for method in METHODS:
        if method == "mahalanobis":
            scorer = MahalanobisScorer().fit(k_emb, y_known)
            detector = OpenSetDetector(method=method, temperature=args.temperature, mahalanobis=scorer)
        else:
            detector = OpenSetDetector(method=method, temperature=args.temperature)

        # Calibrate threshold on KNOWN validation data only at target TPR
        threshold = detector.calibrate(k_logits, k_emb if method == "mahalanobis" else None, tpr=args.tpr)

        k_scores = detector.confidence(k_logits, k_emb if method == "mahalanobis" else None)
        u_scores = detector.confidence(u_logits, u_emb if method == "mahalanobis" else None)

        overall_auroc = auroc(k_scores, u_scores)

        # Sliced metrics per SNR
        per_snr_auroc = {}
        for s in unique_snrs:
            k_mask = snr_known == s
            u_mask = snr_unknown == s
            if k_mask.sum() > 0 and u_mask.sum() > 0:
                per_snr_auroc[int(s)] = float(auroc(k_scores[k_mask], u_scores[u_mask]))

        # High-SNR metrics (>= +10 dB) as required by PRD
        high_snr_k_mask = snr_known >= 10
        high_snr_u_mask = snr_unknown >= 10
        if high_snr_k_mask.sum() > 0 and high_snr_u_mask.sum() > 0:
            high_snr_auroc = float(auroc(k_scores[high_snr_k_mask], u_scores[high_snr_u_mask]))
            # False reject rate (FRR): fraction of known signals marked as unknown
            high_snr_is_unknown = detector.is_unknown(k_scores[high_snr_k_mask])
            high_snr_frr = float(high_snr_is_unknown.mean())
        else:
            high_snr_auroc = overall_auroc
            high_snr_frr = 1.0 - args.tpr

        snr_aurocs[method] = per_snr_auroc
        results[method] = {
            "method": method,
            "calibrated_threshold": float(threshold),
            "target_tpr": float(args.tpr),
            "overall_auroc": float(overall_auroc),
            "high_snr_gte_10dB_auroc": float(high_snr_auroc),
            "high_snr_gte_10dB_false_reject_rate": float(high_snr_frr),
            "auroc_by_snr": {str(k): v for k, v in sorted(per_snr_auroc.items())},
        }

        prd_pass_auroc = "PASS" if high_snr_auroc >= 0.90 else "FAIL"
        prd_pass_frr = "PASS" if high_snr_frr <= 0.10 else "FAIL"
        print(
            f"  [{method.upper():12s}] Threshold: {threshold:8.4f} | "
            f"Overall AUROC: {overall_auroc:.4f} | "
            f"High SNR AUROC: {high_snr_auroc:.4f} ({prd_pass_auroc}) | "
            f"FRR: {high_snr_frr*100:.1f}% ({prd_pass_frr})"
        )

    # ── Pick best method based on high-SNR AUROC ──
    best_method = max(results.keys(), key=lambda m: results[m]["high_snr_gte_10dB_auroc"])
    best_result = results[best_method]
    print(f"\n  * Best Method: {best_method.upper()} with threshold {best_result['calibrated_threshold']:.4f}")

    # ── Plot curves ──
    plot_auroc_vs_snr(snr_aurocs, out_dir / "auroc_vs_snr.png")

    if best_method == "mahalanobis":
        scorer = MahalanobisScorer().fit(k_emb, y_known)
        best_detector = OpenSetDetector(method=best_method, temperature=args.temperature, mahalanobis=scorer)
    else:
        best_detector = OpenSetDetector(method=best_method, temperature=args.temperature)
    best_thresh = best_result["calibrated_threshold"]
    best_k_scores = best_detector.confidence(k_logits, k_emb if best_method == "mahalanobis" else None)
    best_u_scores = best_detector.confidence(u_logits, u_emb if best_method == "mahalanobis" else None)
    plot_score_distributions(
        best_k_scores,
        best_u_scores,
        best_thresh,
        best_method,
        out_dir / "score_distributions.png",
    )

    # ── Save calibration report ──
    report = {
        "model_path": str(Path(args.model).resolve()),
        "model_sha256": model_sha256,
        "arch": arch,
        "n_classes": n_classes,
        "class_names": class_names,
        "target_tpr": args.tpr,
        "best_method": best_method,
        "recommended_threshold": float(best_thresh),
        "methods": results,
    }
    with open(out_dir / "openset_calibration.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report to {out_dir / 'openset_calibration.json'}")

    # ── Print serving commands ──
    print("\n" + "=" * 60)
    print("Open-Set Calibration Summary:")
    print(f"  Recommended Method:    {best_method.upper()}")
    print(f"  Recommended Threshold: {best_thresh:.4f}")
    print(f"  High SNR AUROC:        {best_result['high_snr_gte_10dB_auroc']:.4f} (target >= 0.90)")
    print(f"  High SNR False Reject: {best_result['high_snr_gte_10dB_false_reject_rate']*100:.1f}% (target <= 10%)")
    print("\nTo deploy this calibrated service:")
    print(f"  export RFMC_CLASSES='{','.join(class_names)}'")
    print(f"  export RFMC_OPENSET_THRESHOLD='{best_thresh:.4f}'")
    print(f"  export RFMC_MODEL_DIR='{Path(args.model).parent.resolve()}'")
    print(f"  export RFMC_MODEL_PATH='{Path(args.model).resolve()}'")
    print(f"  export RFMC_MODEL_SHA256='{model_sha256}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
