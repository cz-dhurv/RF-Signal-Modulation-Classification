#!/usr/bin/env python3
"""Evaluate a trained RFNet model and produce reports for the release gate.

Usage:
    # Evaluate on a test .npz
    python scripts/evaluate.py --model models/rfnet_best.pt --data data/rml2016.npz --out eval_out/

    # Evaluate on simulator-generated data
    python scripts/evaluate.py --model models/rfnet_best.pt --simulate --out eval_out/

    # Produce the acceptance .npz for RFMC_EVAL_NPZ (model gate)
    python scripts/evaluate.py --model models/rfnet_best.pt --data data/rml2016.npz --out eval_out/ --acceptance

Outputs:
    eval_results.npz        — arrays for the model acceptance gate
    accuracy_vs_snr.png     — accuracy-vs-SNR curve
    confusion_matrix.png    — normalised confusion matrix
    eval_report.json        — structured evaluation summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RFNet and produce release-gate reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to rfnet_best.pt")
    parser.add_argument("--data", type=str, default=None, help="Path to test .npz")
    parser.add_argument("--simulate", action="store_true", help="Generate test data from simulator")
    parser.add_argument("--arch", choices=["auto", "rfnet", "cldnn", "transformer"], default="auto",
                        help="Model architecture backbone (auto reads from model card or defaults to rfnet)")
    parser.add_argument("--model-card", type=str, default=None,
                        help="Path to model_card.json (auto-detects class names and architecture)")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names (overrides model card)")
    parser.add_argument("--out", type=str, default="eval_out")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--acceptance", action="store_true",
                        help="Also produce acceptance .npz for the model gate")
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data and not args.simulate:
        parser.error("Provide --data or --simulate")

    import torch

    # ── Resolve class names and architecture ──
    class_names = None
    arch = "rfnet" if args.arch == "auto" else args.arch
    if args.model_card:
        with open(args.model_card) as f:
            card = json.load(f)
        class_names = card.get("class_names")
        if args.arch == "auto":
            arch = card.get("arch", "rfnet")
    if args.classes:
        class_names = [c.strip() for c in args.classes.split(",")]
    # else: will be inferred from data

    # ── Load data ──
    print("=== Loading data ===")
    if args.simulate:
        from rfmc.simulator import KNOWN_MODS, SimulatorConfig, generate_dataset
        sim_cfg = SimulatorConfig(n_samples=args.input_length)
        X, y, snr, data_class_names = generate_dataset(
            mods=list(KNOWN_MODS),
            n_per_class_per_snr=100,
            rng=np.random.default_rng(args.seed + 1000),  # different seed from training
            cfg=sim_cfg,
        )
    else:
        d = np.load(args.data, allow_pickle=False)
        X = d["X"].astype(np.float32)
        y = d["y"].astype(np.int32)
        snr = d["snr"].astype(np.int32)
        data_class_names = [str(c) for c in d["class_names"]]

    if class_names is None:
        class_names = data_class_names

    # Pad/trim
    n = X.shape[2]
    if n != args.input_length:
        if n > args.input_length:
            start = (n - args.input_length) // 2
            X = X[:, :, start:start + args.input_length]
        else:
            X = np.pad(X, ((0, 0), (0, 0), (0, args.input_length - n)))

    n_classes = len(class_names)
    print(f"  Samples: {len(X)}, Classes: {n_classes}")
    print(f"  Shape: {X.shape}, SNR range: [{snr.min()}, {snr.max()}] dB")

    # ── Load model ──
    print(f"\n=== Loading model ({arch}) ===")
    from rfmc.model_torch import build_model

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = build_model(arch, n_classes)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Loaded from {args.model}")
    print(f"  Device: {device}")

    model_hash = sha256_file(Path(args.model))
    print(f"  SHA-256: {model_hash}")

    # ── Evaluate ──
    print("\n=== Running evaluation ===")
    all_logits, all_embeddings, all_preds, all_labels = [], [], [], []

    dataset_size = len(X)
    with torch.inference_mode():
        for start in range(0, dataset_size, args.batch_size):
            end = min(start + args.batch_size, dataset_size)
            batch_x = torch.from_numpy(
                np.ascontiguousarray(X[start:end], dtype=np.float32)
            ).to(device)
            logits, emb = model(batch_x)
            preds = logits.argmax(dim=1)
            all_logits.append(logits.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y[start:end])
            if (end - start) > 0 and end % 10000 < args.batch_size:
                print(f"  {end}/{dataset_size} ...", end="\r")

    all_logits = np.concatenate(all_logits)
    all_embeddings = np.concatenate(all_embeddings)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    overall_acc = float((all_preds == all_labels).mean())
    print(f"\n  Overall accuracy: {overall_acc:.4f}")

    # ── Output ──
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import plot helpers from train.py
    from train import plot_accuracy_vs_snr, plot_confusion_matrix

    snr_acc = plot_accuracy_vs_snr(
        all_labels, all_preds, snr, class_names, out_dir / "accuracy_vs_snr.png"
    )

    cm = plot_confusion_matrix(
        all_labels, all_preds, class_names, out_dir / "confusion_matrix.png"
    )

    # Per-class accuracy
    per_class = {}
    for ci, name in enumerate(class_names):
        mask = all_labels == ci
        if mask.sum() > 0:
            per_class[name] = float((all_preds[mask] == all_labels[mask]).mean())

    # SNR band summaries
    high_snr_acc = np.mean([v for k, v in snr_acc.items() if k >= 10]) if snr_acc else 0.0
    mid_snr_acc = np.mean([v for k, v in snr_acc.items() if -2 <= k <= 2]) if snr_acc else 0.0
    low_snr_acc = np.mean([v for k, v in snr_acc.items() if k <= -4]) if snr_acc else 0.0

    report = {
        "model_path": str(Path(args.model).resolve()),
        "model_sha256": model_hash,
        "data_source": args.data or "simulator",
        "n_samples": len(X),
        "n_classes": n_classes,
        "class_names": class_names,
        "overall_accuracy": overall_acc,
        "accuracy_high_snr_gte_10dB": float(high_snr_acc),
        "accuracy_mid_snr_0dB": float(mid_snr_acc),
        "accuracy_low_snr_lte_neg4dB": float(low_snr_acc),
        "accuracy_by_snr": {str(k): v for k, v in sorted(snr_acc.items())},
        "accuracy_by_class": per_class,
    }

    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report to {out_dir / 'eval_report.json'}")

    # ── Save evaluation arrays ──
    np.savez_compressed(
        out_dir / "eval_results.npz",
        X=X, y=all_labels, snr=snr,
        preds=all_preds, logits=all_logits, embeddings=all_embeddings,
        class_names=np.array(class_names),
        confusion_matrix=cm,
    )
    print(f"  Saved eval_results.npz to {out_dir / 'eval_results.npz'}")

    # ── Acceptance file for model gate ──
    if args.acceptance:
        # Format matching PRD: X (M,2,N) float32, y (M,) int, snr (M,) dB
        acc_path = out_dir / "acceptance.npz"
        np.savez_compressed(acc_path, X=X, y=all_labels, snr=snr)
        print(f"  Saved acceptance data to {acc_path}")
        print(f"  Use with: RFMC_EVAL_NPZ={acc_path.resolve()} pytest -m production")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"  Overall accuracy:   {overall_acc:.4f}")
    print(f"  High SNR (>=10 dB): {high_snr_acc:.4f}")
    print(f"  Mid SNR (~0 dB):    {mid_snr_acc:.4f}")
    print(f"  Low SNR (<=-4 dB):  {low_snr_acc:.4f}")

    # PRD target comparison
    print("\n  vs. PRD targets:")
    targets = [
        ("SNR >= +10 dB accuracy", high_snr_acc, 0.95),
        ("SNR ~ 0 dB accuracy", mid_snr_acc, 0.80),
    ]
    for name, actual, target in targets:
        status = "[PASS]" if actual >= target else "[FAIL]"
        print(f"    {status}  {name}: {actual:.4f} (target >= {target:.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()

