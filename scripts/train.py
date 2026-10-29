#!/usr/bin/env python3
"""Phase 1 training pipeline: CNN baseline on RadioML / simulated data.

Usage:
    # Train on converted RadioML .npz
    python scripts/train.py --data data/rml2016.npz --epochs 60 --out models/

    # Train on simulator-generated data (no dataset file needed)
    python scripts/train.py --simulate --epochs 40 --out models/

    # Resume from checkpoint
    python scripts/train.py --data data/rml2016.npz --resume models/checkpoint_latest.pt --out models/

Outputs in --out directory:
    rfnet_best.pt                  — best model weights (by val accuracy)
    checkpoint_latest.pt           — latest checkpoint (model + optimizer + epoch)
    eval_results.npz               — evaluation arrays for the model acceptance gate
    accuracy_vs_snr.png            — accuracy-vs-SNR curve
    confusion_matrix.png           — normalised confusion matrix
    training_log.json              — per-epoch metrics
    model_card.json                — metadata: classes, hash, accuracy summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Add project root to path so we can import rfmc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Configuration ────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""
    # Data
    data_path: str | None = None
    simulate: bool = False
    sim_samples_per_class_per_snr: int = 200
    sim_snr_values: list[int] | None = None

    # Model
    arch: str = "rfnet"           # "rfnet", "cldnn", "transformer"
    n_classes: int | None = None  # auto-detected from data
    input_length: int = 1024      # zero-pad/trim all inputs to this

    # Training
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"   # "cosine" or "step"
    lr_step_epochs: list[int] | None = None
    lr_step_gamma: float = 0.1
    warmup_epochs: int = 3
    snr_weighted: bool = False    # apply SNR weighting in loss for low-SNR focus

    # Augmentation
    augment: bool = True
    aug_phase_prob: float = 0.5
    aug_freq_prob: float = 0.3
    aug_time_shift_prob: float = 0.3
    aug_awgn_prob: float = 0.3
    aug_rayleigh_prob: float = 0.2
    aug_iq_imbalance_prob: float = 0.2

    # Evaluation
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    top_k: int = 3

    # Output
    out_dir: str = "models"
    resume: str | None = None

    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_workers: int = 4
    seed: int = 42


# ─── Dataset ──────────────────────────────────────────────────────────


def load_npz_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load a dataset in the project .npz format."""
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)            # (M, 2, N)
    y = d["y"].astype(np.int32)              # (M,)
    snr = d["snr"].astype(np.int32)          # (M,)
    class_names = [str(c) for c in d["class_names"]]
    return X, y, snr, class_names


def generate_simulated_dataset(cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate training data from the advanced simulator."""
    from rfmc.simulator import KNOWN_MODS, SimulatorConfig, generate_dataset

    snr_values = cfg.sim_snr_values or list(range(-6, 22, 2))
    sim_cfg = SimulatorConfig(n_samples=cfg.input_length)
    X, y, snr, class_names = generate_dataset(
        mods=list(KNOWN_MODS),
        n_per_class_per_snr=cfg.sim_samples_per_class_per_snr,
        snr_values=snr_values,
        rng=np.random.default_rng(cfg.seed),
        cfg=sim_cfg,
    )
    return X, y, snr, class_names


def pad_or_trim(X: np.ndarray, target_len: int) -> np.ndarray:
    """Ensure all samples have shape (2, target_len)."""
    current_len = X.shape[2]
    if current_len == target_len:
        return X
    if current_len > target_len:
        # Centre-crop
        start = (current_len - target_len) // 2
        return X[:, :, start:start + target_len]
    # Zero-pad on the right
    pad_width = ((0, 0), (0, 0), (0, target_len - current_len))
    return np.pad(X, pad_width, mode="constant")


def split_by_snr_stratified(
    X: np.ndarray, y: np.ndarray, snr: np.ndarray,
    val_frac: float, test_frac: float, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split data stratified by (class, SNR) to avoid leakage.

    Within each (class, snr) group, samples are shuffled and split.
    This ensures every split has the same class-SNR distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = np.arange(n)

    # Group by (class, snr)
    groups = {}
    for i in range(n):
        key = (int(y[i]), int(snr[i]))
        groups.setdefault(key, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for _key, idxs in sorted(groups.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_test = max(1, int(len(idxs) * test_frac))
        n_val = max(1, int(len(idxs) * val_frac))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return {
        "train": (X[train_idx], y[train_idx], snr[train_idx]),
        "val": (X[val_idx], y[val_idx], snr[val_idx]),
        "test": (X[test_idx], y[test_idx], snr[test_idx]),
    }


# ─── PyTorch dataset and training ────────────────────────────────────


def _get_device(preference: str) -> str:
    """Resolve 'auto' to the best available device."""
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AugmentTransform:
    """Pickle-safe callable transform for PyTorch multiprocessing DataLoader."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, iq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        from rfmc.augment import augment
        return augment(iq, rng, self.cfg)


def _build_augmenter(cfg: TrainConfig):
    """Build the augmentation pipeline from config."""
    from rfmc.augment import AugmentationConfig

    aug_cfg = AugmentationConfig(
        phase_offset_prob=cfg.aug_phase_prob,
        freq_offset_prob=cfg.aug_freq_prob,
        time_shift_prob=cfg.aug_time_shift_prob,
        awgn_prob=cfg.aug_awgn_prob,
        rayleigh_prob=cfg.aug_rayleigh_prob,
        iq_imbalance_prob=cfg.aug_iq_imbalance_prob,
    )
    return AugmentTransform(aug_cfg)


class IQDataset:
    """Simple dataset wrapping numpy arrays with optional augmentation and sample weighting."""

    def __init__(
        self,
        X: np.ndarray, y: np.ndarray,
        weights: np.ndarray | None = None,
        augmenter=None,
        seed: int = 0,
    ):
        self.X = X
        self.y = y
        self.weights = weights
        self.augmenter = augmenter
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        import torch

        x = self.X[idx].copy()  # (2, N) float32
        if self.augmenter is not None:
            x = self.augmenter(x, self.rng)
        w = float(self.weights[idx]) if self.weights is not None else 1.0
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            int(self.y[idx]),
            torch.tensor(w, dtype=torch.float32),
        )


def evaluate_model(model, loader, criterion, device) -> dict:
    """Evaluate on a dataset, return metrics."""
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_embeddings = []
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in loader:
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            logits, emb = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
            all_logits.append(logits.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    return {
        "val_loss": total_loss / max(total, 1),
        "val_acc": correct / max(total, 1),
        "logits": np.concatenate(all_logits),
        "embeddings": np.concatenate(all_embeddings),
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
    }


# ─── Learning rate schedule ───────────────────────────────────────────


def build_lr_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Build LR scheduler with optional linear warmup."""
    import torch.optim.lr_scheduler as sched

    if cfg.lr_schedule == "cosine":
        main_scheduler = sched.CosineAnnealingLR(
            optimizer, T_max=(cfg.epochs - cfg.warmup_epochs) * steps_per_epoch
        )
    elif cfg.lr_schedule == "step":
        milestones = cfg.lr_step_epochs or [int(cfg.epochs * 0.6), int(cfg.epochs * 0.8)]
        milestone_steps = [m * steps_per_epoch for m in milestones]
        main_scheduler = sched.MultiStepLR(
            optimizer, milestones=milestone_steps, gamma=cfg.lr_step_gamma
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")

    if cfg.warmup_epochs > 0:
        warmup_steps = cfg.warmup_epochs * steps_per_epoch
        warmup_scheduler = sched.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        return sched.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    return main_scheduler


# ─── Evaluation plots ─────────────────────────────────────────────────


def plot_accuracy_vs_snr(
    labels: np.ndarray, preds: np.ndarray, snr: np.ndarray,
    class_names: list[str], out_path: Path
) -> dict[int, float]:
    """Plot overall and per-class accuracy vs SNR. Returns {snr: accuracy}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_values = sorted(set(snr.tolist()))
    overall_acc = {}
    per_class_acc = {name: {} for name in class_names}

    for s in snr_values:
        mask = snr == s
        if mask.sum() == 0:
            continue
        overall_acc[s] = float((preds[mask] == labels[mask]).mean())
        for ci, name in enumerate(class_names):
            class_mask = mask & (labels == ci)
            if class_mask.sum() > 0:
                per_class_acc[name][s] = float((preds[class_mask] == labels[class_mask]).mean())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(overall_acc.keys()), list(overall_acc.values()),
            "k-o", linewidth=2, markersize=6, label="Overall")

    for name in class_names:
        acc = per_class_acc[name]
        if acc:
            ax.plot(list(acc.keys()), list(acc.values()), "--", alpha=0.5, label=name)

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Classification Accuracy vs SNR", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved accuracy-vs-SNR plot to {out_path}")
    return overall_acc


def plot_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray,
    class_names: list[str], out_path: Path,
    title: str = "Confusion Matrix (normalised)"
) -> np.ndarray:
    """Plot and return the normalised confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    # Row-normalise (each row sums to 1)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.7), max(6, n_classes * 0.6)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            if val > 0.005:
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=max(6, 9 - n_classes // 8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix to {out_path}")
    return cm_norm


# ─── SHA-256 ──────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Main training loop ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train RFNet CNN for modulation classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default=None, help="Path to .npz dataset")
    parser.add_argument("--simulate", action="store_true", help="Generate data from simulator")
    parser.add_argument("--sim-samples", type=int, default=200,
                        help="Samples per class per SNR for simulated data")
    parser.add_argument("--arch", choices=["rfnet", "cldnn", "transformer"], default="rfnet",
                        help="Neural network architecture backbone")
    parser.add_argument("--snr-weighted", action="store_true",
                        help="Weight loss to emphasize low-to-mid SNR samples")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=["cosine", "step"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--out", type=str, default="models", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not args.data and not args.simulate:
        parser.error("Provide --data or --simulate")

    cfg = TrainConfig(
        data_path=args.data,
        simulate=args.simulate,
        sim_samples_per_class_per_snr=args.sim_samples,
        arch=args.arch,
        snr_weighted=args.snr_weighted,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        augment=not args.no_augment,
        out_dir=args.out,
        resume=args.resume,
        device=args.device,
        seed=args.seed,
        input_length=args.input_length,
        num_workers=args.num_workers,
    )

    # ── Imports that need torch ──
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rfmc.model_torch import build_model

    # ── Reproducibility ──
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = _get_device(cfg.device)
    print(f"Device: {device}")

    # ── Load data ──
    print("\n=== Loading data ===")
    if cfg.simulate:
        print("Generating simulated dataset ...")
        X, y, snr, class_names = generate_simulated_dataset(cfg)
    else:
        print(f"Loading {cfg.data_path} ...")
        X, y, snr, class_names = load_npz_dataset(cfg.data_path)

    # Pad/trim to uniform length
    X = pad_or_trim(X, cfg.input_length)
    n_classes = len(class_names)
    cfg.n_classes = n_classes

    print(f"  Samples: {len(X)}, Classes: {n_classes} ({', '.join(class_names)})")
    print(f"  Shape: {X.shape}, SNR range: [{snr.min()}, {snr.max()}] dB")

    # ── Split ──
    print("\n=== Splitting data ===")
    splits = split_by_snr_stratified(X, y, snr, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    for name, (sx, sy, ss) in splits.items():
        print(f"  {name:5s}: {len(sx)} samples")

    # ── Datasets and loaders ──
    augmenter = _build_augmenter(cfg) if cfg.augment else None
    if cfg.snr_weighted:
        # Give higher weight to challenging low/mid SNR samples (-6 dB to +4 dB)
        train_snr = splits["train"][2]
        train_weights = np.clip(1.5 - (train_snr / 20.0), 0.5, 2.0).astype(np.float32)
        train_weights = train_weights / np.mean(train_weights)
    else:
        train_weights = None

    train_ds = IQDataset(
        splits["train"][0], splits["train"][1],
        weights=train_weights, augmenter=augmenter, seed=cfg.seed
    )
    val_ds = IQDataset(splits["val"][0], splits["val"][1])
    test_ds = IQDataset(splits["test"][0], splits["test"][1])

    workers = 0 if sys.platform == "win32" else min(cfg.num_workers, 4)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=workers, pin_memory=(device != "cpu"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=workers,
    )

    # ── Model ──
    print(f"\n=== Building {cfg.arch.upper()} (n_classes={n_classes}) ===")
    model = build_model(cfg.arch, n_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    steps_per_epoch = len(train_loader)
    scheduler = build_lr_scheduler(optimizer, cfg, steps_per_epoch)

    # ── Resume ──
    start_epoch = 0
    best_val_acc = 0.0
    if cfg.resume:
        print(f"\n  Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    # ── Output dir ──
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    print(f"\n=== Training for {cfg.epochs - start_epoch} epochs ===")
    log = []

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            batch_w = batch[2].to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            if cfg.snr_weighted:
                losses = criterion_none(logits, batch_y)
                loss = (losses * batch_w).mean()
            else:
                loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_total += len(batch_y)

        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        entry = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_correct / max(train_total, 1),
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_acc"],
            "lr": lr_now,
            "time_s": dt,
        }
        log.append(entry)

        marker = ""
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            torch.save(model.state_dict(), out_dir / "rfnet_best.pt")
            marker = " * best"

        print(
            f"  [{epoch:3d}/{cfg.epochs}]"
            f"  loss={entry['train_loss']:.4f}"
            f"  acc={entry['train_acc']:.4f}"
            f"  val_loss={entry['val_loss']:.4f}"
            f"  val_acc={entry['val_acc']:.4f}"
            f"  lr={lr_now:.2e}"
            f"  {dt:.1f}s{marker}"
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "config": asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else {},
            }, out_dir / "checkpoint_latest.pt")

    # ── Save training log ──
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── Final evaluation on test set ──
    print("\n=== Test-set evaluation ===")
    model.load_state_dict(torch.load(out_dir / "rfnet_best.pt", map_location="cpu", weights_only=True))
    model.to(device)
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    print(f"  Test accuracy: {test_metrics['val_acc']:.4f}")
    print(f"  Test loss:     {test_metrics['val_loss']:.4f}")

    test_snr = splits["test"][2]
    test_preds = test_metrics["preds"]
    test_labels = test_metrics["labels"]

    # Accuracy vs SNR
    snr_acc = plot_accuracy_vs_snr(
        test_labels, test_preds, test_snr, class_names, out_dir / "accuracy_vs_snr.png"
    )

    # Confusion matrix
    cm = plot_confusion_matrix(
        test_labels, test_preds, class_names, out_dir / "confusion_matrix.png"
    )

    # Print accuracy by SNR band
    print("\n  Accuracy by SNR band:")
    for s in sorted(snr_acc.keys()):
        print(f"    {s:+3d} dB: {snr_acc[s]:.4f}")

    # ── Save evaluation results for model acceptance gate ──
    eval_path = out_dir / "eval_results.npz"
    np.savez_compressed(
        eval_path,
        X=splits["test"][0],
        y=test_labels,
        snr=test_snr,
        preds=test_preds,
        logits=test_metrics["logits"],
        embeddings=test_metrics["embeddings"],
        class_names=np.array(class_names),
        confusion_matrix=cm,
    )
    print(f"  Saved evaluation results to {eval_path}")

    # ── Model card ──
    model_hash = sha256_file(out_dir / "rfnet_best.pt")
    high_snr_acc = np.mean([v for k, v in snr_acc.items() if k >= 10]) if snr_acc else 0.0
    mid_snr_acc = np.mean([v for k, v in snr_acc.items() if -2 <= k <= 2]) if snr_acc else 0.0
    low_snr_acc = np.mean([v for k, v in snr_acc.items() if k <= -4]) if snr_acc else 0.0

    card = {
        "model": cfg.arch.upper(),
        "arch": cfg.arch,
        "parameters": param_count,
        "class_names": class_names,
        "n_classes": n_classes,
        "input_length": cfg.input_length,
        "sha256": model_hash,
        "training": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "augmentation": cfg.augment,
            "data_source": cfg.data_path or "simulator",
            "train_samples": len(splits["train"][0]),
            "val_samples": len(splits["val"][0]),
            "test_samples": len(splits["test"][0]),
        },
        "results": {
            "test_accuracy": float(test_metrics["val_acc"]),
            "best_val_accuracy": float(best_val_acc),
            "accuracy_high_snr_gte_10dB": float(high_snr_acc),
            "accuracy_mid_snr_0dB": float(mid_snr_acc),
            "accuracy_low_snr_lte_neg4dB": float(low_snr_acc),
            "accuracy_by_snr": {str(k): v for k, v in sorted(snr_acc.items())},
        },
    }
    with open(out_dir / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"\n  Model SHA-256: {model_hash}")
    print(f"  Model card saved to {out_dir / 'model_card.json'}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best val accuracy:  {best_val_acc:.4f}")
    print(f"  Test accuracy:      {test_metrics['val_acc']:.4f}")
    print(f"  High SNR (>=10 dB): {high_snr_acc:.4f}")
    print(f"  Mid SNR (~0 dB):    {mid_snr_acc:.4f}")
    print(f"  Low SNR (<=-4 dB):  {low_snr_acc:.4f}")
    print(f"\n  To serve this model:")
    print(f"    export RFMC_CLASSES='{','.join(class_names)}'")
    print(f"    export RFMC_MODEL_DIR='{out_dir.resolve()}'")
    print(f"    export RFMC_MODEL_PATH='{(out_dir / 'rfnet_best.pt').resolve()}'")
    print(f"    export RFMC_MODEL_SHA256='{model_hash}'")
    print(f"    uvicorn rfmc.server:app")
    print("=" * 60)


if __name__ == "__main__":
    main()
