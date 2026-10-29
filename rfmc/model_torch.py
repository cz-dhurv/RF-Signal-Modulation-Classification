"""PyTorch backbones (RFNet, CLDNN, Transformer) + safe loader. Import lazily: requires torch.

Safety: path confined to an allowed dir, suffix allow-list, SHA-256 pinned, and
torch.load(weights_only=True) so a tampered file cannot execute code via pickle.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import nn

from .errors import ConfigError, ModelUnavailableError
from .security import ensure_safe_model_path, verify_sha256


class RFNet(nn.Module):
    """4-block 1D CNN baseline for modulation classification."""

    def __init__(self, n_classes: int):
        super().__init__()

        def blk(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)
            )

        self.feat = nn.Sequential(
            blk(2, 64), blk(64, 128), blk(128, 128), blk(128, 256),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # x: (B, 2, N)
        z = self.feat(x)
        return self.head(z), z


class RFCLDNN(nn.Module):
    """Convolutional, Long Short-Term Memory, Deep Neural Network (CLDNN).

    Combines 1D CNN for local feature extraction with Bidirectional LSTM
    for temporal / phase-tracking dynamics, particularly effective in low-SNR regimes.
    """

    def __init__(self, n_classes: int, lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Input to LSTM: (B, seq_len, channels=128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        self.dense = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # x: (B, 2, N)
        c = self.conv(x)  # (B, 128, N/4)
        c = c.transpose(1, 2)  # (B, N/4, 128)
        lstm_out, _ = self.lstm(c)  # (B, N/4, 256)
        z = self.dense(lstm_out[:, -1, :])  # (B, 256) embedding from final step
        return self.head(z), z


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer sequence inputs."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class RFTransformer(nn.Module):
    """Transformer-based modulation classifier with multi-head self-attention.

    Uses 1D convolution for patch/token embedding, sinusoidal positional encoding,
    and TransformerEncoder layers for global sequence modeling.
    """

    def __init__(
        self,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Initial projection to d_model tokens via strided conv
        self.patch_embed = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # x: (B, 2, N)
        # Patch embed downsamples sequence length by 4x
        emb = self.patch_embed(x).transpose(1, 2)  # (B, N/4, d_model)
        emb = self.pos_encoder(emb)
        out = self.transformer(emb)  # (B, N/4, d_model)
        # Global mean pooling
        pooled = out.mean(dim=1)  # (B, d_model)
        z = self.proj(pooled)     # (B, 256) embedding
        return self.head(z), z


MODEL_ARCHS = {
    "rfnet": RFNet,
    "cldnn": RFCLDNN,
    "transformer": RFTransformer,
}


def build_model(
    arch: str,
    n_classes: int,
) -> nn.Module:
    """Instantiate a model architecture by name."""
    arch_key = arch.lower().strip()
    if arch_key not in MODEL_ARCHS:
        raise ConfigError(f"Unsupported architecture {arch!r}. Supported: {list(MODEL_ARCHS.keys())}")
    return MODEL_ARCHS[arch_key](n_classes=n_classes)


def load_torch_backbone(
    model_path,
    *,
    allowed_dir,
    expected_sha256: str,
    n_classes: int,
    arch: str = "rfnet",
    device: str = "cpu",
):
    """Safely load a PyTorch backbone from an approved directory with SHA-256 verification."""
    path = ensure_safe_model_path(model_path, allowed_dir, allowed_suffixes={".pt"})
    verify_sha256(path, expected_sha256)
    net = build_model(arch, n_classes)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)  # strict by default
    except Exception as exc:
        raise ModelUnavailableError("Model failed to load.") from exc
    net.to(device).eval()

    def backbone(batch: np.ndarray):
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(device)
            logits, z = net(x)
        return logits.cpu().numpy(), z.cpu().numpy()

    return backbone
