"""PyTorch backbone + safe loader. Import lazily: requires torch.

Safety: path confined to an allowed dir, suffix allow-list, SHA-256 pinned, and
torch.load(weights_only=True) so a tampered file cannot execute code via pickle.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .errors import ModelUnavailableError
from .security import ensure_safe_model_path, verify_sha256


class RFNet(nn.Module):
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

    def forward(self, x):  # x: (B, 2, N)
        z = self.feat(x)
        return self.head(z), z


def load_torch_backbone(model_path, *, allowed_dir, expected_sha256: str, n_classes: int, device="cpu"):
    path = ensure_safe_model_path(model_path, allowed_dir, allowed_suffixes={".pt"})
    verify_sha256(path, expected_sha256)
    net = RFNet(n_classes)
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
