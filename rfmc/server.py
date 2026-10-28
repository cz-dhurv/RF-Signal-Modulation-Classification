"""Production entrypoint:  uvicorn rfmc.server:app   (all config via environment)."""

from __future__ import annotations

import os

from .api import create_app
from .config import Settings
from .errors import ConfigError
from .model import ClassifierService
from .openset import OpenSetDetector


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required")
    return value


def build_service_from_env() -> ClassifierService:
    class_names = [c.strip() for c in _require("RFMC_CLASSES").split(",") if c.strip()]
    try:
        threshold = float(_require("RFMC_OPENSET_THRESHOLD"))
    except ValueError as exc:
        raise ConfigError("RFMC_OPENSET_THRESHOLD must be a number") from exc

    from .model_torch import load_torch_backbone  # lazy: needs torch

    backbone = load_torch_backbone(
        _require("RFMC_MODEL_PATH"),
        allowed_dir=_require("RFMC_MODEL_DIR"),
        expected_sha256=_require("RFMC_MODEL_SHA256"),
        n_classes=len(class_names),
    )
    detector = OpenSetDetector("msp", threshold=threshold)
    return ClassifierService(
        backbone, class_names, detector, model_version=os.environ.get("RFMC_MODEL_VERSION", "unversioned")
    )


def build_app_from_env():
    settings = Settings.from_env()  # validate config first: bad config fails before any model I/O
    return create_app(build_service_from_env(), settings)


def __getattr__(name: str):  # `app` is built lazily so importing this module has no side effects
    if name == "app":
        return build_app_from_env()
    raise AttributeError(name)
