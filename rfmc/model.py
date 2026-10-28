"""Inference service: wraps any backbone returning (logits, embedding) with output checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .errors import ConfigError, InferenceError, RFMCError, ValidationError
from .openset import OpenSetDetector, softmax

Backbone = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Prediction:
    label: str  # predicted class, or "unknown"
    unknown: bool
    closest_known: str
    confidence: float  # max softmax probability
    openset_score: float
    top_k: list = field(default_factory=list)  # [(label, probability), ...]


class ClassifierService:
    def __init__(
        self,
        backbone: Backbone,
        class_names: list[str],
        detector: OpenSetDetector,
        *,
        model_version: str = "dev",
        top_k: int = 3,
    ):
        if not class_names or len(set(class_names)) != len(class_names):
            raise ConfigError("class_names must be non-empty and unique")
        if UNKNOWN in class_names:
            raise ConfigError("'unknown' is reserved and cannot be a class name")
        if detector.threshold is None:
            raise ConfigError("detector must be calibrated before serving")
        self._backbone = backbone
        self.class_names = list(class_names)
        self._detector = detector
        self.model_version = model_version
        self._top_k = max(1, min(top_k, len(class_names)))

    @property
    def ready(self) -> bool:
        return True

    def classify(self, iq: np.ndarray) -> Prediction:
        iq = np.asarray(iq)
        if iq.ndim != 2 or iq.shape[0] != 2:
            raise ValidationError("iq must have shape (2, N)")
        batch = np.ascontiguousarray(iq[None, ...], dtype=np.float32)
        try:
            logits, emb = self._backbone(batch)
            logits = np.asarray(logits, dtype=np.float64)
            emb = np.asarray(emb, dtype=np.float64)
            if logits.shape != (1, len(self.class_names)):
                raise InferenceError("backbone returned logits of unexpected shape")
            if not (np.isfinite(logits).all() and np.isfinite(emb).all()):
                raise InferenceError("backbone returned non-finite values")
            probs = softmax(logits)[0]
            score = float(self._detector.confidence(logits, emb)[0])
            unknown = bool(self._detector.is_unknown(np.array([score]))[0])
        except RFMCError:
            raise
        except Exception as exc:  # never leak backend errors to callers
            raise InferenceError("backbone failed") from exc
        order = np.argsort(-probs)
        top = [(self.class_names[i], float(probs[i])) for i in order[: self._top_k]]
        closest = self.class_names[int(order[0])]
        return Prediction(
            label=UNKNOWN if unknown else closest,
            unknown=unknown,
            closest_known=closest,
            confidence=float(probs[order[0]]),
            openset_score=score,
            top_k=top,
        )
