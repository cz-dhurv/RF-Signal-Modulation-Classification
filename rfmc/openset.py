"""Open-set recognition: score "how known does this look", threshold, evaluate (Problem B)."""

from __future__ import annotations

import numpy as np

from .errors import ConfigError

METHODS = ("msp", "energy", "mahalanobis")


def softmax(logits, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _logsumexp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def msp_score(logits) -> np.ndarray:
    return softmax(logits).max(axis=-1)


def energy_score(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    return temperature * _logsumexp(z)


class MahalanobisScorer:
    """Class-mean distance with a shared, regularised covariance. Confidence = -min distance."""

    def fit(self, embeddings, labels) -> "MahalanobisScorer":
        x = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("embeddings must be (N, D) with N >= 2 and one label per row")
        if not np.isfinite(x).all():
            raise ValueError("embeddings contain NaN or inf")
        classes = np.unique(y)
        means = np.stack([x[y == c].mean(axis=0) for c in classes])
        centered = x - means[np.searchsorted(classes, y)]
        d = x.shape[1]
        cov = centered.T @ centered / len(x)
        cov = cov + (1e-3 * np.trace(cov) / d + 1e-9) * np.eye(d)
        self.means_ = means
        self.precision_ = np.linalg.pinv(cov)
        return self

    def score(self, embeddings) -> np.ndarray:
        if not hasattr(self, "means_"):
            raise ConfigError("MahalanobisScorer is not fitted")
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.means_.shape[1]:
            raise ValueError("embedding dimension mismatch")
        diff = x[:, None, :] - self.means_[None, :, :]
        dist = np.einsum("bcd,de,bce->bc", diff, self.precision_, diff)
        return -dist.min(axis=1)


class OpenSetDetector:
    """Higher confidence = more 'known'. Anything below the threshold is flagged unknown."""

    def __init__(
        self,
        method: str = "msp",
        *,
        temperature: float = 1.0,
        threshold: float | None = None,
        mahalanobis: MahalanobisScorer | None = None,
    ):
        if method not in METHODS:
            raise ConfigError(f"method must be one of {METHODS}")
        if not (np.isfinite(temperature) and temperature > 0):
            raise ConfigError("temperature must be positive and finite")
        if method == "mahalanobis" and mahalanobis is None:
            raise ConfigError("mahalanobis method needs a fitted MahalanobisScorer")
        self.method = method
        self.temperature = float(temperature)
        self.threshold = threshold
        self.mahalanobis = mahalanobis

    def confidence(self, logits, embeddings=None) -> np.ndarray:
        logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
        if self.method == "msp":
            return msp_score(logits)
        if self.method == "energy":
            return energy_score(logits, self.temperature)
        if embeddings is None:
            raise ValueError("mahalanobis scoring needs embeddings")
        return self.mahalanobis.score(np.atleast_2d(embeddings))

    def calibrate(self, logits, embeddings=None, tpr: float = 0.95) -> float:
        """Pick the threshold that keeps `tpr` of known-class samples as known."""
        if not 0.0 < tpr < 1.0:
            raise ValueError("tpr must be in (0, 1)")
        conf = self.confidence(logits, embeddings)
        if not np.isfinite(conf).all():
            raise ValueError("calibration scores must be finite")
        self.threshold = float(np.quantile(conf, 1.0 - tpr))
        return self.threshold

    def is_unknown(self, conf) -> np.ndarray:
        if self.threshold is None:
            raise ConfigError("OpenSetDetector is not calibrated")
        # `~(c >= t)` so NaN scores fail safe to "unknown".
        return ~(np.asarray(conf, dtype=np.float64) >= self.threshold)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(known_conf, unknown_conf) -> float:
    """P(known scores higher than unknown); ties count 0.5. Known = positive class."""
    k = np.asarray(known_conf, dtype=np.float64).ravel()
    u = np.asarray(unknown_conf, dtype=np.float64).ravel()
    if len(k) == 0 or len(u) == 0:
        raise ValueError("need at least one known and one unknown score")
    ranks = _rankdata(np.concatenate([k, u]))
    return float((ranks[: len(k)].sum() - len(k) * (len(k) + 1) / 2.0) / (len(k) * len(u)))
