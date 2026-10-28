"""Open-set scoring, calibration and AUROC."""

from __future__ import annotations

import numpy as np
import pytest

from rfmc.errors import ConfigError
from rfmc.openset import (
    MahalanobisScorer,
    OpenSetDetector,
    auroc,
    energy_score,
    msp_score,
    softmax,
)

PEAKED = np.array([[9.0, 0.0, 0.0, 0.0]])
FLAT = np.array([[0.2, 0.1, 0.2, 0.1]])


def test_softmax_sums_to_one_and_is_stable():
    p = softmax(np.array([[1e4, 0.0, -1e4], [0.0, 0.0, 0.0]]))
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_msp_and_energy_rank_peaked_above_flat():
    assert msp_score(PEAKED)[0] > msp_score(FLAT)[0]
    assert energy_score(PEAKED)[0] > energy_score(FLAT)[0]


def test_detector_requires_calibration():
    det = OpenSetDetector("msp")
    with pytest.raises(ConfigError):
        det.is_unknown(np.array([0.9]))


def test_nan_scores_fail_safe_to_unknown():
    det = OpenSetDetector("msp", threshold=0.5)
    assert det.is_unknown(np.array([np.nan, 0.9, 0.1])).tolist() == [True, False, True]


@pytest.mark.parametrize("bad", ["", "softmax", None])
def test_detector_rejects_bad_method(bad):
    with pytest.raises(ConfigError):
        OpenSetDetector(bad)


@pytest.mark.parametrize("t", [0, -1.0, float("nan"), float("inf")])
def test_detector_rejects_bad_temperature(t):
    with pytest.raises(ConfigError):
        OpenSetDetector("energy", temperature=t)


def test_calibration_keeps_target_fraction_of_known():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, (2000, 5))
    logits[np.arange(2000), rng.integers(0, 5, 2000)] += 4.0
    det = OpenSetDetector("msp")
    det.calibrate(logits, tpr=0.95)
    kept = 1.0 - det.is_unknown(det.confidence(logits)).mean()
    assert 0.94 <= kept <= 0.96


@pytest.mark.parametrize("tpr", [0.0, 1.0, -0.1, 1.5])
def test_calibration_rejects_bad_tpr(tpr):
    with pytest.raises(ValueError):
        OpenSetDetector("msp").calibrate(np.ones((10, 3)), tpr=tpr)


def test_calibration_rejects_nonfinite():
    bad = np.ones((10, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        OpenSetDetector("msp").calibrate(bad)


def _clusters(rng, n=300, d=6):
    centers = rng.normal(0, 6, (3, d))
    y = rng.integers(0, 3, n)
    return centers[y] + rng.normal(0, 1, (n, d)), y, centers


def test_mahalanobis_separates_unknown_cluster():
    rng = np.random.default_rng(3)
    x, y, centers = _clusters(rng)
    scorer = MahalanobisScorer().fit(x, y)
    known = scorer.score(centers[rng.integers(0, 3, 200)] + rng.normal(0, 1, (200, 6)))
    unknown = scorer.score(rng.normal(0, 1, (200, 6)) + 20.0)
    assert auroc(known, unknown) > 0.99


def test_mahalanobis_handles_degenerate_covariance():
    x = np.ones((20, 4))  # zero variance
    s = MahalanobisScorer().fit(x, np.zeros(20, dtype=int)).score(np.ones((3, 4)) * 2)
    assert np.isfinite(s).all()


def test_mahalanobis_input_validation():
    with pytest.raises(ValueError):
        MahalanobisScorer().fit(np.array([[np.nan, 1.0], [1.0, 2.0]]), [0, 1])
    with pytest.raises(ValueError):
        MahalanobisScorer().fit(np.ones((3, 2)), [0, 1])
    fitted = MahalanobisScorer().fit(np.random.default_rng(0).normal(size=(10, 3)), [0, 1] * 5)
    with pytest.raises(ValueError):
        fitted.score(np.ones((2, 5)))
    with pytest.raises(ConfigError):
        MahalanobisScorer().score(np.ones((1, 3)))


def test_mahalanobis_detector_needs_scorer_and_embeddings():
    with pytest.raises(ConfigError):
        OpenSetDetector("mahalanobis")
    scorer = MahalanobisScorer().fit(np.random.default_rng(0).normal(size=(10, 3)), [0, 1] * 5)
    with pytest.raises(ValueError):
        OpenSetDetector("mahalanobis", mahalanobis=scorer).confidence(np.ones((1, 4)))


def test_auroc_edge_cases():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    with pytest.raises(ValueError):
        auroc([], [1.0])


def test_auroc_matches_bruteforce():
    rng = np.random.default_rng(11)
    k = np.round(rng.normal(1, 1, 60), 1)  # rounding forces ties
    u = np.round(rng.normal(0, 1, 70), 1)
    wins = sum((a > b) + 0.5 * (a == b) for a in k for b in u)
    assert auroc(k, u) == pytest.approx(wins / (len(k) * len(u)))
