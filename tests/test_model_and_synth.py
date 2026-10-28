"""Signal generator + inference service unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from rfmc.errors import ConfigError, InferenceError, ValidationError
from rfmc.model import ClassifierService
from rfmc.openset import OpenSetDetector
from rfmc.synth import SUPPORTED, add_awgn, clean_symbols, generate_signal
from tests.conftest import CLASSES, CONFIDENT, FLAT, FakeBackbone, make_service

# ---------------- synth ----------------


@pytest.mark.parametrize("mod", SUPPORTED)
def test_clean_symbols_unit_power(mod):
    x = clean_symbols(mod, 20000, np.random.default_rng(0))
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=0.03)


def test_constellation_sizes():
    rng = np.random.default_rng(0)
    counts = {m: len(np.unique(np.round(clean_symbols(m, 5000, rng), 6))) for m in SUPPORTED}
    assert counts == {"BPSK": 2, "QPSK": 4, "8PSK": 8, "QAM16": 16}


@pytest.mark.parametrize("snr", [-10, 0, 10, 20])
def test_awgn_hits_requested_snr(snr):
    rng = np.random.default_rng(5)
    x = clean_symbols("QPSK", 200_000, rng)
    y = add_awgn(x, snr, rng)
    measured = 10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(y - x) ** 2))
    assert measured == pytest.approx(snr, abs=0.2)


def test_generate_is_seed_deterministic_and_shaped():
    a = generate_signal("8PSK", 256, 5, np.random.default_rng(9))
    b = generate_signal("8PSK", 256, 5, np.random.default_rng(9))
    assert a.shape == (2, 256) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [0, -1, 2.5, True, "8"])
def test_generate_rejects_bad_length(n):
    with pytest.raises(ValueError):
        generate_signal("BPSK", n, 0)


def test_generate_rejects_unknown_mod_and_bad_snr():
    with pytest.raises(ValueError):
        generate_signal("WBFM", 128, 0)
    with pytest.raises(ValueError):
        generate_signal("BPSK", 128, float("nan"))


# ---------------- service ----------------


def iq(n=256):
    return generate_signal("BPSK", n, 10, np.random.default_rng(0))


def test_confident_prediction_is_known():
    p = make_service().classify(iq())
    assert (p.label, p.unknown, p.closest_known) == ("BPSK", False, "BPSK")
    assert p.confidence > 0.99


def test_flat_logits_flagged_unknown_but_closest_reported():
    p = make_service(FakeBackbone(FLAT)).classify(iq())
    assert p.unknown and p.label == "unknown" and p.closest_known in CLASSES


def test_top_k_sorted_and_bounded():
    p = make_service(FakeBackbone([1.0, 3.0, 2.0, 0.0])).classify(iq())
    probs = [pr for _, pr in p.top_k]
    assert [l for l, _ in p.top_k] == ["QPSK", "8PSK", "BPSK"]
    assert probs == sorted(probs, reverse=True) and sum(probs) <= 1.0


def test_backbone_receives_float32_batch_and_input_untouched():
    bb = FakeBackbone()
    x = iq(300)
    before = x.copy()
    make_service(bb).classify(x)
    assert bb.last_batch.shape == (1, 2, 300) and bb.last_batch.dtype == np.float32
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("bad", [np.zeros(8), np.zeros((3, 8)), np.zeros((2, 4, 4))])
def test_service_rejects_bad_input_shape(bad):
    with pytest.raises(ValidationError):
        make_service().classify(bad)


def test_backbone_exception_becomes_inference_error_without_leak():
    def boom(_):
        raise RuntimeError("secret-internal-detail")

    with pytest.raises(InferenceError) as ei:
        make_service(boom).classify(iq())
    assert "secret" not in str(ei.value) and "secret" not in ei.value.to_public()[0]


@pytest.mark.parametrize(
    "logits",
    [[np.nan, 0, 0, 0], [np.inf, 0, 0, 0], [1.0, 2.0, 3.0]],
    ids=["nan", "inf", "wrong-width"],
)
def test_bad_backbone_output_rejected(logits):
    with pytest.raises(InferenceError):
        make_service(FakeBackbone(logits)).classify(iq())


def test_non_finite_embedding_rejected():
    bb = lambda b: (np.array([CONFIDENT]), np.full((1, 4), np.nan))  # noqa: E731
    with pytest.raises(InferenceError):
        make_service(bb).classify(iq())


@pytest.mark.parametrize(
    "names", [[], ["A", "A"], ["A", "unknown"]], ids=["empty", "duplicate", "reserved"]
)
def test_service_rejects_bad_class_names(names):
    with pytest.raises(ConfigError):
        ClassifierService(FakeBackbone(), names, OpenSetDetector("msp", threshold=0.5))


def test_service_refuses_uncalibrated_detector():
    with pytest.raises(ConfigError):
        ClassifierService(FakeBackbone(), CLASSES, OpenSetDetector("msp"))


def test_prediction_is_deterministic():
    svc = make_service()
    assert svc.classify(iq()) == svc.classify(iq())


def test_torch_model_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from rfmc.errors import SecurityError
    from rfmc.model_torch import RFNet, load_torch_backbone
    import hashlib

    net = RFNet(4)
    path = tmp_path / "m.pt"
    torch.save(net.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bb = load_torch_backbone(path, allowed_dir=tmp_path, expected_sha256=digest, n_classes=4)
    logits, emb = bb(np.zeros((1, 2, 128), dtype=np.float32))
    assert logits.shape == (1, 4) and emb.shape == (1, 256)
    with pytest.raises(SecurityError):
        load_torch_backbone(path, allowed_dir=tmp_path, expected_sha256="0" * 64, n_classes=4)
