"""Shared fixtures: a fake backbone so API/security tests need no trained model."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rfmc.api import create_app
from rfmc.config import Settings
from rfmc.model import ClassifierService
from rfmc.openset import OpenSetDetector

CLASSES = ["BPSK", "QPSK", "8PSK", "QAM16"]
KEY = "k" * 32
CONFIDENT = [8.0, 0.0, 0.0, 0.0]  # msp ~ 0.999
FLAT = [0.1, 0.1, 0.1, 0.1]  # msp = 0.25 -> below the 0.6 threshold -> unknown


class FakeBackbone:
    def __init__(self, logits=CONFIDENT, *, delay: float = 0.0):
        self.logits = logits
        self.delay = delay
        self.calls = 0
        self.last_batch = None

    def __call__(self, batch):
        import time

        self.calls += 1
        self.last_batch = batch
        if self.delay:
            time.sleep(self.delay)
        return np.array([self.logits], dtype=np.float64), np.zeros((1, 8))


def make_service(backbone=None):
    detector = OpenSetDetector("msp", threshold=0.6)
    return ClassifierService(backbone or FakeBackbone(), CLASSES, detector, model_version="test-1")


def make_settings(**overrides):
    base = dict(env="test", api_keys=frozenset({KEY}), rate_limit_per_min=600, rate_limit_burst=1000)
    base.update(overrides)
    return Settings(**base)


def good_payload(n: int = 128, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"iq": rng.standard_normal((2, n)).tolist()}


@pytest.fixture
def make_client():
    def _make(backbone=None, service=None, **settings_overrides):
        svc = service or make_service(backbone)
        return TestClient(create_app(svc, make_settings(**settings_overrides)))

    return _make


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def auth():
    return {"X-API-Key": KEY}
