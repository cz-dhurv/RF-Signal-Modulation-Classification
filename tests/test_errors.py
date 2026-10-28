"""Error handling: uniform envelope, no internal leakage, correct status codes, logging."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from rfmc import errors
from rfmc.errors import RFMCError, error_body_for
from tests.conftest import KEY, FakeBackbone, good_payload, make_service


def _all_error_classes():
    return [c for c in vars(errors).values() if isinstance(c, type) and issubclass(c, RFMCError) and c is not RFMCError]


def test_error_codes_are_unique_and_statuses_sane():
    classes = _all_error_classes()
    codes = [c.code for c in classes]
    assert len(codes) == len(set(codes))
    assert all(400 <= c.http_status <= 599 for c in classes)


# Explicit policy, not derived from the class flags (otherwise the test could never fail).
MUST_HIDE = {"InternalError", "InferenceError", "UnauthorizedError", "ConfigError", "SecurityError",
             "ModelUnavailableError", "InferenceTimeoutError"}
MAY_SHOW = {"BadRequestError", "PayloadTooLargeError", "UnsupportedMediaTypeError",
            "ValidationError", "RateLimitedError"}


def test_error_policy_covers_every_class():
    assert {c.__name__ for c in _all_error_classes()} == MUST_HIDE | MAY_SHOW  # new class => decide its policy


@pytest.mark.parametrize("name", sorted(MUST_HIDE))
def test_server_side_errors_never_expose_constructor_message_or_details(name):
    exc = getattr(errors, name)("secret-internal-detail", details={"path": "/etc/passwd"})
    body = error_body_for(exc, "rid")["error"]
    assert "secret" not in str(body) and "passwd" not in str(body) and "details" not in body


@pytest.mark.parametrize("name", sorted(MAY_SHOW))
def test_client_errors_show_their_safe_message(name):
    body = error_body_for(getattr(errors, name)("safe text"), "rid")["error"]
    assert body["message"] == "safe text"


def test_envelope_shape():
    body = error_body_for(errors.ValidationError("bad", details={"got_shape": [3, 4]}), "r1")
    assert body == {"error": {"code": "validation_error", "message": "bad", "request_id": "r1",
                              "details": {"got_shape": [3, 4]}}}


@pytest.mark.parametrize(
    "method,path,status,code",
    [("get", "/nope", 404, "not_found"), ("get", "/v1/classify", 405, "method_not_allowed"),
     ("post", "/healthz", 405, "method_not_allowed")],
)
def test_routing_errors_use_uniform_envelope(client, method, path, status, code):
    r = getattr(client, method)(path)
    assert r.status_code == status
    err = r.json()["error"]
    assert err["code"] == code and err["request_id"] == r.headers["x-request-id"]
    assert "detail" not in r.json()  # no FastAPI default shape


@pytest.mark.parametrize(
    "body,status",
    [(b"{not json", 400), (b"", 400), (b"\xff\xfe\x00", 400), (b"[1,2,3]", 422), (b"null", 422),
     (b'{"iq": "abc"}', 422), (b'{"iq": [[NaN,1],[1,2]]}', 422), (b'{"sample_rate_hz": 1}', 422)],
)
def test_client_errors_are_4xx_with_envelope(client, auth, body, status):
    r = client.post("/v1/classify", content=body, headers={**auth, "Content-Type": "application/json"})
    assert r.status_code == status
    assert set(r.json()["error"]) >= {"code", "message", "request_id"}


def test_validation_error_reports_shape_not_data(client, auth):
    r = client.post("/v1/classify", json={"iq": [[1.0] * 128] * 3}, headers=auth)
    assert r.status_code == 422
    assert r.json()["error"]["details"] == {"got_shape": [3, 128]}


def test_unhandled_exception_returns_generic_500_and_logs_with_request_id(make_client, auth, caplog):
    class Exploding:
        ready = True
        model_version = "x"

        def classify(self, iq):
            raise RuntimeError("secret-internal-detail /srv/models/key.pt")

    c = make_client(service=Exploding())
    with caplog.at_level(logging.ERROR):
        r = c.post("/v1/classify", json=good_payload(), headers=auth)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert "secret" not in r.text and "Traceback" not in r.text and "/srv" not in r.text
    rid = r.headers["x-request-id"]
    records = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert records and any(rid in rec.getMessage() for rec in records)
    assert any(rec.exc_info for rec in records)  # traceback is in logs, not in the response


def test_backbone_failure_maps_to_inference_error(make_client, auth):
    def boom(_):
        raise ValueError("secret-internal-detail")

    r = make_client(backbone=boom).post("/v1/classify", json=good_payload(), headers=auth)
    assert r.status_code == 500 and r.json()["error"]["code"] == "inference_error"
    assert "secret" not in r.text


@pytest.mark.parametrize("logits", [[float("nan"), 0, 0, 0], [1.0, 2.0]])
def test_garbage_model_output_is_500_not_a_wrong_answer(make_client, auth, logits):
    r = make_client(backbone=FakeBackbone(logits)).post("/v1/classify", json=good_payload(), headers=auth)
    assert r.status_code == 500 and r.json()["error"]["code"] == "inference_error"


def test_inference_timeout_returns_504(make_client, auth):
    c = make_client(backbone=FakeBackbone(delay=0.4), inference_timeout_s=0.05)
    r = c.post("/v1/classify", json=good_payload(), headers=auth)
    assert r.status_code == 504 and r.json()["error"]["code"] == "inference_timeout"


def test_readyz_reports_503_when_model_not_ready(make_client):
    class NotReady:
        ready = False
        model_version = "x"

    r = make_client(service=NotReady()).get("/readyz")
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_health_and_ready_ok(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["model_version"] == "test-1"


def test_server_keeps_working_after_errors(client, auth):
    for _ in range(3):
        client.post("/v1/classify", content=b"{bad", headers={**auth, "Content-Type": "application/json"})
    assert client.post("/v1/classify", json=good_payload(), headers=auth).status_code == 200


def test_successful_classification_contract(client, auth):
    r = client.post("/v1/classify", json={**good_payload(256), "sample_rate_hz": 2e6}, headers=auth)
    assert r.status_code == 200
    b = r.json()
    assert b["modulation"] == "BPSK" and b["unknown"] is False and b["num_samples"] == 256
    assert b["request_id"] == r.headers["x-request-id"] and b["model_version"] == "test-1"
    assert [t["label"] for t in b["top_k"]][0] == "BPSK"


def test_unknown_signal_is_flagged_over_http(make_client, auth):
    c = make_client(backbone=FakeBackbone([0.1, 0.1, 0.1, 0.1]))
    b = c.post("/v1/classify", json=good_payload(), headers=auth).json()
    assert b["unknown"] is True and b["modulation"] == "unknown" and b["closest_known"] in {"BPSK", "QPSK", "8PSK", "QAM16"}


def test_server_module_fails_fast_on_missing_env(monkeypatch):
    from rfmc.server import build_app_from_env

    for var in ["RFMC_CLASSES", "RFMC_OPENSET_THRESHOLD", "RFMC_MODEL_PATH", "RFMC_MODEL_DIR", "RFMC_MODEL_SHA256"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RFMC_ENV", "test")
    with pytest.raises(errors.ConfigError):
        build_app_from_env()
    monkeypatch.setenv("RFMC_CLASSES", "A,B")
    monkeypatch.setenv("RFMC_OPENSET_THRESHOLD", "not-a-number")
    with pytest.raises(errors.ConfigError):
        build_app_from_env()
