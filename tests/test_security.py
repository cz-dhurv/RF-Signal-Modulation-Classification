"""Security tests: auth, rate limiting, size caps, headers, log hygiene, model supply chain."""

from __future__ import annotations

import hashlib
import logging
import os
import re

import pytest

from rfmc import security
from rfmc.config import Settings
from rfmc.errors import ConfigError, RateLimitedError, SecurityError
from rfmc.security import (
    TokenBucketRateLimiter,
    ensure_safe_model_path,
    key_fingerprint,
    safe_request_id,
    sanitize_for_log,
    verify_api_key,
    verify_sha256,
)
from tests.conftest import KEY, good_payload

# ---------------- api key ----------------


def test_verify_api_key_matrix():
    allowed = {KEY, "another-valid-key-0123456789"}
    assert verify_api_key(KEY, allowed)
    assert verify_api_key("another-valid-key-0123456789", allowed)
    for bad in [None, "", "wrong", KEY + "x", KEY[:-1], 123, b"bytes", "k" * 10_000]:
        assert not verify_api_key(bad, allowed), bad
    assert not verify_api_key(KEY, set())  # fail closed with no keys configured


def test_verify_api_key_compares_every_key(monkeypatch):
    calls = []
    real = security.hmac.compare_digest
    monkeypatch.setattr(security.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    assert verify_api_key("a" * 30, {"a" * 30, "b" * 30, "c" * 30})
    assert len(calls) == 3  # no early exit -> no timing oracle on key position


def test_key_fingerprint_does_not_reveal_key():
    fp = key_fingerprint(KEY)
    assert len(fp) == 12 and KEY not in fp and fp == key_fingerprint(KEY)
    assert fp != key_fingerprint(KEY + "x")


# ---------------- rate limiter ----------------


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_rate_limiter_burst_then_block_then_refill():
    clock = Clock()
    rl = TokenBucketRateLimiter(rate_per_min=60, burst=3, clock=clock)  # 1 token/s
    for _ in range(3):
        rl.check("a")
    with pytest.raises(RateLimitedError) as ei:
        rl.check("a")
    assert ei.value.retry_after >= 1
    clock.t += 1.0
    rl.check("a")  # refilled one token
    with pytest.raises(RateLimitedError):
        rl.check("a")


def test_rate_limiter_isolated_per_key_and_capped_at_burst():
    clock = Clock()
    rl = TokenBucketRateLimiter(60, 2, clock=clock)
    rl.check("a"); rl.check("a")
    rl.check("b")  # other key unaffected
    clock.t += 10_000  # long idle must not bank more than `burst`
    rl.check("a"); rl.check("a")
    with pytest.raises(RateLimitedError):
        rl.check("a")


def test_rate_limiter_memory_is_bounded():
    rl = TokenBucketRateLimiter(60, 1, max_keys=5)
    for i in range(500):
        rl.check(f"k{i}")
    assert len(rl) <= 5


@pytest.mark.parametrize("kw", [dict(rate_per_min=0, burst=1), dict(rate_per_min=1, burst=0)])
def test_rate_limiter_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(**kw)


# ---------------- sanitising ----------------


def test_sanitize_for_log_blocks_log_injection_and_truncates():
    s = sanitize_for_log("ok\r\nINFO forged line\x1b[31m" + "x" * 1000)
    assert "\n" not in s and "\r" not in s and "\x1b" not in s and len(s) <= 200


@pytest.mark.parametrize("evil", ["abc\r\nX-Evil: 1", "short", "a" * 65, "id with space!!", None, ""])
def test_request_id_rejects_unsafe_values(evil):
    rid = safe_request_id(evil)
    assert re.fullmatch(r"[0-9a-f]{32}", rid)


def test_request_id_accepts_plain_token():
    assert safe_request_id("req-1234_ABCD") == "req-1234_ABCD"


# ---------------- model supply chain ----------------


def test_model_path_confinement(tmp_path):
    good = tmp_path / "m.pt"
    good.write_bytes(b"x")
    assert ensure_safe_model_path(good, tmp_path) == good.resolve()

    outside = tmp_path.parent / "outside.pt"
    outside.write_bytes(b"x")
    with pytest.raises(SecurityError):
        ensure_safe_model_path(outside, tmp_path)
    with pytest.raises(SecurityError):
        ensure_safe_model_path(tmp_path / ".." / "outside.pt", tmp_path)

    link = tmp_path / "link.pt"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("symlinks not supported or permitted on this platform")
    with pytest.raises(SecurityError):  # symlink escape
        ensure_safe_model_path(link, tmp_path)


def test_model_path_suffix_existence_and_type(tmp_path):
    (tmp_path / "m.pkl").write_bytes(b"x")
    (tmp_path / "dir.pt").mkdir()
    for p in ["m.pkl", "dir.pt", "missing.pt"]:
        with pytest.raises(SecurityError):
            ensure_safe_model_path(tmp_path / p, tmp_path)


def test_sha256_verification(tmp_path):
    f = tmp_path / "m.pt"
    f.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    verify_sha256(f, digest)
    verify_sha256(f, digest.upper())
    for bad in ["0" * 64, "short", "", None, "z" * 64]:
        with pytest.raises(SecurityError):
            verify_sha256(f, bad)


# ---------------- settings hardening ----------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(env="production"),  # no keys
        dict(env="production", api_keys=frozenset({"short"})),
        dict(env="production", api_keys=frozenset({"has space " + "x" * 30})),
        dict(env="staging"),
        dict(max_body_bytes=10),
        dict(rate_limit_per_min=0),
        dict(inference_timeout_s=0),
        dict(min_len=5000, max_len=4096),
    ],
)
def test_insecure_or_invalid_settings_refuse_to_start(kw):
    with pytest.raises(ConfigError):
        Settings(**kw)


def test_settings_from_env_parsing():
    s = Settings.from_env({"RFMC_ENV": "production", "RFMC_API_KEYS": f" {KEY} , ,", "RFMC_MAX_BODY_BYTES": "4096"})
    assert s.api_keys == frozenset({KEY}) and s.max_body_bytes == 4096
    with pytest.raises(ConfigError):
        Settings.from_env({"RFMC_MAX_BODY_BYTES": "lots"})


# ---------------- HTTP layer ----------------


def test_missing_and_wrong_key_are_indistinguishable(client):
    a = client.post("/v1/classify", json=good_payload())
    b = client.post("/v1/classify", json=good_payload(), headers={"X-API-Key": "wrong-key"})
    assert a.status_code == b.status_code == 401
    strip = lambda r: {k: v for k, v in r.json()["error"].items() if k != "request_id"}  # noqa: E731
    assert strip(a) == strip(b)
    assert a.headers["www-authenticate"] == "ApiKey"


def test_fails_closed_when_no_keys_configured(make_client):
    c = make_client(api_keys=frozenset())
    assert c.post("/v1/classify", json=good_payload(), headers={"X-API-Key": KEY}).status_code == 401


def test_auth_checked_before_body_is_parsed(client):
    r = client.post("/v1/classify", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 401


def test_security_headers_on_success_and_error(client, auth):
    for r in [client.get("/healthz"), client.get("/nope"), client.post("/v1/classify", json={})]:
        h = r.headers
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert h["cache-control"] == "no-store"
        assert h["referrer-policy"] == "no-referrer"
        assert "default-src 'none'" in h["content-security-policy"]
        assert "strict-transport-security" not in h  # only set in production


def test_production_hides_docs_and_sets_hsts(make_client):
    c = make_client(env="production")
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404
    assert "max-age" in c.get("/healthz").headers["strict-transport-security"]


def test_docs_available_outside_production(client):
    assert client.get("/openapi.json").status_code == 200


def test_oversize_body_by_content_length(make_client, auth):
    c = make_client(max_body_bytes=1024)
    r = c.post("/v1/classify", content=b"x" * 2000, headers={**auth, "Content-Type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


def test_oversize_chunked_body_is_cut_off(make_client, auth):
    c = make_client(max_body_bytes=1024)

    def gen():
        for _ in range(50):
            yield b"x" * 100

    r = c.post("/v1/classify", content=gen(), headers={**auth, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_wrong_content_type_is_415(client, auth):
    r = client.post("/v1/classify", content=b"{}", headers={**auth, "Content-Type": "text/plain"})
    assert r.status_code == 415


def test_json_nesting_bomb_is_400_not_500(client, auth):
    bomb = b"[" * 100_000 + b"]" * 100_000
    r = client.post("/v1/classify", content=bomb, headers={**auth, "Content-Type": "application/json"})
    assert r.status_code == 400


def test_rate_limit_over_http(make_client, auth):
    c = make_client(rate_limit_per_min=1, rate_limit_burst=2)
    codes = [c.post("/v1/classify", json=good_payload(), headers=auth).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[2:] == [429, 429]
    r = c.post("/v1/classify", json=good_payload(), headers=auth)
    assert int(r.headers["retry-after"]) >= 1


def test_ip_rate_limit_applies_before_auth(make_client):
    c = make_client(rate_limit_burst=1, ip_rate_limit_per_min=1)  # IP burst floor is 20
    codes = [c.post("/v1/classify", json={}).status_code for _ in range(25)]
    assert 429 in codes and codes[0] == 401


def test_secrets_and_payloads_never_logged(make_client, caplog):
    c = make_client()
    with caplog.at_level(logging.DEBUG):
        c.post("/v1/classify", json=good_payload(), headers={"X-API-Key": KEY})
        c.post("/v1/classify", json=good_payload(), headers={"X-API-Key": "wrong-key-xyz"})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert KEY not in text and "wrong-key-xyz" not in text
    assert "iq" not in text.lower().split()


def test_client_supplied_request_id_is_validated(client):
    ok = client.get("/healthz", headers={"X-Request-ID": "trace-abcdef12"})
    assert ok.headers["x-request-id"] == "trace-abcdef12"
    bad = client.get("/healthz", headers={"X-Request-ID": "bad id!!"})
    assert re.fullmatch(r"[0-9a-f]{32}", bad.headers["x-request-id"])
