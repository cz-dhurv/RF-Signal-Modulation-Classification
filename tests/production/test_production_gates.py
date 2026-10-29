"""Production release gates. Run with:  pytest -m production

These are the checks that must be green before a deploy. Most need no trained model;
the model-acceptance gate at the bottom is skipped unless evaluation data is supplied.
"""

from __future__ import annotations

import ast
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rfmc import errors
from rfmc.api import create_app
from tests.conftest import KEY, good_payload, make_service, make_settings

pytestmark = pytest.mark.production

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "rfmc"
DOCS = ROOT / "docs"


# ---------------------------------------------------------------- supply chain


def test_all_dependencies_are_pinned():
    for req in ROOT.glob("requirements*.txt"):
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r")):
                continue
            assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==\d[\w.\-+]*", line), f"{req.name}: unpinned '{line}'"


def test_no_secrets_committed():
    patterns = [
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"(?i)(?:api[_-]?key|secret|passw(?:or)?d|token)\s*[:=]\s*[\"'][^\"']{12,}[\"']"),
    ]
    assert not list(ROOT.rglob(".env*")), "env files must not be committed"
    for path in list(PKG.rglob("*.py")) + list(DOCS.rglob("*.md")) + list(ROOT.glob("*.md")) + list(ROOT.glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            assert not pat.search(text), f"possible secret in {path.relative_to(ROOT)}"


# ---------------------------------------------------------------- static security scan

FORBIDDEN_IMPORTS = {"pickle", "cPickle", "marshal", "dill", "shelve", "subprocess"}
FORBIDDEN_CALLS = {"eval", "exec", "__import__"}


def find_violations(source: str) -> list[str]:
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
            out.append(f"from {node.module} import ...")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            out.append("bare except")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                out.append(f"{f.id}()")
            elif isinstance(f, ast.Attribute):
                owner = f.value.id if isinstance(f.value, ast.Name) else ""
                if owner == "os" and f.attr in {"system", "popen"}:
                    out.append(f"os.{f.attr}()")
                if owner == "yaml" and f.attr == "load":
                    out.append("yaml.load()")
                if owner == "torch" and f.attr == "load":
                    safe = any(k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value is True
                               for k in node.keywords)
                    if not safe:
                        out.append("torch.load without weights_only=True")
    return out


def test_scanner_actually_detects_violations():  # guards against a vacuous gate
    bad = "import pickle\nimport os\ntry:\n  eval('1')\nexcept:\n  pass\nos.system('x')\nimport torch\ntorch.load('f')\n"
    found = find_violations(bad)
    assert {"import pickle", "bare except", "eval()", "os.system()", "torch.load without weights_only=True"} <= set(found)
    assert find_violations("import torch\ntorch.load('f', weights_only=True)\n") == []


@pytest.mark.parametrize("path", sorted(PKG.glob("*.py")), ids=lambda p: p.name)
def test_package_has_no_dangerous_constructs(path):
    assert find_violations(path.read_text(encoding="utf-8")) == []


# ---------------------------------------------------------------- hostile-input gate


def _random_json(rng, depth=0):
    kinds = ["int", "float", "str", "none", "bool", "list", "dict"]
    k = kinds[int(rng.integers(len(kinds if depth < 3 else kinds[:5])))]
    if k == "int":
        return int(rng.integers(-10**12, 10**12))
    if k == "float":
        return float(rng.choice([0.0, -0.0, 1e308, -1e308, 1e-320, 3.14]))
    if k == "str":
        return "".join(chr(int(c)) for c in rng.integers(0, 0x2FF, int(rng.integers(0, 20))))
    if k == "none":
        return None
    if k == "bool":
        return bool(rng.integers(2))
    if k == "list":
        return [_random_json(rng, depth + 1) for _ in range(int(rng.integers(0, 6)))]
    return {"iq": _random_json(rng, depth + 1), "x": _random_json(rng, depth + 1)}


def test_hostile_input_never_causes_5xx(auth):
    client = TestClient(create_app(make_service(), make_settings(max_body_bytes=4096)))
    rng = np.random.default_rng(2026)
    bodies = [json.dumps(_random_json(rng)).encode() for _ in range(150)]
    bodies += [rng.bytes(int(rng.integers(0, 300))) for _ in range(100)]
    bodies += [b'{"iq": [[NaN, Infinity],[-Infinity, 1]]}', b"[" * 5000, b'{"iq":' + b"[" * 3000, b"\x00" * 100]
    for body in bodies:
        r = client.post("/v1/classify", content=body, headers={**auth, "Content-Type": "application/json"})
        assert r.status_code in {400, 413, 422}, (r.status_code, body[:60])
        assert r.json()["error"]["code"] and r.headers["x-request-id"]


# ---------------------------------------------------------------- behaviour under load


def test_concurrent_load_has_no_errors_and_bounded_latency():
    app = create_app(make_service(), make_settings())
    payload = good_payload(1024)

    def one(_):
        t = time.perf_counter()
        r = TestClient(app).post("/v1/classify", json=payload, headers={"X-API-Key": KEY})
        return r.status_code, time.perf_counter() - t

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(160)))
    assert {s for s, _ in results} == {200}
    lat = sorted(t for _, t in results)
    p95 = lat[int(0.95 * len(lat)) - 1]
    # Service-layer overhead only (fake backbone); loose bound so CI noise cannot flake it.
    assert p95 < 1.0, f"p95={p95:.3f}s median={statistics.median(lat):.3f}s"


def test_responses_are_deterministic(client, auth):
    a = client.post("/v1/classify", json=good_payload(seed=4), headers=auth).json()
    b = client.post("/v1/classify", json=good_payload(seed=4), headers=auth).json()
    a.pop("request_id"), b.pop("request_id")
    assert a == b


def test_production_profile_boots_and_serves():
    settings = make_settings(env="production", api_keys=frozenset({KEY}))
    c = TestClient(create_app(make_service(), settings))
    assert c.get("/healthz").status_code == 200 and c.get("/docs").status_code == 404
    r = c.post("/v1/classify", json=good_payload(), headers={"X-API-Key": KEY})
    assert r.status_code == 200 and "strict-transport-security" in r.headers


# ---------------------------------------------------------------- docs <-> code consistency


@pytest.mark.parametrize("name", ["PRD.md", "SECURITY.md", "ERROR_HANDLING.md", "TEST_PLAN.md"])
def test_required_docs_exist(name):
    assert len((DOCS / name).read_text().strip()) > 1500


def test_every_error_code_is_documented():
    text = (DOCS / "ERROR_HANDLING.md").read_text()
    codes = {c.code for c in vars(errors).values() if isinstance(c, type) and issubclass(c, errors.RFMCError)}
    assert [c for c in sorted(codes) if f"`{c}`" not in text] == []


def test_every_env_var_is_documented():
    readme = (ROOT / "README.md").read_text()
    used = set()
    for p in PKG.glob("*.py"):
        used |= set(re.findall(r"RFMC_[A-Z0-9_]+", p.read_text()))
    assert [v for v in sorted(used) if v not in readme] == []


# ---------------------------------------------------------------- model acceptance (needs real model + data)

EVAL_NPZ = os.environ.get("RFMC_EVAL_NPZ")


@pytest.mark.skipif(not EVAL_NPZ, reason="set RFMC_EVAL_NPZ and RFMC_MODEL_* to run model acceptance")
def test_model_meets_acceptance_thresholds():
    """npz keys: X (M,2,N) float32, y (M,) int class index or -1 for unknown, snr (M,) dB."""
    pytest.importorskip("torch")
    from rfmc.openset import auroc
    from rfmc.server import build_service_from_env
    from rfmc.validation import normalize_power

    min_acc_hi = float(os.environ.get("RFMC_MIN_ACC_HIGH_SNR", "0.95"))  # snr >= 10 dB
    min_acc_0 = float(os.environ.get("RFMC_MIN_ACC_0DB", "0.80"))  # -2 <= snr <= 2 dB
    min_auroc = float(os.environ.get("RFMC_MIN_AUROC", "0.90"))  # snr >= 10 dB
    max_false_reject = float(os.environ.get("RFMC_MAX_FALSE_REJECT", "0.10"))  # snr >= 10 dB

    d = np.load(EVAL_NPZ, allow_pickle=False)
    X, y, snr = d["X"], d["y"], d["snr"]
    svc = build_service_from_env()
    preds = [svc.classify(normalize_power(x).astype(np.float32)) for x in X]
    known = y >= 0
    correct = np.array([p.closest_known == svc.class_names[t] if t >= 0 else False for p, t in zip(preds, y)])
    score = np.array([p.openset_score for p in preds])
    flagged = np.array([p.unknown for p in preds])

    hi = known & (snr >= 10)
    assert correct[hi].mean() >= min_acc_hi, "closed-set accuracy at >=10 dB"
    mid = known & (snr >= -2) & (snr <= 2)
    assert correct[mid].mean() >= min_acc_0, "closed-set accuracy near 0 dB"
    assert flagged[hi].mean() <= max_false_reject, "known signals wrongly rejected at >=10 dB"
    unk_hi = (~known) & (snr >= 10)
    if unk_hi.any():
        assert auroc(score[hi], score[unk_hi]) >= min_auroc, "open-set AUROC at >=10 dB"
