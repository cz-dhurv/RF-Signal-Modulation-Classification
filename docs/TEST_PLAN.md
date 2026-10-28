# Validation and Test Plan

## 1. How to run

```bash
pip install -r requirements-dev.txt
pytest                      # everything (model acceptance is skipped without data)
pytest -m "not production"  # fast unit tests only
pytest -m production        # release gates only
```

## 2. Test layout

| File | Covers |
|---|---|
| `tests/test_validation.py` | Input validation matrix, boundary lengths, fuzz (only `ValidationError` may escape), normalisation |
| `tests/test_openset.py` | softmax stability, MSP/energy/Mahalanobis, calibration, AUROC vs brute force, NaN fail-safe |
| `tests/test_model_and_synth.py` | Signal generator (power, SNR accuracy, constellations, determinism), inference service guards, torch round-trip (skipped without torch) |
| `tests/test_security.py` | Auth, timing safety, rate limiter, request-id and log sanitising, model path/hash checks, settings hardening, headers, size caps, nesting bomb, log hygiene |
| `tests/test_errors.py` | Envelope shape, no-leak guarantees, status mapping, timeout, readiness, recovery after errors, success contract |
| `tests/production/test_production_gates.py` | Release gates (section 4) |
| `tests/conftest.py` | Fake backbone and app fixtures so no trained model is needed |

## 3. Validation rules (input contract)

| Field | Rule | Failure |
|---|---|---|
| Body | Valid JSON object, `application/json`, <= `RFMC_MAX_BODY_BYTES` | 400 / 415 / 413 |
| Unknown fields | Rejected | 422 |
| `iq` | Required; real numbers only (not strings, bools, complex, null, bigints) | 422 |
| `iq` shape | Exactly 2 rows (I, Q), equal length, rectangular | 422 (reports received shape) |
| `iq` length | `N` in [128, 4096] samples per row (configurable) | 422 |
| `iq` values | Finite (no NaN/Inf), magnitude <= 1e6, not all zero | 422 |
| `iq` scaling | Normalised to unit mean power (safe for 1e-200 .. 1e6) | n/a |
| `sample_rate_hz` | Optional; positive, finite, <= 1e11, not bool or string | 422 |
| Model output | Logits shape `(1, K)`, logits and embeddings finite | 500 `inference_error` |
| Open-set score | NaN is treated as `unknown` | n/a |
| Settings | Validated at startup; production keys >= 24 chars | process refuses to start |

## 4. Production release gates (`pytest -m production`)

| Gate | What it proves |
|---|---|
| Pinned dependencies | Every requirement uses `==` |
| No committed secrets | Key/private-key patterns absent; no `.env` files |
| Static security scan | No pickle/marshal/subprocess/eval/exec/os.system/yaml.load, no bare `except`, `torch.load` always `weights_only=True` |
| Scanner self-test | The scan actually detects violations (not vacuous) |
| Hostile-input gate | 250+ random/garbage bodies, nesting bombs, NaN literals: always 400/413/422 with envelope, never 5xx |
| Concurrent load | 160 requests over 8 threads: all 200, p95 < 1 s (service-layer overhead with a fake model) |
| Determinism | Same input gives identical output |
| Production boot | Production profile serves, hides docs, sends HSTS |
| Docs consistency | Required docs exist; every error code is in `ERROR_HANDLING.md`; every `RFMC_*` variable is in `README.md` |
| **Model acceptance** | **Skipped unless `RFMC_EVAL_NPZ` is set.** Closed-set accuracy at >= 10 dB and near 0 dB, false-reject rate, open-set AUROC |

### Running the model acceptance gate

```bash
export RFMC_EVAL_NPZ=/data/eval.npz         # X (M,2,N) float32, y (M,) class idx or -1 unknown, snr (M,)
export RFMC_CLASSES=BPSK,QPSK,8PSK,QAM16     # same order as training
export RFMC_MODEL_DIR=/models RFMC_MODEL_PATH=/models/rfnet.pt RFMC_MODEL_SHA256=<hash>
export RFMC_OPENSET_THRESHOLD=0.62           # calibrated on known validation data only
pytest -m production -k acceptance
```

Thresholds default to the proposed targets in the PRD and can be overridden with `RFMC_MIN_ACC_HIGH_SNR`, `RFMC_MIN_ACC_0DB`, `RFMC_MIN_AUROC`, `RFMC_MAX_FALSE_REJECT`. Freeze the final values after the Phase 1 baseline.

## 5. Requirement traceability

| PRD | Tests |
|---|---|
| FR-1, FR-5 | `test_validation.py`, hostile-input gate |
| FR-2, FR-3 | `test_model_and_synth.py`, `test_openset.py`, `test_errors.py::test_unknown_signal_is_flagged_over_http` |
| FR-4 | Model acceptance gate |
| FR-6 | `test_security.py` |
| FR-7, FR-8 | `test_errors.py` |
| FR-9 | `test_security.py` (path, hash), static scan, torch round-trip |

## 6. What these tests do not prove

- **Real-world RF performance.** No trained model or real captures are in the repo. The accuracy/AUROC gate only runs when you supply data.
- **Torch code path.** `rfmc/model_torch.py` is only tested where torch is installed.
- **Realistic latency.** The load gate uses a fake model and bounds only service overhead; measure real p95 on target hardware.
- **Adversarial robustness and multi-replica rate limiting.** See `SECURITY.md`, section 3.
- **Penetration testing.** These are automated checks, not a substitute for an independent security review.
