# rfmc: RF modulation classification service

Classifies the modulation of raw I/Q samples, flags signals that match no known modulation as `unknown`, and serves this through a hardened API. See `docs/` for the PRD, security model, error handling and test plan.

## Layout

```
rfmc/            errors, config, validation, security, openset, model, model_torch, synth, api, server
tests/           unit + security + error tests;  tests/production/ = release gates
docs/            PRD.md  SECURITY.md  ERROR_HANDLING.md  TEST_PLAN.md
```

## Quick start

```bash
pip install -r requirements-dev.txt
pytest                      # all tests
pytest -m production        # release gates
```

Serve (needs `torch` and a trained, hashed model file):

```bash
export RFMC_ENV=production RFMC_API_KEYS="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
export RFMC_CLASSES=BPSK,QPSK,8PSK,QAM16 RFMC_OPENSET_THRESHOLD=0.62
export RFMC_MODEL_DIR=/models RFMC_MODEL_PATH=/models/rfnet.pt RFMC_MODEL_SHA256=<sha256>
uvicorn rfmc.server:app --host 0.0.0.0 --port 8000   # put a TLS proxy in front
```

```bash
curl -s localhost:8000/v1/classify -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"iq": [[...128+ I samples...], [...matching Q samples...]], "sample_rate_hz": 2000000}'
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `RFMC_ENV` | `development` | `development`, `test` or `production` (production enforces strict key rules, hides docs, sends HSTS) |
| `RFMC_API_KEYS` | none | Comma-separated API keys; production needs >= 24 chars each |
| `RFMC_MAX_BODY_BYTES` | 262144 | Request body cap (1 KiB to 10 MiB) |
| `RFMC_RATE_LIMIT_PER_MIN` | 60 | Per-key sustained rate |
| `RFMC_RATE_LIMIT_BURST` | 10 | Per-key burst |
| `RFMC_IP_RATE_LIMIT_PER_MIN` | 600 | Per-IP sustained rate (applied before auth) |
| `RFMC_INFERENCE_TIMEOUT_S` | 5.0 | Hard inference timeout (0 to 60) |
| `RFMC_CLASSES` | required | Comma-separated class names in training order |
| `RFMC_OPENSET_THRESHOLD` | required | Calibrated open-set threshold (max-softmax) |
| `RFMC_MODEL_DIR` | required | Only directory models may be loaded from |
| `RFMC_MODEL_PATH` | required | Path to the `.pt` state dict inside the model dir |
| `RFMC_MODEL_SHA256` | required | Expected SHA-256 of the model file |
| `RFMC_MODEL_VERSION` | `unversioned` | Reported in responses |

## Status

Built and tested: validation, error model, auth and rate limiting, safe model loading, open-set scoring, API, release gates. **Not yet built:** training pipeline, trained weights, SDR capture and the wideband scanner (see PRD phases 1 to 5).
