# Error Handling: 2018 Dataset API Simulation Subsystem

## 1. Scope & Philosophy

The 2018API simulation subsystem interfaces with the core `rfmc` error architecture defined in [docs/ERROR_HANDLING.md](file:///c:/Users/ASUS/RF-Signal-Modulation-Classification/docs/ERROR_HANDLING.md). When using the API calling client as a filler for SDR hardware, errors may arise across three layers:
1. **Client / Data Feed Layer:** Dataset loading, sampling, JSON serialization, and timing drift.
2. **Transport / Network Layer:** Connection refusals, network timeouts, and HTTP status codes.
3. **API / Server Layer:** Authentication failure, rate limit throttling, validation rejections, and inference exceptions.

All errors conform to the standard `rfmc` JSON error envelope:
```json
{
  "status": "error",
  "request_id": "req-9b3c4f72-8a1e",
  "error": {
    "code": "ERROR_CODE_ENUM",
    "message": "Human-readable diagnostic message (generic in production)."
  }
}
```

---

## 2. Error Matrix for API Simulation

| Failure Scenario | Layer | Error Code | HTTP Status | Root Cause | Client Recovery Action |
|---|---|---|---|---|---|
| **Missing API Key** | Auth | `UNAUTHORIZED` | `401` | Request sent without `X-API-Key` header | Inject valid key via `--api-key` or `RFMC_API_KEYS` env variable |
| **Invalid / Weak API Key** | Auth | `UNAUTHORIZED` | `401` | Mismatched key or key length < 24 chars | Re-generate key with `scripts/generate_api_key.py` |
| **Rate Limit Exceeded** | Security | `RATE_LIMITED` | `429` | Client sending faster than 60 req/min (e.g. rate < 1.0s) | Backoff; clamp send interval $\ge 1.0\text{ s}$ per token-bucket rules |
| **Server Not Running** | Transport | `ECONNREFUSED` | N/A | Uvicorn/FastAPI server not started on target port | Launch server via `scripts/run_2018_api_server.py` |
| **Dataset Not Found** | Data Feed | `DATASET_NOT_FOUND` | N/A | `rml2018.npz` missing in `data/` | Run `scripts/download_rml2018.py --sample` |
| **Malformed I/Q Shape** | Validation | `INVALID_INPUT` | `422` | I/Q array is not $2 \times N$ or $N \notin [128, 4096]$ | Reshape sample array to canonical $(2, 1024)$ |
| **NaN or Inf Values** | Validation | `INVALID_INPUT` | `422` | Numerical overflow in simulated sample | Check power normalizer; reject unfinite frames |
| **Zero-Power Signal** | Validation | `ZERO_POWER` | `422` | I/Q samples are all zeros ($\sum |x_i|^2 = 0$) | Discard dead frames; verify dataset noise floor |
| **Inference Timeout** | Inference | `INFERENCE_TIMEOUT` | `504` | Model forward pass exceeded timeout (5.0s default) | Investigate server CPU/GPU load; optimize batch size |

---

## 3. Detailed Handling Protocols

### 3.1 Authentication & Credential Errors
- **Symptom:** Client receives HTTP 401 with `{"error": {"code": "UNAUTHORIZED"}}`.
- **Handling in Simulation Client:**
  The simulation client immediately halts rather than hammering the server, logging:
  `[ERROR] Authentication failed. Ensure RFMC_API_KEY matches server configuration.`
- **Mitigation:**
  Run `python scripts/generate_api_key.py --sync` to export matching keys to both `.env` and the client invocation command.

### 3.2 Rate Throttling & Clock Drift (1 Data/Sec Alignment)
- **Symptom:** Client receives HTTP 429 with `{"error": {"code": "RATE_LIMITED"}}`.
- **Handling in Simulation Client:**
  `rfmc` configures a sustained limit of 60 requests/min ($1.0\text{ req/sec}$). If system clock jitter causes two requests to fire within $<1.0\text{ s}$ after the token bucket burst is exhausted, the server rejects the request.
  The client implements an adaptive sleep mechanism:
  $$\Delta t_{\text{sleep}} = \max\left(0, T_{\text{target}} - (t_{\text{recv}} - t_{\text{send}})\right)$$
  If an HTTP 429 is encountered, the client performs an exponential backoff sleep of $1.5\text{ s} \to 2.0\text{ s}$ before resuming the 1 Hz cadence.

### 3.3 Missing or Corrupted Dataset File
- **Symptom:** `FileNotFoundError: data/rml2018.npz does not exist`.
- **Handling:**
  The `simulate_hardware_filler.py` script automatically verifies dataset existence on startup. If missing, it provides clear actionable instructions:
  ```text
  [WARNING] data/rml2018.npz not found!
  Run: python scripts/download_rml2018.py --sample
  to generate a calibrated 24-modulation test dataset.
  ```

### 3.4 Disconnection & Network Resilience
- **Symptom:** `requests.exceptions.ConnectionError: [Errno 111] Connection refused`.
- **Handling:**
  Rather than crashing ungracefully, the client enters a reconnect retry loop with a countdown timer (e.g. attempting reconnect every 2 seconds for up to 30 attempts) to permit the inference server to restart or load model weights.

---

## 4. Error Testing Checklist

- [x] Verify client response on missing `X-API-Key` (Expect HTTP 401).
- [x] Verify client response on invalid `X-API-Key` (Expect HTTP 401).
- [x] Verify client rate throttling when accelerated to 0.05s intervals (Expect HTTP 429).
- [x] Verify client resilience to server restart during an active streaming session.
- [x] Verify validation rejection for non-finite (NaN) simulated frames.

