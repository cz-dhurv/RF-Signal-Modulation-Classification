# Test Plan: 2018API & Sample Simulation Subsystem

## 1. Objectives

This test plan validates the **Hardware Filler via API Simulation (2018API)** subsystem. It ensures that:
1. The API key generator creates cryptographically secure keys conforming to production constraints.
2. The 2018 dataset fetcher and generator produces valid `(2, 1024)` power-normalised tensors covering all 24 canonical RadioML 2018 modulations.
3. The hardware filler simulation client streams I/Q frames accurately at **1 data frame per second (1 Hz)**.
4. The system honors authentication (`X-API-Key`) and rejects invalid/unauthenticated requests with HTTP 401.
5. Rate limiting conforms to token-bucket parameters without triggering false-positive throttles.
6. Existing codebase integrity is strictly preserved (zero modifications to existing source code).

---

## 2. Test Suite Matrix

| Test ID | Category | Component | Description | Expected Result |
|---|---|---|---|---|
| **TP-SIM-01** | Security | `generate_api_key.py` | Generate new API key; verify length $\ge 32$ chars, URL-safe format | Valid string, exits code 0 |
| **TP-SIM-02** | Data | `download_rml2018.py` | Generate sample dataset with 24 classes | File `data/rml2018.npz` created, shape `(M, 2, 1024)`, unit power |
| **TP-SIM-03** | Auth | `simulate_hardware_filler.py` | Attempt API call with missing or invalid `X-API-Key` | HTTP 401 `UNAUTHORIZED` returned |
| **TP-SIM-04** | Auth | `simulate_hardware_filler.py` | Send classification payload with valid `X-API-Key` | HTTP 200 `OK`, prediction payload returned |
| **TP-SIM-05** | Cadence | `simulate_hardware_filler.py` | Measure inter-request timing over 5 frames at `--rate 1.0` | Interval is $1.00\text{ s} \pm 0.08\text{ s}$ per frame |
| **TP-SIM-06** | Limits | `simulate_hardware_filler.py` | Accelerated send at `--rate 0.05` to exhaust burst limit | HTTP 429 `RATE_LIMITED` returned after burst |
| **TP-SIM-07** | E2E | Simulation Pipeline | Run 10-second end-to-end simulation against active API | 10 predictions logged with modulation, confidence & SNR |
| **TP-SIM-08** | Regression | Existing Codebase | Verify zero modifications in `rfmc/` and existing scripts | `git diff main -- rfmc/` returns 0 changes |

---

## 3. Step-by-Step Execution Guide

### Test 1: Verify API Key Generation
```bash
python scripts/generate_api_key.py --env
# Check that .env file exists and contains RFMC_API_KEYS with >= 32 characters
```

### Test 2: Generate and Verify RadioML 2018 Dataset
```bash
python scripts/download_rml2018.py --sample --samples-per-class 50 --output data/rml2018.npz
# Verify dataset summary:
python -c "import numpy as np; d=np.load('data/rml2018.npz'); print('X:', d['X'].shape, 'Classes:', len(d['class_names']))"
# Expected output: X: (1200, 2, 1024) Classes: 24
```

### Test 3: Start Inference API Server
```bash
# In Terminal 1:
python scripts/run_2018_api_server.py --port 8000
```

### Test 4: Run Hardware Filler Simulation (1 Data / Second)
```bash
# In Terminal 2:
python scripts/simulate_hardware_filler.py --url http://127.0.0.1:8000/v1/classify --count 10 --rate 1.0
```

**Expected Terminal Output:**
```text
================================================================================
     RF Modulation Classification - Virtual SDR Hardware Filler (2018API)      
================================================================================
[INFO] Loaded 1200 frames from data/rml2018.npz across 24 modulations.
[INFO] Ingestion rate: 1.00 data frame/sec (1 Hz) | Target: http://127.0.0.1:8000/v1/classify
[INFO] Authentication: API-Key enabled (fingerprint: 4f8a...c291)
--------------------------------------------------------------------------------
[00:01] True: QPSK (SNR: +10 dB) -> Pred: QPSK (Conf: 0.96) [RTT: 8ms]
[00:02] True: 16QAM (SNR: +18 dB) -> Pred: 16QAM (Conf: 0.93) [RTT: 7ms]
[00:03] True: BPSK (SNR: -4 dB) -> Pred: BPSK (Conf: 0.78) [RTT: 9ms]
[00:04] True: FM (SNR: +12 dB) -> Pred: FM (Conf: 0.99) [RTT: 8ms]
...
--------------------------------------------------------------------------------
[INFO] Completed 10 simulated frames. Success rate: 100.0%. Average RTT: 8.1 ms.
```

---

## 4. Acceptance Criteria

- [x] Zero changes to pre-existing files in repository (`git status` confirms untracked additions only).
- [x] API client communicates seamlessly with `/v1/classify` using valid `X-API-Key`.
- [x] Stream rate precisely regulated at 1 data frame per second.
- [x] Dataset compliant with RadioML 2018.01A tensor dimensions (`2 x 1024`).
- [x] All 5 documentation files present and contextually updated in `docs/sample_simulation/`.

