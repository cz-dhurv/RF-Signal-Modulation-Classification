# Project Status: What Is Built and What Remains

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Overall** | Production scaffolding is done and tested. The ML core (trained model, training pipeline, scanner) is not started. |
| **Tests** | 213 passing, 2 skipped (torch not installed here; no evaluation data supplied) |

## 1. Built

| Area | What exists | Where | Proven by |
|---|---|---|---|
| Input validation | Strict `2 x N` I/Q checks: types, shape, length 128..4096, NaN/Inf, magnitude, zero power, unknown fields; power normalisation | `rfmc/validation.py` | `tests/test_validation.py` (40+ bad-input cases, fuzz) |
| Error model | One JSON error envelope, typed errors, generic messages for server-side failures, request ids | `rfmc/errors.py`, `rfmc/api.py` | `tests/test_errors.py` |
| Security | API-key auth (constant-time, fails closed), per-IP and per-key rate limits, body size caps, security headers, log sanitising, model path confinement and SHA-256 pinning | `rfmc/security.py`, `rfmc/api.py` | `tests/test_security.py` |
| Config | Validated settings from environment; production refuses weak or missing keys | `rfmc/config.py` | `tests/test_security.py` |
| Open-set scoring | Max-softmax, energy and Mahalanobis scores, threshold calibration, AUROC, NaN resolves to `unknown` | `rfmc/openset.py` | `tests/test_openset.py` |
| Inference service | Wraps any model; checks output shape and finiteness; returns label or `unknown` plus closest known class | `rfmc/model.py` | `tests/test_model_and_synth.py` |
| API | `POST /v1/classify`, `/healthz`, `/readyz`, timeout handling | `rfmc/api.py`, `rfmc/server.py` | `tests/test_errors.py`, `tests/test_security.py` |
| Signal generator | BPSK, QPSK, 8PSK, 16-QAM with AWGN at a requested SNR | `rfmc/synth.py` | `tests/test_model_and_synth.py` |
| Torch model code | A CNN (`RFNet`) and a safe loader (`weights_only=True`) | `rfmc/model_torch.py` | Only tested where torch is installed |
| Release gates | Pinned dependencies, secrets scan, dangerous-code scan, hostile-input gate, load gate, docs-match-code check, model acceptance gate | `tests/production/` | Checked by deliberately breaking the code: the gates fail as expected |
| Documents | PRD, security model, error handling, test plan, README | `docs/`, `README.md` | Docs-match-code gate |

## 2. Not built

| # | Item | PRD phase | Why it matters | Depends on |
|---|---|---|---|---|
| 1 | **Trained model.** No weights exist, so the API cannot classify real signals. The tests use a fake model. | 1 | The core deliverable | Item 2 |
| 2 | **Training pipeline.** Dataset loading, training loop, evaluation, accuracy-vs-SNR curves, confusion matrix. | 1 | Produces the model and the numbers every target depends on | Dataset choice, hardware |
| 3 | **Low-SNR improvements.** CNN+LSTM or Transformer, SNR-weighted training, augmentation (phase, frequency offset, time shift). | 2 | Your Problem A | Item 2 |
| 4 | **Open-set training and calibration.** Hold out classes as unknown, calibrate the threshold on known validation data, report AUROC per SNR band. | 3 | Your Problem B. The scoring code exists but has never run on a real model. | Items 2, 3 |
| 5 | **Realistic simulator.** Fading, carrier offset, multipath, held-out unknown modulations. The current generator is basic (1 sample per symbol, AWGN only). | 4 | Robustness testing and a clean unknown-class set | None |
| 6 | **SDR capture.** Reading live I/Q from RTL-SDR, HackRF or ADALM-Pluto. | 5 | Real-signal testing; measures the sim-to-real gap | Hardware |
| 7 | **Wideband scanner.** Band-occupancy detection, channelisation, per-channel classification. | 5 | Your Problem C | Items 1, 4, 6 |
| 8 | **Occupancy dashboard and free-band selection.** | 5 | The cognitive-radio demo | Item 7 |
| 9 | **Real-data acceptance run.** The model gate is written but has never run with real data; targets in the PRD are still proposals. | 1 to 3 | Turns proposed targets into frozen release criteria | Items 2, 4 |

## 3. Production hardening still open

| Item | Detail |
|---|---|
| Torch not in `requirements.txt` | Serving needs it; add a pinned version once the training environment is chosen |
| Torch loader coverage | `rfmc/model_torch.py` and the `uvicorn rfmc.server:app` path are untested without torch |
| Packaging and CI | No Dockerfile, no CI pipeline, no `pip-audit` step yet |
| Metrics and alerting | Only structured logs exist; no metrics endpoint, no error-rate or latency alerts, no circuit breaker |
| Multi-replica rate limiting | Limiter is per process; enforce at a gateway for more than one worker |
| Real latency | Load gate uses a fake model; measure p95 on target hardware |
| ONNX/quantised export | Optional, only needed for edge targets |

## 4. Known caveats for the remaining work

- **RadioML 2016.10A is distributed as a Python pickle file.** Do not load it inside the service. Convert it once to `.npz` in an isolated script outside `rfmc/` (the security gate scans only `rfmc/`), and check the file hash first.
- **RadioML does not expose capture or channel ids**, so the leakage-safe split the PRD calls for is only fully possible with your own simulator or real captures. Report this limit with any RadioML numbers.
- **RadioML 2018 is non-commercial licensed.** Check before any product use.
- **Adversarial robustness is not addressed anywhere** (see `SECURITY.md`, section 3).
- **Legal check before any over-the-air capture.** The system is receive-only and classifies modulation without decoding content, but local spectrum and interception rules still apply.

## 5. Suggested order

1. Choose dataset, hardware and goal (demo, paper or product). These three answers set scope.
2. Phase 1: data converter, training script, CNN baseline, accuracy-vs-SNR report. Add torch to the pinned requirements.
3. Run the model acceptance gate on the baseline, then freeze the numeric targets.
4. Phase 3: open-set training and threshold calibration. Then phase 2 improvements for low SNR.
5. Phase 4 simulator and robustness tests.
6. Phase 5 only after the above: SDR capture, scanner, dashboard.
7. Packaging, CI and metrics before any deployment.

## 6. Open decisions needed from you

1. RadioML 2016 (small, fast) or 2018 (larger, better headline numbers)?
2. Training hardware: GPU or CPU only?
3. Hackathon demo, paper, or product?
4. Which SDR, if any, will you use first?
5. Final class list: 11 (RML2016), 24 (RML2018), or custom?
