# Product Requirements Document: RF Modulation Classification

| | |
|---|---|
| **Version** | 0.1 (draft) |
| **Date** | 2026-10-28 |
| **Source concept** | *Project Concept: RF Modulation Classification* (problems A, B, C) |
| **Status legend** | **Built** = in this repo and covered by tests. **Planned** = specified here, not yet built. |

## 1. Summary

An autonomous system that takes raw I/Q samples from a radio receiver and identifies the **modulation scheme** of the signal (for example BPSK, QPSK, 16-QAM, WBFM). It does not decode payloads. It works on three problems from the concept document:

| Problem | Need | Product answer |
|---|---|---|
| **A. Noise barrier** | Classify when SNR is low and rule-based receivers fail | Learned I/Q classifier trained across mixed SNR, with SNR-sliced evaluation |
| **B. Unknown threat** | Do not confidently mislabel a proprietary or foreign waveform | Open-set scoring: every prediction carries a "known-ness" score and low-score signals return `unknown` |
| **C. Spectrum congestion** | Know what is occupying the band and move to free spectrum | Wideband scanner: detect, channelise, classify, publish an occupancy map |

## 2. Goals and non-goals

**Goals**
1. Accurate closed-set modulation classification down to low SNR, with honest accuracy-vs-SNR reporting.
2. Reliable "unknown" detection that degrades gracefully in noise.
3. A production-grade service: validated inputs, authenticated access, uniform errors, observable, safe model loading.
4. A reproducible evaluation protocol that gates releases.

**Non-goals**
- Decoding, demodulating or decrypting message content.
- Transmitting. This system is **receive-only**.
- Claiming to identify specific military or encrypted systems. The honest claim is "does not match any known modulation structure".
- Geolocation or emitter fingerprinting.

## 3. Users and use cases

| User | Use case |
|---|---|
| Spectrum-monitoring engineer | Point an SDR at a band, see which modulations are present and which signals are unknown |
| Cognitive-radio / IoT researcher | Pick the emptiest channel from a live occupancy map |
| EW / security analyst (authorised use only) | Flag signals that do not match any known scheme for follow-up |
| ML engineer | Retrain, evaluate and ship new model versions through the release gates |

## 4. Scope by phase

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Input validation, error model, auth, rate limiting, secure model loading, open-set scoring, API, full test and release-gate suite | **Built** |
| 1 | Training pipeline, CNN baseline on RadioML, accuracy-vs-SNR report | Planned |
| 2 | Low-SNR improvements (CLDNN/Transformer, SNR-weighted training, augmentation) | Planned |
| 3 | Open-set training/calibration on held-out classes, AUROC report | Planned |
| 4 | Simulator-based robustness (fading, carrier offset, multipath) | Planned |
| 5 | SDR capture, wideband scanner, occupancy dashboard | Planned |

## 5. Functional requirements

| ID | Requirement | Pri | Status | Verified by |
|---|---|---|---|---|
| FR-1 | Accept an I/Q window as a `2 x N` array (`N` 128..4096) over `POST /v1/classify` | P0 | Built | `test_validation.py`, `test_errors.py` |
| FR-2 | Return predicted modulation, max-softmax probability and top-k | P0 | Built | `test_model_and_synth.py`, `test_errors.py` |
| FR-3 | Return `unknown` when the open-set score is below the calibrated threshold, plus the closest known class | P0 | Built (scoring); threshold must be calibrated per model | `test_openset.py`, `test_model_and_synth.py` |
| FR-4 | Meet accuracy targets across SNR bands (section 7) | P0 | Planned (needs trained model) | `test_production_gates.py::test_model_meets_acceptance_thresholds` |
| FR-5 | Reject malformed, non-finite, oversized or wrongly-shaped input with a precise 4xx | P0 | Built | `test_validation.py`, `test_production_gates.py` (hostile-input gate) |
| FR-6 | API-key authentication (fail closed), per-key and per-IP rate limiting | P0 | Built | `test_security.py` |
| FR-7 | Uniform JSON error envelope; no internal detail in responses | P0 | Built | `test_errors.py` |
| FR-8 | `/healthz` (liveness) and `/readyz` (model ready) | P0 | Built | `test_errors.py` |
| FR-9 | Load models only from an allowed directory, pinned by SHA-256, with pickle-safe loading | P0 | Built | `test_security.py`, `test_model_and_synth.py` (torch) |
| FR-10 | Wideband scan: detect occupied bands, channelise, classify each, publish occupancy map | P1 | Planned | n/a |
| FR-11 | Reproducible training and evaluation pipeline with held-out unknown classes | P1 | Planned | n/a |
| FR-12 | Metrics endpoint and structured request logs | P2 | Partial (structured logs only) | `test_security.py` (log hygiene) |

## 6. Non-functional requirements

| Area | Requirement |
|---|---|
| **Security** | See `SECURITY.md`. No secrets in repo, pinned dependencies, no pickle/eval/exec, production hides API docs and sets HSTS. |
| **Reliability** | A bad request never takes the process down. Inference has a hard timeout (default 5 s). Garbage model output is an error, never a wrong answer. |
| **Performance (proposed)** | p95 <= 100 ms per 1024-sample window on a 4-core CPU at batch size 1; to be measured once a real model exists. The repo's load gate only bounds service-layer overhead. |
| **Privacy and legal** | Passive reception and modulation classification only. Verify local spectrum, licensing and interception law before any over-the-air capture. Do not log payload samples. |
| **Operability** | Every response carries `X-Request-ID`; every error log line carries it. |
| **Portability** | Python >= 3.10, CPU-only inference supported. ONNX export is an optional later step. |

## 7. Success metrics (proposed targets, recalibrate after the Phase 1 baseline)

These numbers are starting points, not measured results. Set final values once the baseline exists, then freeze them as the release gate.

| Metric | Slice | Proposed target |
|---|---|---|
| Closed-set accuracy | SNR >= +10 dB | >= 95% |
| Closed-set accuracy | SNR around 0 dB | >= 80% |
| Open-set AUROC (known vs held-out unknown) | SNR >= +10 dB | >= 0.90 |
| False-reject rate on known signals | SNR >= +10 dB | <= 10% |
| p95 latency | 1024 samples, CPU | <= 100 ms |

Always report the full accuracy-vs-SNR curve and the confusion matrix, not only the headline numbers. Expect known confusable pairs (16-QAM vs 64-QAM, AM-DSB vs WBFM).

## 8. API contract

`POST /v1/classify`, headers `X-API-Key`, `Content-Type: application/json`

```json
{ "iq": [[I0, I1, ...], [Q0, Q1, ...]], "sample_rate_hz": 2000000 }
```

Unknown fields are rejected. `sample_rate_hz` is optional.

Success (200):

```json
{
  "request_id": "…", "modulation": "QPSK", "unknown": false, "closest_known": "QPSK",
  "confidence": 0.97, "openset_score": 0.97,
  "top_k": [{"label": "QPSK", "probability": 0.97}],
  "num_samples": 1024, "model_version": "…"
}
```

When `unknown` is true, `modulation` is `"unknown"` and `closest_known` still names the nearest known class.

| Status | Meaning |
|---|---|
| 400 | Malformed JSON or Content-Length |
| 401 | Missing or invalid key (identical response for both) |
| 413 / 415 | Body too large / wrong content type |
| 422 | Valid JSON, invalid I/Q data |
| 429 | Rate limited (`Retry-After` set) |
| 500 / 503 / 504 | Inference failure / model not ready / inference timeout |

Full error catalogue: `ERROR_HANDLING.md`.

## 9. Architecture

```
SDR / file ──▶ [Scanner: detect + channelise]  (Phase 5)
                         │ I/Q windows
                         ▼
 client ──HTTPS──▶ proxy/TLS ──▶ API middleware (id, size cap, headers, catch-all)
                                   │ auth → rate limit → content-type → parse → validate
                                   ▼
                          ClassifierService
                   backbone(logits, embedding) ──▶ output guards (shape, finite)
                                   │
                          OpenSetDetector (msp | energy | mahalanobis) + calibrated threshold
                                   ▼
                          label | "unknown" + closest_known
```

## 10. Data and evaluation protocol

- **Datasets:** RadioML 2016.10A for fast iteration, RadioML 2018.01A for headline numbers (check its non-commercial licence), plus a simulator for controlled fading/offset and for held-out "unknown" modulations, plus a small real SDR capture set to measure the sim-to-real gap.
- **Splits:** by capture/channel realisation, never random by sample, to avoid leakage.
- **Open-set protocol:** train on K known classes, hold out several modulations as unknown, calibrate the threshold at a fixed true-positive rate on **known validation data only**, report AUROC and false-reject rate per SNR band.
- **Acceptance file format** (used by the release gate): `.npz` with `X (M,2,N) float32`, `y (M,) int` (class index, `-1` = unknown), `snr (M,) dB`.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sim-to-real gap makes lab numbers misleading | Real SDR test set; headline numbers from held-out real or channel-split data |
| RML2016 dataset artefacts inflate accuracy | Prefer RML2018 or own simulator for headline results |
| Open-set detection collapses at low SNR | Report it per SNR band; do not hide it behind a single AUROC |
| Adversarial perturbations of I/Q fool CNN classifiers | Treat output as advisory; document limitation; consider adversarial training later |
| Tampered model file | Directory confinement, suffix allow-list, SHA-256 pin, `weights_only` load |
| Misuse for interception | Non-goals above; no payload decoding; legal review before OTA use |
| Per-process rate limiter in multi-replica deploys | Enforce limits at the gateway too (see `SECURITY.md`) |

## 12. Open questions

1. Target hardware (laptop GPU, edge device, server CPU)? This sets the latency budget and quantisation needs.
2. Which SDR will be used first (RTL-SDR, HackRF, ADALM-Pluto)? This fixes sample rate and bandwidth limits for Phase 5.
3. Is this a hackathon demo, a paper, or a product? This decides how strict the release gates and licence checks need to be.
4. Final class list: the 11 RML2016 classes, the 24 RML2018 classes, or a custom set?
