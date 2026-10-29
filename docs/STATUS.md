# Project Status: What Is Built and What Remains

| | |
|---|---|
| **Date** | 2026-10-29 |
| **Overall** | Phases 0–5 core modules, training/evaluation pipelines, RadioML 2016 dataset integration, and Google Colab training notebook are fully built and verified. |
| **Tests** | 305 passing, 2 skipped (1 symlink privilege check on Windows, 1 release-gate awaiting live OTA weights) |

## 1. Built

| Area | What exists | Where | Proven by |
|---|---|---|---|
| Input validation | Strict `2 x N` I/Q checks: types, shape, length 128..4096 (up to 131072 for scan), NaN/Inf, magnitude, zero power, unknown fields; power normalisation | `rfmc/validation.py` | `tests/test_validation.py` (40+ bad-input cases, fuzz) |
| Error model | One JSON error envelope, typed errors, generic messages for server-side failures, request ids | `rfmc/errors.py`, `rfmc/api.py` | `tests/test_errors.py` |
| Security | API-key auth (constant-time, fails closed), per-IP and per-key rate limits, body size caps, security headers, log sanitising, model path confinement and SHA-256 pinning | `rfmc/security.py`, `rfmc/api.py` | `tests/test_security.py` |
| Config | Validated settings from environment; production refuses weak or missing keys | `rfmc/config.py` | `tests/test_security.py` |
| Open-set scoring | Max-softmax, energy and Mahalanobis scores, threshold calibration, AUROC, NaN resolves to `unknown` | `rfmc/openset.py` | `tests/test_openset.py` |
| Inference service | Wraps any model; checks output shape and finiteness; returns label or `unknown` plus closest known class | `rfmc/model.py` | `tests/test_model_and_synth.py` |
| API | `POST /v1/classify`, `POST /v1/scan`, `/healthz`, `/readyz`, timeout handling | `rfmc/api.py`, `rfmc/server.py` | `tests/test_errors.py`, `tests/test_security.py`, `tests/test_scanner.py` |
| Signal generator & simulator | Deterministic synth (synth.py) + advanced simulator: pulse shaping (RRC), carrier frequency/clock offset, Rayleigh/Rician fading, multipath, held-out unknown modulations | `rfmc/synth.py`, `rfmc/simulator.py` | `tests/test_simulator_augment.py` (74 tests) |
| Data augmentation | Channel impairment transforms: random phase, frequency offset, time shift, amplitude scaling, AWGN, Rayleigh/Rician fading, I/Q imbalance | `rfmc/augment.py` | `tests/test_simulator_augment.py` |
| PyTorch models | Multiple backbones: `RFNet` (CNN baseline), `RFCLDNN` (CNN+BiLSTM for low SNR), `RFTransformer` (1D Self-Attention), and safe loader (`weights_only=True`) | `rfmc/model_torch.py` | `tests/test_model_and_synth.py` |
| Training pipeline | Stratified dataset loading, training loop with warmup & cosine/step schedules, SNR-weighted loss, checkpoints, accuracy-vs-SNR and confusion matrix plots, model card export | `scripts/train.py` | Verified with simulated training run & RadioML 2016 run |
| RadioML converter & dataset | Safe CLI converting RML2016 pickle and RML2018 HDF5 to power-normalised `.npz` format. `data/rml2016.npz` (220,000 samples, 11 classes, SNR [-20, +18] dB) created and verified | `scripts/convert_radioml.py`, `data/rml2016.npz` | Tested & verified |
| Colab GPU notebook | End-to-end interactive training, evaluation, open-set calibration, and artifact download notebook with Google Colab extension support | `notebooks/train_rml2016_colab.ipynb` | Verified structure |
| Local GPU acceleration | Verified training and inference execution on NVIDIA GeForce RTX 3050 Laptop GPU via PyTorch CUDA | `scripts/train.py` | Verified with trial epoch run |
| Open-set calibration | Calibration of MSP, Energy, Mahalanobis on known validation data at target TPR, AUROC vs SNR curves, score distributions, environment variable recommendations | `scripts/calibrate_openset.py` | Verified (98.7% high-SNR AUROC on simulated holdouts) |
| Standalone evaluator | Model evaluation on test data, SNR-sliced reporting, acceptance file generator for release gate | `scripts/evaluate.py` | Verified with acceptance export |
| Wideband scanner & cognitive radio | Welch PSD computation, adaptive noise floor estimation, active emitter detection, DDC + decimation channelisation, per-channel classification, free spectrum search | `rfmc/scanner.py` | `tests/test_scanner.py` |
| Release gates | Pinned dependencies, secrets scan, dangerous-code AST scan, hostile-input fuzz, load gate, docs-match-code check, model acceptance gate | `tests/production/` | 27 passing, release criteria validated |
| Documents | PRD, security model, error handling, test plan, README | `docs/`, `README.md` | Docs-match-code gate |

## 2. What remains

| # | Item | PRD phase | Why it matters | Depends on |
|---|---|---|---|---|
| 1 | **RadioML extended multi-epoch run.** Complete extended 50-60 epoch benchmark training run on GPU (locally or via Google Colab) to produce finalized release weights. | 1 | Final benchmark numbers for paper or production deployment | GPU execution |
| 2 | **Live SDR capture module.** Driver / script reading live I/Q over USB (RTL-SDR, HackRF, ADALM-Pluto) to feed `/v1/scan`. | 5 | Measures real over-the-air sim-to-real gap | Hardware SDR |
| 3 | **Interactive occupancy dashboard.** UI visualizing live spectral waterfall, detected emitters, and cognitive radio free-band channels. | 5 | Visual operator cognitive-radio demonstration | Stream/WebSocket or polling UI |

## 3. Production hardening status

| Item | Status | Detail |
|---|---|---|
| PyTorch pinned in requirements | Completed | Added `torch==2.6.0` to `requirements.txt` |
| Evaluation dependencies pinned | Completed | Added `scikit-learn==1.7.0`, `matplotlib==3.10.3`, `prometheus-client==0.22.1` |
| Multi-arch backbones | Completed | `RFNet`, `RFCLDNN`, `RFTransformer` in `rfmc/model_torch.py` |
| AST safety scan | Passing | All 12 files in `rfmc/` pass without dangerous constructs |
| Secrets scan | Passing | Clean UTF-8 secrets scan on Windows & Linux |
| RadioML 2016.10A integrated | Completed | Converted and verified in `data/rml2016.npz` |
| Colab GPU pipeline ready | Completed | `notebooks/train_rml2016_colab.ipynb` added |
