# Project Analysis: 2018 Dataset API Simulation Subsystem

## 1. Executive Summary & Context

The **RF-Signal-Modulation-Classification** repository was designed to process high-frequency electromagnetic raw baseband signals (In-Phase / Quadrature, $I/Q$), determine signal modulations accurately under noisy and multipath conditions, and detect unmodelled or hostile signals using open-set recognition.

Under full production deployment, the system ingests live I/Q samples directly from a Software-Defined Radio (SDR) hardware front-end (e.g. RTL-SDR v4, HackRF One, USRP B210). However, in development, validation, or lab environments without connected SDR hardware, live physical spectrum feeds are unavailable. 

The **2018API** branch introduces the **Hardware Filler via API Simulation**, a drop-in replacement that clocks authentic **RadioML 2018.01A** frames into the classification engine over REST at **1 sample frame per second**. This enables comprehensive verification of the model pipeline, API authorization, rate limiting, and inference analytics without needing physical hardware.

---

## 2. Dual-Feature System Decomposition

The project explicitly bifurcates signal ingestion into two distinct features:

```
+-----------------------------------------------------------------------------------------+
|                                    System Ingress Paths                                 |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|   +---------------------------------------+   +-------------------------------------+   |
|   |    FEATURE 1: Physical Hardware       |   |    FEATURE 2: API Stream Simulation |   |
|   |         (When Available)              |   |           (Branch: 2018API)         |   |
|   +---------------------------------------+   +-------------------------------------+   |
|   | - Physical Antenna & Mixer Front-End  |   | - RadioML 2018.01A (24 modulations) |   |
|   | - Direct USB/PCIe sample streaming    |   | - 1 data frame per second (1 Hz)    |   |
|   | - Hardware gain, AGC, DC offset null  |   | - Validated API Key Authentication  |   |
|   | - Sub-millisecond continuous capture  |   | - JSON REST Payload over HTTP(S)    |   |
|   | - feeds Wideband Scanner / Local IPC  |   | - Hardware Filler Surrogate         |   |
|   +---------------------------------------+   +-------------------------------------+   |
|                       |                                           |                     |
|                       +--------------------+----------------------+                     |
|                                            |                                            |
|                                            v                                            |
|                       +-------------------------------------------+                     |
|                       |     Normalized (2, N) I/Q Sample Array    |                     |
|                       +-------------------------------------------+                     |
|                                            |                                            |
|                                            v                                            |
|                       +-------------------------------------------+                     |
|                       |       Neural Inference & Open-Set         |                     |
|                       +-------------------------------------------+                     |
+-----------------------------------------------------------------------------------------+
```

### Feature 1: Hardware-Driven Ingress (Physical SDR)
- **Status:** Specification established; awaiting hardware availability.
- **Operating Model:** Direct-sampling or superheterodyne RF tuner delivering $2\times N$ float arrays directly into memory via native C/Cython drivers (`librtlsdr`, `libhackrf`, `SoapySDR`).
- **Use Case:** Field tactical monitoring, wideband spectrum survey, cognitive radio dynamic spectrum access.

### Feature 2: API-Driven Simulation (Hardware Filler)
- **Status:** **Active & Implemented on branch `2018API`**.
- **Operating Model:** Software client loads validated signal tensors from the RadioML 2018.01A dataset and transmits 1 I/Q frame every $1.0\text{ s}$ over HTTP POST to `/v1/classify`.
- **Use Case:** End-to-end regression testing, UI waterfall demonstration, model performance validation across SNR gradients ($-20\text{ dB}$ to $+30\text{ dB}$), and production security validation.

---

## 3. Why 1 Data Per Second (1 Hz Clocking)?

1. **Human-Observable Telemetry:**
   High-frequency SDRs capture millions of samples per second ($2\text{ MSps}$ to $20\text{ MSps}$). Streaming raw packets at RF sample rates over JSON HTTP exhausts network stack and rate limiters. A rate of **1 data per second** mirrors typical monitoring dashboard refresh cycles, enabling clear terminal logging and live inspection.

2. **Compliance with Security & Rate Limiting:**
   `rfmc` incorporates a strict **Token Bucket Rate Limiter** (`rfmc/security.py`):
   - Default sustained rate: $60\text{ requests/min} = 1.0\text{ req/sec}$.
   - Burst cap: $10\text{ tokens}$.
   Clocking the hardware filler at 1 data per second aligns with the API's sustained rate limit without triggering HTTP 429 (`RATE_LIMITED`).

3. **Realistic Inference Budget:**
   Inference overhead for a 1024-length 1D CNN/CLDNN is typically $2\text{ ms} - 15\text{ ms}$ on CPU. An interval of $1.0\text{ s}$ leaves $>98\%$ idle duty cycle for network transport, telemetry parsing, and dashboard rendering.

---

## 4. RadioML 2018.01A Analytical Specification

The RadioML 2018.01A dataset (DeepSig) is the gold standard benchmark for automatic modulation classification:

- **Modulation Diversity (24 classes):**
  - Linear single-carrier digital: BPSK, QPSK, 8PSK, 16PSK, 32PSK, OQPSK
  - High-order Quadrature Amplitude: 16QAM, 32QAM, 64QAM, 128QAM, 256QAM
  - Amplitude Phase Shift Keying: 16APSK, 32APSK, 64APSK, 128APSK
  - Constant envelope / frequency: GMSK, OOK, 4ASK, 8ASK
  - Continuous-wave analog: AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM
- **Sample Geometry:**
  Each signal window contains $N = 1024$ samples across 2 channels:
  $$\mathbf{X} \in \mathbb{R}^{2 \times 1024}$$
  Channel 0: In-phase component $I(t)$
  Channel 1: Quadrature component $Q(t)$
- **Channel Impairments Present:**
  Carrier frequency offset (CFO), sample clock offset (SCO), multi-path delay spread, and Additive White Gaussian Noise (AWGN) across SNR values from $-20\text{ dB}$ up to $+30\text{ dB}$.

---

## 5. Security & Authentication Contract

The API-based simulation uses the production security model:
1. **API Key Generation:** Cryptographically random, URL-safe 256-bit token (`secrets.token_urlsafe(32)`).
2. **Transmission:** Supplied via the `X-API-Key` HTTP header.
3. **Verification:** Constant-time comparison (`hmac.compare_digest`) against configured hashes in memory.
4. **Safety Isolation:** Fails closed on missing or mismatched keys with generic HTTP 401 error payloads to prevent enumeration or side-channel leakage.

---

## 6. Architectural Conclusion

The API simulation feature provides complete functional parity with physical hardware from the inference engine's perspective. The downstream layers (validation, model backbones, open-set rejection, and result packaging) execute identical code paths regardless of whether the I/Q samples originated from an antenna or the RadioML 2018 dataset stream.

