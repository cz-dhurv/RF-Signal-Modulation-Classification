# Sample Simulation Project Index (2018API)

## 1. Overview

The **Sample Simulation (2018API)** subsystem provides an end-to-end software simulation bridge for the RF Signal Modulation Classification platform. Because physical Software-Defined Radio (SDR) hardware (e.g., RTL-SDR, HackRF One, ADALM-Pluto) is not always available in early development and test environments, this subsystem acts as a **Hardware Filler**.

Instead of capturing live electromagnetic signals over the air, the hardware filler streams authentic I/Q signal data at **1 sample frame per second (1 data/sec)** from the **RadioML 2018.01A** dataset to the hardened REST API (`/v1/classify`), authenticated via an **API key**.

---

## 2. Dual-Feature Architecture: Hardware vs. API Calling

The RF Modulation Classification platform is architected around two operational ingress modes:

```
               +-------------------------------------------------------+
               |                  RF Ingress Sources                   |
               +-------------------------------------------------------+
                               |                               |
             [Feature 1: Planned Hardware]        [Feature 2: Built & Active]
             Physical SDR Receiver (Live RF)      API Simulation Hardware Filler
             (RTL-SDR / HackRF / PlutoSDR)        (RadioML 2018.01A @ 1 data/sec)
                               |                               |
                               | (Raw I/Q Stream)              | (HTTP POST /v1/classify)
                               |                               |   with X-API-Key
                               v                               v
               +-------------------------------------------------------+
               |             FastAPI Classification Service             |
               |       (rfmc/api.py, Token-Bucket Rate Limiter)        |
               +-------------------------------------------------------+
                               |
                               v
               +-------------------------------------------------------+
               |        Safe Model Inference & Open-Set Engine         |
               |          (RFNet / CLDNN / OpenSetDetector)            |
               +-------------------------------------------------------+
                               |
                               v
               +-------------------------------------------------------+
               | Real-Time Prediction: Modulation / Confidence / SNR   |
               +-------------------------------------------------------+
```

| Feature | Ingress Mechanism | Status | Data Source | Ingestion Rate | Authentication |
|---|---|---|---|---|---|
| **Feature 1: Live Hardware Capture** | USB / PCIe SDR Driver (pyrtlsdr, SoapySDR) | Planned (when hardware available) | Over-the-air RF spectrum (e.g. 2.4 GHz ISM, VHF/UHF) | Variable / Burst (2 MSps - 20 MSps) | Local IPC / Internal Bus |
| **Feature 2: API Calling Simulation** | HTTP POST Client (`simulate_hardware_filler.py`) | **Built & Active (Branch 2018API)** | RadioML 2018.01A (24 modulations, 1024 I/Q samples) | Clocked 1 data frame/second (1 Hz) | Secret API Key (`X-API-Key`) |

---

## 3. Directory & File Structure

```
RF-Signal-Modulation-Classification/
├── docs/
│   ├── sample_simulation/
│   │   ├── indexproject.md        # This master index
│   │   ├── Project_analysis.md    # Architectural analysis & hardware-filler mechanics
│   │   ├── error_handling.md      # Simulation-specific error handling & edge cases
│   │   ├── status.md              # Milestones, current progress, and release roadmap
│   │   └── Test_plan.md           # Verification suite for 1 data/sec API streaming
├── scripts/
│   ├── download_rml2018.py        # RadioML 2018 dataset fetcher & sample generator
│   ├── generate_api_key.py        # Cryptographic API key generator & environment config
│   ├── simulate_hardware_filler.py # 1 data/sec simulated SDR hardware stream runner
│   └── run_2018_api_server.py     # Server bootstrap with pre-configured 2018 schema
├── data/
│   └── rml2018.npz                # Converted/sampled 2018 dataset storage
```

---

## 4. RadioML 2018.01A Dataset Integration

The 2018 dataset (`GOLD_XYZ_OSC.0001_1024.hdf5`) introduces realistic wireless channel phenomena across 24 modulation formats:

1. **Digital Modulations (18):**
   - PSK: `BPSK`, `QPSK`, `8PSK`, `16PSK`, `32PSK`, `OQPSK`
   - QAM: `16QAM`, `32QAM`, `64QAM`, `128QAM`, `256QAM`
   - APSK: `16APSK`, `32APSK`, `64APSK`, `128APSK`
   - ASK / FSK: `OOK`, `4ASK`, `8ASK`, `GMSK`
2. **Analog Modulations (6):**
   - AM: `AM-SSB-WC`, `AM-SSB-SC`, `AM-DSB-WC`, `AM-DSB-SC`
   - FM: `FM`
3. **Tensor Geometry:**
   - Input shape: `(2, 1024)` float32 representing In-phase (I) and Quadrature (Q) channels.
   - Power-normalised: Unit average power $\frac{1}{N}\sum (I_k^2 + Q_k^2) = 1.0$.
   - SNR coverage: $-20\text{ dB}$ to $+30\text{ dB}$ in $2\text{ dB}$ increments.

---

## 5. Quick Start Guide

### Step 1: Generate API Key and Environment Config
```bash
python scripts/generate_api_key.py --env
```

### Step 2: Acquire / Generate RadioML 2018 Dataset Samples
```bash
python scripts/download_rml2018.py --sample --samples-per-class 100
```

### Step 3: Launch the RFMC Inference Server
```bash
python scripts/run_2018_api_server.py --port 8000
```

### Step 4: Run the 1 Data/Sec Hardware Filler Simulation
```bash
python scripts/simulate_hardware_filler.py --url http://127.0.0.1:8000/v1/classify --rate 1.0
```

