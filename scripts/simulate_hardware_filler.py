#!/usr/bin/env python3
"""Simulated SDR Hardware Filler: Streams 2018 dataset frames at 1 data/sec via API.

When physical SDR hardware (RTL-SDR / HackRF) is unavailable, this client acts as a
hardware filler surrogate. It reads authentic RadioML 2018.01A I/Q frames from data/rml2018.npz
and feeds them to the RFMC API (/v1/classify) at a precise rate of 1 data frame per second.

Usage:
    # Basic continuous streaming at 1 data per second
    python scripts/simulate_hardware_filler.py

    # Stream 10 frames with explicit API key and custom endpoint
    python scripts/simulate_hardware_filler.py --count 10 --api-key <YOUR_KEY> --url http://127.0.0.1:8000/v1/classify

    # Log results to a JSON lines file
    python scripts/simulate_hardware_filler.py --save-log logs/simulation_stream.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_env_key(env_path: Path) -> str | None:
    """Extract first API key from .env file."""
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("RFMC_API_KEYS="):
                val = line.split("=", 1)[1].strip()
                keys = [k.strip() for k in val.split(",") if k.strip()]
                if keys:
                    return keys[0]
    return None


def ensure_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load dataset, generating sample dataset if absent."""
    if not dataset_path.is_file():
        print(f"[WARNING] Dataset file not found at {dataset_path}!")
        print("[INFO] Invoking sample generator to produce 24-class RadioML 2018 dataset...")
        from scripts.download_rml2018 import generate_rml2018_sample
        generate_rml2018_sample(dataset_path, samples_per_class=50)

    print(f"[INFO] Loading RadioML 2018 dataset from {dataset_path} ...")
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]                    # (M, 2, 1024)
    y = data["y"]                    # (M,)
    snr = data["snr"]                # (M,)
    class_names = [str(c) for c in data["class_names"]]

    print(f"[INFO] Loaded {len(X)} frames across {len(class_names)} modulations. SNR range: [{snr.min()}, {snr.max()}] dB")
    return X, y, snr, class_names


def stream_simulation(
    url: str,
    api_key: str,
    dataset_path: Path,
    rate_hz: float = 1.0,
    count: int = 0,
    save_log: Path | None = None,
) -> None:
    """Stream I/Q frames at specified rate to API endpoint."""
    X, y, snr, class_names = ensure_dataset(dataset_path)

    interval_s = 1.0 / rate_hz if rate_hz > 0 else 1.0
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    key_fp = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "****"

    print("\n" + "=" * 78)
    print("      RF Modulation Classification - Virtual SDR Hardware Filler (2018API)      ")
    print("=" * 78)
    print(f"Target API:       {url}")
    print(f"Authentication:   API-Key active (fingerprint: {key_fp})")
    print(f"Streaming Rate:   {rate_hz:.2f} data frame/sec (1 frame every {interval_s:.2f}s)")
    print(f"Frames to stream: {'Continuous (Ctrl+C to stop)' if count <= 0 else str(count) + ' frames'}")
    print("=" * 78 + "\n")
    print(f"{'TIME':<10} | {'TRUE MOD':<10} | {'SNR':<7} | {'PREDICTED':<12} | {'CONF':<6} | {'STATUS':<9} | {'RTT':<6}")
    print("-" * 78)

    log_fh = None
    if save_log:
        save_log.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(save_log, "a", encoding="utf-8")

    session = requests.Session()
    total_sent = 0
    total_success = 0
    total_rtt_ms = 0.0

    try:
        frame_idx = 0
        n_total_frames = len(X)

        while True:
            if count > 0 and total_sent >= count:
                break

            # Current sample
            sample_iq = X[frame_idx % n_total_frames]
            true_idx = int(y[frame_idx % n_total_frames])
            true_mod = class_names[true_idx]
            true_snr = int(snr[frame_idx % n_total_frames])
            frame_idx += 1

            payload = {
                "iq": sample_iq.tolist(),
                "sample_rate_hz": 2000000,
            }

            t_start = time.perf_counter()
            now_str = datetime.datetime.now().strftime("%H:%M:%S")

            try:
                resp = session.post(url, headers=headers, json=payload, timeout=5.0)
                rtt_ms = (time.perf_counter() - t_start) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    # Support both flat API envelope and nested prediction structure
                    pred_label = data.get("modulation") or data.get("prediction", {}).get("label", "unknown")
                    conf = float(data.get("confidence") if "confidence" in data else data.get("prediction", {}).get("confidence", 0.0))
                    is_unknown = bool(data.get("unknown") if "unknown" in data else data.get("prediction", {}).get("unknown", False))
                    status_tag = "UNKNOWN" if is_unknown else "CONFIRMED"

                    total_success += 1
                    total_rtt_ms += rtt_ms

                    print(f"{now_str:<10} | {true_mod:<10} | {true_snr:+3d} dB | {pred_label:<12} | {conf:5.2f} | {status_tag:<9} | {rtt_ms:4.1f}ms")

                    if log_fh:
                        record = {
                            "timestamp": now_str,
                            "true_mod": true_mod,
                            "true_snr": true_snr,
                            "modulation": pred_label,
                            "confidence": conf,
                            "unknown": is_unknown,
                            "rtt_ms": rtt_ms,
                            "request_id": data.get("request_id"),
                        }
                        log_fh.write(json.dumps(record) + "\n")
                        log_fh.flush()

                elif resp.status_code == 401:
                    print(f"[AUTH ERROR 401] Invalid or rejected API key: {api_key}. Stopping.")
                    break
                elif resp.status_code == 429:
                    print(f"[RATE LIMIT 429] Server rate limit exceeded. Backing off for 1.5s...")
                    time.sleep(1.5)
                else:
                    print(f"[ERROR {resp.status_code}] {resp.text}")

            except requests.exceptions.ConnectionError:
                print(f"{now_str:<10} | [DISCONNECTED] Could not reach {url}. Retrying in {interval_s:.1f}s ...")
            except requests.exceptions.Timeout:
                print(f"{now_str:<10} | [TIMEOUT] Inference request exceeded timeout.")

            total_sent += 1

            # Exact clocking at 1 data per second
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.0, interval_s - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user (Ctrl+C).")
    finally:
        if log_fh:
            log_fh.close()

    if total_sent > 0:
        avg_rtt = (total_rtt_ms / total_success) if total_success > 0 else 0.0
        print("-" * 78)
        print(f"[SUMMARY] Total frames sent: {total_sent} | Successful: {total_success} | Avg Latency: {avg_rtt:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual SDR Hardware Filler Simulation (RadioML 2018)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1/classify", help="Classifier API URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key for X-API-Key header")
    parser.add_argument("--rate", type=float, default=1.0, help="Frames per second (default: 1.0 = 1 data/sec)")
    parser.add_argument("--count", type=int, default=0, help="Number of frames to send (0 = continuous)")
    parser.add_argument("--data", type=Path, default=Path("data/rml2018.npz"), help="Path to 2018 dataset .npz")
    parser.add_argument("--save-log", type=Path, default=None, help="Optional JSONL file to save inference logs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Resolve API Key
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("RFMC_API_KEY") or os.environ.get("RFMC_API_KEYS")
    if not api_key:
        api_key = load_env_key(project_root / ".env")
    if not api_key:
        print("[ERROR] No API key found! Generate one using: python scripts/generate_api_key.py --env")
        sys.exit(1)

    data_path = args.data if args.data.is_absolute() else (project_root / args.data)

    stream_simulation(
        url=args.url,
        api_key=api_key,
        dataset_path=data_path,
        rate_hz=args.rate,
        count=args.count,
        save_log=args.save_log,
    )


if __name__ == "__main__":
    main()
