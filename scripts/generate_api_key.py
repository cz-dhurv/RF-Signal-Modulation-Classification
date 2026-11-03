#!/usr/bin/env python3
"""Generate a cryptographically secure API key for the RFMC service and simulation.

Usage:
    python scripts/generate_api_key.py
    python scripts/generate_api_key.py --env
    python scripts/generate_api_key.py --key-length 32
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys
from pathlib import Path


def generate_key(n_bytes: int = 32) -> str:
    """Generate a URL-safe cryptographically secure random key."""
    return secrets.token_urlsafe(n_bytes)


def fingerprint(key: str) -> str:
    """Compute a non-reversible key fingerprint for safe logging."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def save_to_env(key: str, env_path: Path) -> None:
    """Save or update RFMC_API_KEYS in a .env file."""
    lines: list[str] = []
    found = False
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("RFMC_API_KEYS="):
                lines.append(f"RFMC_API_KEYS={key}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"RFMC_API_KEYS={key}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] API key written to {env_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate secure API key for RFMC service")
    parser.add_argument("--env", action="store_true", help="Write key to .env in project root")
    parser.add_argument("--key-length", type=int, default=32, help="Entropy byte length (default: 32)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Optional file path to write key to")
    args = parser.parse_args()

    key = generate_key(args.key_length)
    fp = fingerprint(key)

    print("=" * 60)
    print("      RF Modulation Classification - API Key Generator      ")
    print("=" * 60)
    print(f"Key:         {key}")
    print(f"Length:      {len(key)} chars")
    print(f"Fingerprint: {fp}...")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent

    if args.env:
        save_to_env(key, project_root / ".env")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(key + "\n", encoding="utf-8")
        print(f"[INFO] Key saved to {args.output}")

    print("\nHow to use with simulation client:")
    print(f"  python scripts/simulate_hardware_filler.py --api-key {key}")
    print("\nHow to use with curl:")
    print(f'  curl -X POST http://127.0.0.1:8000/v1/classify -H "X-API-Key: {key}" ...')


if __name__ == "__main__":
    main()

