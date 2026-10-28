"""Runtime settings with fail-fast validation (invalid config never starts the server)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .errors import ConfigError

ENVIRONMENTS = ("development", "test", "production")
MIN_PROD_KEY_LEN = 24


@dataclass(frozen=True)
class Settings:
    env: str = "development"
    api_keys: frozenset = frozenset()
    max_body_bytes: int = 262_144
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 10
    ip_rate_limit_per_min: int = 600
    inference_timeout_s: float = 5.0
    min_len: int = 128
    max_len: int = 4096

    def __post_init__(self) -> None:
        self.validate()

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def validate(self) -> None:
        if self.env not in ENVIRONMENTS:
            raise ConfigError(f"RFMC_ENV must be one of {ENVIRONMENTS}")
        if not 1024 <= self.max_body_bytes <= 10 * 1024 * 1024:
            raise ConfigError("max_body_bytes must be between 1 KiB and 10 MiB")
        for name in ("rate_limit_per_min", "rate_limit_burst", "ip_rate_limit_per_min"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")
        if not 0 < self.inference_timeout_s <= 60:
            raise ConfigError("inference_timeout_s must be in (0, 60]")
        if not 16 <= self.min_len <= self.max_len <= 65_536:
            raise ConfigError("require 16 <= min_len <= max_len <= 65536")
        if self.is_production:
            if not self.api_keys:
                raise ConfigError("production requires at least one API key")
            for key in self.api_keys:
                if len(key) < MIN_PROD_KEY_LEN or any(c.isspace() for c in key):
                    raise ConfigError(
                        f"production API keys must be >= {MIN_PROD_KEY_LEN} chars, no whitespace"
                    )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ

        def num(name: str, default, cast):
            raw = env.get(name)
            if raw is None or raw == "":
                return default
            try:
                return cast(raw)
            except ValueError as exc:
                raise ConfigError(f"{name} is not a valid {cast.__name__}") from exc

        keys = frozenset(k.strip() for k in env.get("RFMC_API_KEYS", "").split(",") if k.strip())
        return cls(
            env=env.get("RFMC_ENV", "development"),
            api_keys=keys,
            max_body_bytes=num("RFMC_MAX_BODY_BYTES", 262_144, int),
            rate_limit_per_min=num("RFMC_RATE_LIMIT_PER_MIN", 60, int),
            rate_limit_burst=num("RFMC_RATE_LIMIT_BURST", 10, int),
            ip_rate_limit_per_min=num("RFMC_IP_RATE_LIMIT_PER_MIN", 600, int),
            inference_timeout_s=num("RFMC_INFERENCE_TIMEOUT_S", 5.0, float),
        )
