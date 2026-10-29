"""FastAPI serving layer: auth, rate limits, size caps, uniform errors, security headers."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .errors import (
    BadRequestError,
    InferenceTimeoutError,
    InternalError,
    ModelUnavailableError,
    PayloadTooLargeError,
    RateLimitedError,
    RFMCError,
    UnauthorizedError,
    UnsupportedMediaTypeError,
    code_for_http_status,
    error_body,
    error_body_for,
)
from .scanner import WidebandScanner
from .security import (
    HSTS,
    SECURITY_HEADERS,
    TokenBucketRateLimiter,
    key_fingerprint,
    safe_request_id,
    sanitize_for_log,
    verify_api_key,
)
from .validation import validate_classify_payload, validate_scan_payload

log = logging.getLogger("rfmc.api")


async def _read_body(request: Request, limit: int) -> bytes:
    """Stream the body and abort as soon as it exceeds the limit (covers chunked uploads)."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(f"Request body exceeds {limit} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(service, settings: Settings) -> FastAPI:
    prod = settings.is_production
    app = FastAPI(
        title="RF Modulation Classifier",
        docs_url=None if prod else "/docs",
        redoc_url=None,
        openapi_url=None if prod else "/openapi.json",
    )
    key_limiter = TokenBucketRateLimiter(settings.rate_limit_per_min, settings.rate_limit_burst)
    ip_limiter = TokenBucketRateLimiter(
        settings.ip_rate_limit_per_min, max(settings.rate_limit_burst, 20)
    )

    def _json_error(exc: RFMCError, rid: str) -> JSONResponse:
        resp = JSONResponse(error_body_for(exc, rid), status_code=exc.http_status)
        if isinstance(exc, RateLimitedError):
            resp.headers["Retry-After"] = str(exc.retry_after)
        if isinstance(exc, UnauthorizedError):
            resp.headers["WWW-Authenticate"] = "ApiKey"
        return resp

    @app.middleware("http")
    async def guard(request: Request, call_next):
        rid = safe_request_id(request.headers.get("x-request-id"))
        request.state.request_id = rid
        try:
            declared = request.headers.get("content-length")
            if declared is not None:
                if not declared.isdigit():
                    raise BadRequestError("Invalid Content-Length header.")
                if int(declared) > settings.max_body_bytes:
                    raise PayloadTooLargeError(f"Request body exceeds {settings.max_body_bytes} bytes.")
            response = await call_next(request)
        except RFMCError as exc:
            response = _json_error(exc, rid)
        except Exception:
            log.exception("unhandled error rid=%s", rid)
            response = _json_error(InternalError(), rid)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        if prod:
            response.headers["Strict-Transport-Security"] = HSTS
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(RFMCError)
    async def _rfmc_handler(request: Request, exc: RFMCError):
        rid = getattr(request.state, "request_id", "unknown")
        if exc.http_status >= 500:
            log.error("server error rid=%s code=%s", rid, exc.code, exc_info=exc)
        return _json_error(exc, rid)

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException):
        rid = getattr(request.state, "request_id", "unknown")
        message = {404: "Not found.", 405: "Method not allowed."}.get(exc.status_code, "Request failed.")
        return JSONResponse(
            error_body(code_for_http_status(exc.status_code), message, rid),
            status_code=exc.status_code,
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        if not getattr(service, "ready", False):
            raise ModelUnavailableError()
        return {"status": "ready", "model_version": service.model_version}

    @app.post("/v1/classify")
    async def classify(request: Request):
        rid = request.state.request_id
        ip = request.client.host if request.client else "unknown"
        ip_limiter.check(ip)

        api_key = request.headers.get("x-api-key")
        if not verify_api_key(api_key, settings.api_keys):
            log.warning("auth failed rid=%s ip=%s", rid, sanitize_for_log(ip))
            raise UnauthorizedError()
        fingerprint = key_fingerprint(api_key)
        key_limiter.check(fingerprint)

        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise UnsupportedMediaTypeError()
        body = await _read_body(request, settings.max_body_bytes)
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise BadRequestError("Body is not valid JSON.") from None

        req = validate_classify_payload(payload, min_len=settings.min_len, max_len=settings.max_len)

        started = time.perf_counter()
        try:
            pred = await asyncio.wait_for(
                run_in_threadpool(service.classify, req.iq), timeout=settings.inference_timeout_s
            )
        except asyncio.TimeoutError:
            raise InferenceTimeoutError() from None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        log.info(
            "classify ok rid=%s key=%s n=%d unknown=%s ms=%.1f",
            rid, fingerprint, req.iq.shape[1], pred.unknown, elapsed_ms,
        )
        return {
            "request_id": rid,
            "modulation": pred.label,
            "unknown": pred.unknown,
            "closest_known": pred.closest_known,
            "confidence": pred.confidence,
            "openset_score": pred.openset_score,
            "top_k": [{"label": l, "probability": p} for l, p in pred.top_k],
            "num_samples": int(req.iq.shape[1]),
            "model_version": service.model_version,
        }

    scanner = WidebandScanner(classifier=service)

    @app.post("/v1/scan")
    async def scan(request: Request):
        rid = request.state.request_id
        ip = request.client.host if request.client else "unknown"
        ip_limiter.check(ip)

        api_key = request.headers.get("x-api-key")
        if not verify_api_key(api_key, settings.api_keys):
            log.warning("auth failed rid=%s ip=%s", rid, sanitize_for_log(ip))
            raise UnauthorizedError()
        fingerprint = key_fingerprint(api_key)
        key_limiter.check(fingerprint)

        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise UnsupportedMediaTypeError()
        body = await _read_body(request, settings.max_body_bytes)
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise BadRequestError("Body is not valid JSON.") from None

        req = validate_scan_payload(payload)

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(
                    scanner.scan,
                    req.iq,
                    sample_rate_hz=req.sample_rate_hz,
                    threshold_db=req.threshold_db,
                    min_free_bandwidth_hz=req.min_free_bandwidth_hz,
                ),
                timeout=settings.inference_timeout_s * 2,
            )
        except asyncio.TimeoutError:
            raise InferenceTimeoutError() from None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        log.info(
            "scan ok rid=%s key=%s n=%d channels=%d free=%d ms=%.1f",
            rid, fingerprint, req.iq.shape[1], len(result.channels), len(result.free_bands), elapsed_ms,
        )

        channels_out = []
        for ch in result.channels:
            item = {
                "channel_id": ch.channel_id,
                "center_freq_hz": ch.center_freq_hz,
                "bandwidth_hz": ch.bandwidth_hz,
                "power_db": ch.power_db,
                "snr_db": ch.snr_db,
                "occupied": ch.occupied,
            }
            if ch.prediction:
                item["classification"] = {
                    "modulation": ch.prediction.label,
                    "unknown": ch.prediction.unknown,
                    "closest_known": ch.prediction.closest_known,
                    "confidence": ch.prediction.confidence,
                    "openset_score": ch.prediction.openset_score,
                }
            channels_out.append(item)

        free_out = [
            {
                "start_freq_hz": fb.start_freq_hz,
                "stop_freq_hz": fb.stop_freq_hz,
                "center_freq_hz": fb.center_freq_hz,
                "bandwidth_hz": fb.bandwidth_hz,
                "noise_power_db": fb.noise_power_db,
            }
            for fb in result.free_bands
        ]

        rec_out = None
        if result.recommended_free_band:
            rec_out = {
                "start_freq_hz": result.recommended_free_band.start_freq_hz,
                "stop_freq_hz": result.recommended_free_band.stop_freq_hz,
                "center_freq_hz": result.recommended_free_band.center_freq_hz,
                "bandwidth_hz": result.recommended_free_band.bandwidth_hz,
                "noise_power_db": result.recommended_free_band.noise_power_db,
            }

        return {
            "request_id": rid,
            "sample_rate_hz": result.sample_rate_hz,
            "noise_floor_db": result.noise_floor_db,
            "total_bandwidth_hz": result.total_bandwidth_hz,
            "occupied_bandwidth_hz": result.occupied_bandwidth_hz,
            "spectral_occupancy_ratio": result.spectral_occupancy_ratio,
            "num_channels_detected": len(channels_out),
            "channels": channels_out,
            "free_bands": free_out,
            "recommended_free_band": rec_out,
            "model_version": service.model_version,
        }

    return app
