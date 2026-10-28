"""Typed error hierarchy and the public error envelope.

Rule: anything that is not explicitly marked ``expose_message`` is replaced by a
fixed, generic message before it reaches a client. Internal detail goes to logs only.
"""

from __future__ import annotations

from typing import Any


class RFMCError(Exception):
    code = "internal_error"
    http_status = 500
    public_message = "An internal error occurred."
    expose_message = False  # may the constructor message be shown to clients?

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None):
        super().__init__(message or self.public_message)
        self.message = message or self.public_message
        self.details = details or {}

    def to_public(self) -> tuple[str, dict[str, Any]]:
        if self.expose_message:
            return self.message, self.details
        return self.public_message, {}


class BadRequestError(RFMCError):
    code = "bad_request"
    http_status = 400
    public_message = "Malformed request."
    expose_message = True


class UnauthorizedError(RFMCError):
    code = "unauthorized"
    http_status = 401
    public_message = "Missing or invalid API key."


class PayloadTooLargeError(RFMCError):
    code = "payload_too_large"
    http_status = 413
    public_message = "Request body is too large."
    expose_message = True


class UnsupportedMediaTypeError(RFMCError):
    code = "unsupported_media_type"
    http_status = 415
    public_message = "Content-Type must be application/json."
    expose_message = True


class ValidationError(RFMCError):
    code = "validation_error"
    http_status = 422
    public_message = "Request validation failed."
    expose_message = True


class RateLimitedError(RFMCError):
    code = "rate_limited"
    http_status = 429
    public_message = "Too many requests."
    expose_message = True

    def __init__(self, message: str | None = None, *, retry_after: int = 1, **kw: Any):
        super().__init__(message, **kw)
        self.retry_after = max(1, int(retry_after))


class InternalError(RFMCError):
    pass


class InferenceError(RFMCError):
    code = "inference_error"
    public_message = "Inference failed."


class ModelUnavailableError(RFMCError):
    code = "model_unavailable"
    http_status = 503
    public_message = "Model is not available."


class InferenceTimeoutError(RFMCError):
    code = "inference_timeout"
    http_status = 504
    public_message = "Inference timed out."


class ConfigError(RFMCError):
    code = "config_error"
    public_message = "Server misconfiguration."


class SecurityError(RFMCError):
    code = "security_error"
    public_message = "Security check failed."


def error_body(
    code: str, message: str, request_id: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if details:
        body["details"] = details
    return {"error": body}


def error_body_for(exc: RFMCError, request_id: str) -> dict[str, Any]:
    message, details = exc.to_public()
    return error_body(exc.code, message, request_id, details)


def code_for_http_status(status: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        404: "not_found",
        405: "method_not_allowed",
        413: "payload_too_large",
        415: "unsupported_media_type",
        429: "rate_limited",
    }.get(status, "http_error")
