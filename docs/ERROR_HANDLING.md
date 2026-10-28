# Error Handling

## 1. Principles

1. **One envelope, always.** Every error, including 404/405 and unexpected crashes, uses the same JSON shape.
2. **Fail closed.** Unknown state means deny: no keys configured means 401, uncalibrated detector means refuse to start, NaN score means `unknown`.
3. **Never leak internals.** Anything not explicitly marked `expose_message` is replaced by a fixed generic message. Stack traces, paths and backend exception text go to logs only.
4. **Wrong is worse than failed.** A model that returns NaN or the wrong number of logits produces a 500, not a guess.
5. **One catch-all.** The middleware converts any unhandled exception to a generic 500 so a single bad request cannot crash or hang the worker.
6. **Correlate everything.** Each response has `X-Request-ID`; each log line for that request carries it.

## 2. Envelope

```json
{ "error": { "code": "validation_error", "message": "…", "request_id": "9f2c…", "details": { "got_shape": [3, 128] } } }
```

`details` appears only for client errors where it is safe and useful (for example the shape that was received, never the data).

## 3. Error catalogue

| `code` | HTTP | Raised when | Message shown to client | Retry? |
|---|---|---|---|---|
| `bad_request` | 400 | Invalid JSON, invalid `Content-Length` | Specific, safe text | No, fix the request |
| `unauthorized` | 401 | Missing or wrong API key | Generic, identical for both | No |
| `payload_too_large` | 413 | Body exceeds the configured cap (header or streamed) | States the limit | No |
| `unsupported_media_type` | 415 | `Content-Type` is not `application/json` | Fixed | No |
| `validation_error` | 422 | Bad shape, length, type, NaN/Inf, zero power, unknown field | Specific, safe text | No |
| `rate_limited` | 429 | Per-IP or per-key bucket empty | Fixed; `Retry-After` header | Yes, after `Retry-After` |
| `internal_error` | 500 | Any unhandled exception | Generic | Maybe, with backoff |
| `inference_error` | 500 | Backbone raised, or returned wrong shape or non-finite values | Generic | Maybe, with backoff |
| `config_error` | 500 | Invalid settings or detector state at runtime | Generic | No, operator action |
| `security_error` | 500 | Model path or hash check failed | Generic | No, operator action |
| `model_unavailable` | 503 | `/readyz` when no model is loaded | Generic | Yes, after readiness |
| `inference_timeout` | 504 | Inference exceeded `RFMC_INFERENCE_TIMEOUT_S` | Generic | Yes, with backoff and jitter |
| `not_found` | 404 | Unknown route | "Not found." | No |
| `method_not_allowed` | 405 | Wrong HTTP method | "Method not allowed." | No |
| `http_error` | other | Other framework HTTP errors | "Request failed." | Depends on status |

Startup failures (`ConfigError`, `SecurityError`, `ModelUnavailableError`) abort the process before it serves traffic. Settings are validated before any model I/O.

## 4. Request pipeline and where each error is raised

```
middleware: request id → Content-Length check (400/413) → [route] → catch-all (500) → headers
route: per-IP limit (429) → auth (401) → per-key limit (429) → content-type (415)
       → streamed body cap (413) → JSON parse (400) → validate (422) → inference (500/504)
```

Auth runs before the body is parsed so unauthenticated callers cannot make the server parse large payloads.

## 5. Logging

| Event | Level | Contains | Never contains |
|---|---|---|---|
| Successful classify | INFO | request id, key fingerprint, N, unknown flag, ms | API key, samples |
| Auth failure | WARNING | request id, sanitised client IP | the attempted key |
| 5xx | ERROR + traceback | request id, error code | included in logs only, never in the response |
| Unhandled exception | ERROR + traceback | request id | n/a |

Operators find the cause of a client's error by searching logs for the `request_id` in the response.

## 6. Client guidance

- Honour `Retry-After` on 429.
- Retry 500/503/504 with exponential backoff and jitter, capped at a few attempts. Classification is idempotent, so retries are safe.
- Never retry 4xx other than 429 without changing the request.
- Quote the `request_id` when reporting a problem.

## 7. Not covered yet

Metrics and alerting (error-rate, p95 latency, 429 rate), a circuit breaker around a failing backbone, and per-error-code dashboards. Add before a high-availability deployment.
