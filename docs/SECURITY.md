# Security

## 1. Scope and trust boundaries

```
Internet ─▶ [TLS proxy / gateway] ─▶ rfmc API ─▶ model file (disk)
 untrusted        semi-trusted          this repo      trusted only after hash check
```

Untrusted: every request byte (headers, body, JSON content), client-supplied request ids, model files until verified.
Trusted: server configuration and environment, the process's own code.

## 2. Threat model

| # | Threat | Control | Test |
|---|---|---|---|
| T1 | Unauthenticated use | `X-API-Key` required; **fails closed** when no keys are configured | `test_security.py::test_fails_closed_when_no_keys_configured` |
| T2 | Key guessing via timing | SHA-256 digests compared with `hmac.compare_digest` against **every** key (no early exit) | `test_verify_api_key_compares_every_key` |
| T3 | Key enumeration via error differences | Missing and wrong key return identical bodies | `test_missing_and_wrong_key_are_indistinguishable` |
| T4 | Brute force and abuse | Per-IP limiter before auth, per-key limiter after; bounded memory; `Retry-After` | `test_rate_limit_over_http`, `test_ip_rate_limit_applies_before_auth`, `test_rate_limiter_memory_is_bounded` |
| T5 | Memory/CPU exhaustion by large bodies | `Content-Length` check plus streaming byte cap for chunked uploads; N capped at 4096 | `test_oversize_body_by_content_length`, `test_oversize_chunked_body_is_cut_off` |
| T6 | JSON nesting bomb | Parser `RecursionError` mapped to 400 | `test_json_nesting_bomb_is_400_not_500` |
| T7 | Malformed numerics (NaN/Inf/huge/strings/bools) reaching the model | Strict dtype, shape, finiteness and magnitude validation; power normalisation | `test_validation.py` |
| T8 | Information leakage in errors | Fixed generic messages for 5xx and auth; tracebacks only in logs | `test_errors.py::test_unhandled_exception_returns_generic_500…` |
| T9 | Log injection and secret leakage | Control characters stripped; request id validated; keys logged only as 12-char hash prefix; payloads never logged | `test_sanitize_for_log_…`, `test_secrets_and_payloads_never_logged` |
| T10 | Malicious or tampered model file | Path confined to an allowed directory (symlinks resolved), suffix allow-list, SHA-256 pin, `torch.load(weights_only=True)` | `test_model_path_confinement`, `test_sha256_verification`, static scan gate |
| T11 | Dangerous code constructs introduced later | AST gate forbids `pickle`, `marshal`, `subprocess`, `eval`, `exec`, `os.system`, `yaml.load`, bare `except`, unsafe `torch.load` | `test_package_has_no_dangerous_constructs` |
| T12 | Insecure production config | `Settings` refuses to start: production needs keys >= 24 chars, valid limits and timeouts | `test_insecure_or_invalid_settings_refuse_to_start` |
| T13 | Recon via API docs | `/docs` and `/openapi.json` disabled in production | `test_production_hides_docs_and_sets_hsts` |
| T14 | Browser-side attacks on responses | `nosniff`, `X-Frame-Options: DENY`, `no-store`, CSP `default-src 'none'`, HSTS in production | `test_security_headers_on_success_and_error` |
| T15 | Vulnerable or drifting dependencies | All requirements pinned (`==`); gate enforces it | `test_all_dependencies_are_pinned` |
| T16 | Committed secrets | Pattern scan and `.env` ban | `test_no_secrets_committed` |
| T17 | Hanging inference | Hard timeout, returns 504 | `test_inference_timeout_returns_504` |
| T18 | Silent wrong answers from a broken model | Output guards: shape and finiteness checked; NaN open-set scores resolve to `unknown` | `test_bad_backbone_output_rejected`, `test_nan_scores_fail_safe_to_unknown` |

## 3. Known limitations (be honest about these)

1. **Adversarial ML.** Small crafted perturbations of I/Q samples can flip CNN modulation predictions, and open-set scores can be fooled too. Treat output as advisory, not as proof. No control here addresses this.
2. **Training-data poisoning and licence.** Datasets are not verified by this repo. Check RadioML licence terms before commercial use.
3. **Static API keys.** No scopes, no rotation endpoint, no per-key revocation without a restart. Fine for a service behind a gateway; use OAuth/mTLS for a multi-tenant product.
4. **In-process rate limiting.** State is per process. With several replicas or workers the effective limit multiplies. Enforce limits at the gateway as well.
5. **No TLS in the app.** Terminate TLS at a proxy. HSTS is emitted in production, so the proxy must serve HTTPS.
6. **Thread-pool timeout.** A timed-out inference keeps running in its worker thread until it finishes. Bound this with a small worker pool and process-level limits.
7. **The torch loader path is only exercised when torch is installed.** `test_torch_model_roundtrip` is skipped otherwise.

## 4. Deployment checklist

- [ ] `RFMC_ENV=production`; keys come from a secret manager (generate with `python -c "import secrets;print(secrets.token_urlsafe(32))"`), never from the repo or image.
- [ ] TLS at the proxy; proxy also enforces body size and rate limits.
- [ ] Run as a non-root user, read-only root filesystem, model directory mounted read-only.
- [ ] `RFMC_MODEL_SHA256` set from the build pipeline; model artefact stored in access-controlled storage.
- [ ] `pip-audit` (or equivalent) in CI against the pinned requirements.
- [ ] Logs shipped to a store with access control; verify no payloads or keys appear.
- [ ] `pytest -m production` green on the exact image being deployed, including the model acceptance gate with real evaluation data.
- [ ] Legal review done for any over-the-air capture; receive-only hardware.

## 5. Reporting

Report vulnerabilities privately to the project owner; do not open public issues for security bugs.
