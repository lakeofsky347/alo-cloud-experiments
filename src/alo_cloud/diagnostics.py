"""Redacted provider diagnostics. These files live in the run directory, not the repo.

A diagnostic is written for every outcome that is not a non-empty settled
completion: an empty billed completion, a vendor error settled at zero, and an
unresolved unknown hold. Auth headers are never stored. Credential values are
replaced even if a vendor echoes them in a message.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


DIAGNOSTIC_SCHEMA = "alo-provider-diagnostic-v1"
CAUSES = frozenset({
    "empty_completion",
    "vendor_error",
    "timeout",
    "connection_reset",
    "unparseable_success",
    "transport_error",
    "charge_rejected",
})
_SECRET_FIELDS = frozenset({
    "authorization", "api-key", "api_key", "apikey", "access_token", "refresh_token",
    "secret", "password",
})
_STRING_LIMIT = 4000


def diagnostics_directory(ledger):
    """``$ALO_RUN_DIR/diagnostics`` when that variable is set, else beside the ledger."""
    env = os.environ.get("ALO_RUN_DIR", "").strip()
    root = Path(env) if env else Path(ledger.path).parent
    path = root / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _secrets():
    found = []
    for name in ("DEEPSEEK_API_KEY", "MIMO_API_KEY", "ZAI_API_KEY"):
        value = os.environ.get(name)
        if isinstance(value, str) and value:
            found.append(value)
    return found


def redact(value, secrets):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_FIELDS:
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = redact(item, secrets)
        return cleaned
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret and secret in value:
                value = value.replace(secret, "[redacted]")
        if len(value) > _STRING_LIMIT:
            return value[:_STRING_LIMIT] + "…[truncated]"
        return value
    if isinstance(value, bool) or value is None or type(value) is int:
        return value
    if type(value) is float:
        return value
    return "[unsupported]"


def write_diagnostic(ledger, request_id, *, cause, http_status, vendor_code, vendor_message,
                     response, finish_reason, latency_ms, settled, model, unit_id):
    """Persist one redacted JSON file named ``<request_id>.json``."""
    if cause not in CAUSES:
        raise ValueError("diagnostic_cause_rejected")
    if not isinstance(request_id, str) or "/" in request_id or "\\" in request_id or not request_id:
        raise ValueError("diagnostic_request_id_rejected")
    secrets = _secrets()
    body = {
        "schema": DIAGNOSTIC_SCHEMA,
        "request_id": request_id,
        "unit_id": unit_id or "",
        "model": model,
        "cause": cause,
        "http_status": http_status if type(http_status) is int else None,
        "vendor_code": redact(vendor_code, secrets) if isinstance(vendor_code, str) else vendor_code,
        "vendor_message": redact(vendor_message, secrets) if isinstance(vendor_message, str) else None,
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        "latency_ms": latency_ms if type(latency_ms) is int else None,
        "settled": bool(settled),
        "response": redact(response, secrets) if isinstance(response, dict) else None,
    }
    path = diagnostics_directory(ledger) / f"{request_id}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    return path
