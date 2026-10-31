"""Charged-deadline accounting for one run directory.

A deadline timeout that this process aborted and then settled at the reserved
worst case is a charged timeout. It is not an unknown outcome. Counts are
rebuilt from the ledger and the diagnostics beside it, so a resume sees the
timeouts already written in this epoch. ``progress/timeout_counts.json`` is the
snapshot an operator can read. ``progress/provider_outcomes.jsonl`` is the
append-only log of outcomes this process finished.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .freeze import CHARGED_TIMEOUT_EPOCH_CNY, CHARGED_TIMEOUT_RATE, CHARGED_TIMEOUT_WINDOW
from .prices import MICROCNY_PER_CNY


CHARGED_TIMEOUT_CAUSES = frozenset({"deadline_timeout", "timeout"})
SNAPSHOT_SCHEMA = "alo-charged-timeout-counts-v1"


def _in_epoch(request_id, protocol_epoch):
    if not isinstance(request_id, str) or not request_id.startswith("req-"):
        return False
    if protocol_epoch:
        return request_id.startswith(f"req-e{protocol_epoch}-")
    rest = request_id[4:]
    if not rest.startswith("e"):
        return True
    index = 1
    while index < len(rest) and rest[index].isdigit():
        index += 1
    return not (index > 1 and rest[index:index + 1] == "-")


def _read_diagnostic(directory, request_id):
    path = directory / f"{request_id}.json"
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return body if isinstance(body, dict) else None


def _charged_timeout(row, diagnostic):
    if row["status"] != "settled" or diagnostic is None:
        return False
    if diagnostic.get("cause") not in CHARGED_TIMEOUT_CAUSES:
        return False
    return diagnostic.get("settled") is True


def assess_charged_timeouts(ledger, *, protocol_epoch):
    """Rate and epoch CNY for charged deadline timeouts. This does not halt."""
    directory = Path(ledger.path).parent / "diagnostics"
    by_model = {}
    charged_microcny = 0
    charged_count = 0
    for row in ledger.request_rows():
        if not _in_epoch(row["id"], protocol_epoch):
            continue
        diagnostic = _read_diagnostic(directory, row["id"])
        bucket = by_model.setdefault(row["model"], {"rows": []})
        timeout = _charged_timeout(row, diagnostic)
        bucket["rows"].append(timeout)
        if timeout:
            charged_count += 1
            charged_microcny += row["held_microcny"]
    models = {}
    rate_reason = None
    for model, bucket in sorted(by_model.items()):
        window_rows = bucket["rows"][-CHARGED_TIMEOUT_WINDOW:]
        timeouts = sum(1 for item in window_rows if item)
        full = len(window_rows) >= CHARGED_TIMEOUT_WINDOW
        rate = (timeouts / len(window_rows)) if window_rows else 0.0
        models[model] = {
            "requests_in_epoch": len(bucket["rows"]),
            "window": len(window_rows),
            "window_full": full,
            "charged_timeouts_in_window": timeouts,
            "rate": round(rate, 4),
        }
        if full and timeouts > len(window_rows) * CHARGED_TIMEOUT_RATE and rate_reason is None:
            rate_reason = "charged_timeout_rate"
    cap_microcny = CHARGED_TIMEOUT_EPOCH_CNY * MICROCNY_PER_CNY
    cny_reason = "charged_timeout_cny" if charged_microcny > cap_microcny else None
    return {
        "schema": SNAPSHOT_SCHEMA,
        "epoch": protocol_epoch,
        "window": CHARGED_TIMEOUT_WINDOW,
        "rate_limit": CHARGED_TIMEOUT_RATE,
        "rate_applies_after_requests": CHARGED_TIMEOUT_WINDOW,
        "epoch_cny_cap": CHARGED_TIMEOUT_EPOCH_CNY,
        "charged_timeouts": charged_count,
        "charged_timeout_microcny": charged_microcny,
        "halt_reason": rate_reason or cny_reason,
        "by_model": models,
    }


def publish_timeout_counts(run_dir, ledger, *, protocol_epoch):
    """Rewrite the snapshot from the ledger and diagnostics."""
    body = assess_charged_timeouts(ledger, protocol_epoch=protocol_epoch)
    path = Path(run_dir) / "progress" / "timeout_counts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    return body


def append_provider_outcome(ledger, *, epoch, model, request_id, outcome, charge_microcny):
    """Append one finished provider outcome. Failures to write are ignored by the caller."""
    path = Path(ledger.path).parent / "progress" / "provider_outcomes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({
        "epoch": epoch,
        "model": model,
        "request_id": request_id,
        "outcome": outcome,
        "charge_microcny": charge_microcny,
    }, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
