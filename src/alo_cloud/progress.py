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
from .shardlock import exclusive_lock


CHARGED_TIMEOUT_CAUSES = frozenset({"deadline_timeout", "timeout", "provider_stream_idle"})
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


def _charged_match(row, diagnostic, causes):
    if row["status"] != "settled" or diagnostic is None:
        return False
    if diagnostic.get("cause") not in causes:
        return False
    return diagnostic.get("settled") is True


def _baseline_microcny(value):
    if type(value) is int and value >= 0:
        return value
    return 0


def _outcome_epoch_record(run_dir):
    """Outcome-epoch file. A missing baseline is zero, including old files."""
    path = Path(run_dir) / "progress" / "outcome_epoch.json"
    if not path.is_file():
        return 0, 0, 0
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0
    if not isinstance(value, dict):
        return 0, 0, 0
    epoch = value.get("epoch")
    if type(epoch) is not int or epoch < 0:
        epoch = 0
    return (
        epoch,
        _baseline_microcny(value.get("charged_timeout_baseline_microcny")),
        _baseline_microcny(value.get("charged_unparseable_baseline_microcny")),
    )


def _assess_charged(ledger, *, protocol_epoch, causes, rate_name, cny_name, count_key, micro_key, window_key):
    """Per-model rate and epoch CNY for one settled diagnostic cause.

    The CNY cap is applied to spend since the outcome-epoch baseline. The
    cumulative total stays in ``micro_key``. The rate window is unchanged.
    """
    directory = Path(ledger.path).parent / "diagnostics"
    from .ledger import track_generations
    generations = track_generations(ledger.path.parent)
    by_model = {}
    charged_microcny = 0
    charged_count = 0
    for row in ledger.request_rows():
        if not _in_epoch(row["id"], protocol_epoch):
            continue
        generation = generations.get(row.get("unit_id") or "", 0)
        if generation and f"-pbv{generation}-" not in row["id"]:
            continue
        diagnostic = _read_diagnostic(directory, row["id"])
        bucket = by_model.setdefault(row["model"], {"rows": []})
        matched = _charged_match(row, diagnostic, causes)
        bucket["rows"].append(matched)
        if matched:
            charged_count += 1
            charged_microcny += row["held_microcny"]
    models = {}
    rate_reason = None
    for model, bucket in sorted(by_model.items()):
        window_rows = bucket["rows"][-CHARGED_TIMEOUT_WINDOW:]
        hits = sum(1 for item in window_rows if item)
        full = len(window_rows) >= CHARGED_TIMEOUT_WINDOW
        rate = (hits / len(window_rows)) if window_rows else 0.0
        models[model] = {
            "requests_in_epoch": len(bucket["rows"]),
            "window": len(window_rows),
            "window_full": full,
            window_key: hits,
            "rate": round(rate, 4),
        }
        if full and hits > len(window_rows) * CHARGED_TIMEOUT_RATE and rate_reason is None:
            rate_reason = rate_name
    outcome_epoch, timeout_base, unparseable_base = _outcome_epoch_record(Path(ledger.path).parent)
    if micro_key == "charged_timeout_microcny":
        baseline = timeout_base
        baseline_key = "charged_timeout_baseline_microcny"
        since_key = "charged_timeout_since_baseline_microcny"
    else:
        baseline = unparseable_base
        baseline_key = "charged_unparseable_baseline_microcny"
        since_key = "charged_unparseable_since_baseline_microcny"
    since_baseline = max(0, charged_microcny - baseline)
    cap_microcny = CHARGED_TIMEOUT_EPOCH_CNY * MICROCNY_PER_CNY
    cny_reason = cny_name if since_baseline > cap_microcny else None
    return {
        "schema": SNAPSHOT_SCHEMA,
        "epoch": protocol_epoch,
        "outcome_epoch": outcome_epoch,
        "window": CHARGED_TIMEOUT_WINDOW,
        "rate_limit": CHARGED_TIMEOUT_RATE,
        "rate_applies_after_requests": CHARGED_TIMEOUT_WINDOW,
        "epoch_cny_cap": CHARGED_TIMEOUT_EPOCH_CNY,
        count_key: charged_count,
        micro_key: charged_microcny,
        baseline_key: baseline,
        since_key: since_baseline,
        "halt_reason": rate_reason or cny_reason,
        "by_model": models,
    }


def assess_charged_timeouts(ledger, *, protocol_epoch):
    """Rate and epoch CNY for charged deadline timeouts. This does not halt."""
    body = _assess_charged(
        ledger, protocol_epoch=protocol_epoch, causes=CHARGED_TIMEOUT_CAUSES,
        rate_name="charged_timeout_rate", cny_name="charged_timeout_cny",
        count_key="charged_timeouts", micro_key="charged_timeout_microcny",
        window_key="charged_timeouts_in_window")
    return body


UNPARSEABLE_CAUSES = frozenset({"unparseable_success"})


def assess_unparseable(ledger, *, protocol_epoch):
    """Same window and CNY cap as charged timeouts, for settled unparseable bodies."""
    return _assess_charged(
        ledger, protocol_epoch=protocol_epoch, causes=UNPARSEABLE_CAUSES,
        rate_name="charged_unparseable_rate", cny_name="charged_unparseable_cny",
        count_key="charged_unparseable", micro_key="charged_unparseable_microcny",
        window_key="charged_unparseable_in_window")


def _write_snapshot(run_dir, name, body):
    path = Path(run_dir) / "progress" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    return body


def publish_timeout_counts(run_dir, ledger, *, protocol_epoch):
    """Rewrite the snapshot from the ledger and diagnostics."""
    with exclusive_lock(run_dir, "timeout_counts"):
        body = assess_charged_timeouts(ledger, protocol_epoch=protocol_epoch)
        return _write_snapshot(run_dir, "timeout_counts.json", body)


def publish_unparseable_counts(run_dir, ledger, *, protocol_epoch):
    with exclusive_lock(run_dir, "unparseable_counts"):
        body = assess_unparseable(ledger, protocol_epoch=protocol_epoch)
        return _write_snapshot(run_dir, "unparseable_counts.json", body)


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
    with exclusive_lock(ledger.path.parent, "provider_outcomes"):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
