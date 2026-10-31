"""File locks for run-directory counters shared by task shards.

The budget ledger already serializes CNY with ``BEGIN IMMEDIATE``. These locks
cover the JSON and JSONL files that several shard processes otherwise rewrite
as a whole.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path


@contextmanager
def exclusive_lock(run_dir, name):
    directory = Path(run_dir) / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / name, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _halt_path(run_dir):
    return Path(run_dir) / "schedule_halt.json"


# Only a global CNY breach stops another shard before its next model call.
# Unknown thresholds, valves, tool failures and operator stops drain instead.
HARD_BUDGET_HALTS = frozenset({
    "model_cny_cap_exhausted",
    "cumulative_total_cny_limit",
    "budget_infeasible_all_models",
})


def halt_record(run_dir):
    """The shared halt, or None. A file with no ``hard`` field stays hard.

    That default keeps a halt written before this distinction from becoming a
    drain. ``CLEAR_HALT=1`` removes it. New writers set ``hard`` explicitly.
    """
    path = _halt_path(run_dir)
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"halt": True, "reason": "schedule_halted", "shard": "", "hard": True}
    if isinstance(body, dict) and body.get("halt") is True and isinstance(body.get("reason"), str) and body["reason"]:
        hard = body["hard"] if type(body.get("hard")) is bool else True
        shard = body.get("shard") if isinstance(body.get("shard"), str) else ""
        return {"halt": True, "reason": body["reason"], "shard": shard, "hard": hard}
    return None


def halt_reason(run_dir):
    """The shared stop reason, or None when no shard has halted the schedule."""
    record = halt_record(run_dir)
    if record is None:
        return None
    return record["reason"]


def _drain_path(run_dir):
    return Path(run_dir) / "drain.json"


def drain_requested(run_dir):
    """True when an operator asked shards to stop after the current unit."""
    path = _drain_path(run_dir)
    if not path.is_file():
        return False
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if isinstance(body, dict) and body.get("drain") is False:
        return False
    return True


def set_drain(run_dir):
    path = _drain_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"drain": True}, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def clear_drain(run_dir):
    path = _drain_path(run_dir)
    if path.is_file():
        path.unlink()
        return True
    return False


def set_halt(run_dir, reason, shard="", hard=None):
    """Record the first halt. Later writers leave that reason in place.

    ``hard`` defaults from the reason. Budget-cap reasons stop every shard
    before its next model call. Every other reason is a drain: other shards
    finish the unit they are already in, then exit.
    """
    if not isinstance(reason, str) or not reason.strip():
        reason = "schedule_halted"
    reason = reason.strip()
    if hard is None:
        hard = reason in HARD_BUDGET_HALTS
    if type(hard) is not bool:
        hard = True
    with exclusive_lock(run_dir, "halt"):
        existing = halt_reason(run_dir)
        if existing:
            return existing
        _halt_path(run_dir).write_text(json.dumps(
            {"halt": True, "hard": hard, "reason": reason, "shard": shard},
            ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return reason


def clear_halt(run_dir):
    """Remove ``schedule_halt.json``. Launch does this only when ``CLEAR_HALT=1``."""
    path = _halt_path(run_dir)
    if path.is_file():
        path.unlink()
        return True
    return False
