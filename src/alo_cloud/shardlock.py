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


def halt_reason(run_dir):
    """The shared stop reason, or None when no shard has halted the schedule."""
    path = _halt_path(run_dir)
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "schedule_halted"
    if isinstance(body, dict) and body.get("halt") is True and isinstance(body.get("reason"), str) and body["reason"]:
        return body["reason"]
    return None


def set_halt(run_dir, reason, shard=""):
    """Record the first halt. Later writers leave that reason in place."""
    if not isinstance(reason, str) or not reason.strip():
        reason = "schedule_halted"
    with exclusive_lock(run_dir, "halt"):
        existing = halt_reason(run_dir)
        if existing:
            return existing
        _halt_path(run_dir).write_text(json.dumps(
            {"halt": True, "reason": reason.strip(), "shard": shard},
            ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return reason.strip()
