"""Operator commands for an unknown hold and a one-time unit restart.

``resolve`` settles one unknown reservation and writes an audit row. Passing
``--restart-unit`` archives that unit's partial record and checkpoint so the
scheduler will run it again. Spend already in the ledger stays there. A unit
may be restarted once; a second request is refused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from .budget import BudgetLedger, BudgetStopped
from .shardlock import clear_drain, clear_halt, exclusive_lock, set_drain


def _restarts_path(run_dir):
    return Path(run_dir) / "unit_restarts.json"


def _read_counts(run_dir):
    path = _restarts_path(run_dir)
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("unit_restarts_must_be_object")
    counts = {}
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            raise ValueError("unit_restarts_must_be_object")
        counts[key] = count
    return counts


def restart_counts(run_dir):
    with exclusive_lock(run_dir, "unit_restarts"):
        return _read_counts(run_dir)


def _write_counts(run_dir, counts):
    path = _restarts_path(run_dir)
    path.write_text(json.dumps(counts, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _unused_path(dest):
    """A path that does not exist yet. An occupied path gets ``-dupN``.

    ``shutil.move`` into an existing directory nests the source inside it and
    then crashes when that child already exists. Callers must not do that.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        return dest
    n = 2
    while True:
        candidate = dest.parent / f"{dest.name}-dup{n}"
        if not candidate.exists():
            return candidate
        n += 1


def _safe_move(source, dest):
    """Move ``source`` onto a path that does not already exist. Missing source is a no-op."""
    source = Path(source)
    if not source.exists():
        return None
    dest = _unused_path(dest)
    shutil.move(str(source), str(dest))
    return dest


def _copy_aside(source, dest):
    """Copy a live index into the archive without removing or truncating either file."""
    source = Path(source)
    if not source.is_file():
        return None
    dest = _unused_path(dest)
    shutil.copy2(source, dest)
    return dest


def _append_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _archive_tree(source, dest):
    return _safe_move(source, dest) is not None


def _park_tree(source, dest):
    """Move a live directory aside. Missing sources are left untouched."""
    return _archive_tree(source, dest)


def _archive_records(run_dir, unit_id, dest_dir):
    path = Path(run_dir) / "records.jsonl"
    if not path.is_file():
        return 0
    kept, removed = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("unit_id") == unit_id:
            removed.append(line)
        else:
            kept.append(line)
    if not removed:
        return 0
    _append_text(Path(dest_dir) / "records.jsonl", "".join(line + "\n" for line in removed))
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return len(removed)


def restart_unit(run_dir, unit_id, *, reason, ledger=None, shard_id=""):
    """Archive one interrupted unit and record that its single restart is used.

    Refuses when any unknown hold is still open, and refuses a second restart.
    """
    if not isinstance(unit_id, str) or not unit_id or "/" in unit_id or "\\" in unit_id:
        raise ValueError("restart_unit_id_rejected")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("resolve_reason_required")
    run_dir = Path(run_dir)
    ledger = ledger or BudgetLedger(run_dir / "budget.sqlite3")
    if ledger.blocks_dispatch(shard_id):
        raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")
    with exclusive_lock(run_dir, "unit_restarts"):
        return _restart_unit_locked(run_dir, unit_id, reason=reason, ledger=ledger)


def _restart_unit_locked(run_dir, unit_id, *, reason, ledger):
    counts = _read_counts(run_dir)
    if counts.get(unit_id, 0) >= 1:
        raise BudgetStopped("restart_already_used")
    generation = counts.get(unit_id, 0) + 1
    epoch = current_epoch(run_dir)
    # Generation resets with the epoch, so the directory must name the epoch.
    # A leftover from an earlier crash gets a -dupN sibling instead of a nested move.
    dest = _unused_path(run_dir / "archive" / "restarts" / f"epoch-{epoch}" / unit_id / str(generation))
    dest.mkdir(parents=True, exist_ok=False)
    # Trees move before the live record is removed. A crash here leaves the
    # record in place, so resume skips the unit instead of halting on a partial id.
    _safe_move(run_dir / "checkpoints" / unit_id, dest / "checkpoint")
    _safe_move(run_dir / "units" / unit_id, dest / "unit")
    _safe_move(run_dir / "sealed" / "swe_units" / f"{unit_id}.jsonl", dest / "swe_unit.jsonl")
    _safe_move(run_dir / "sealed" / "programbench" / unit_id, dest / "programbench")
    with exclusive_lock(run_dir, "records"):
        removed = _archive_records(run_dir, unit_id, dest)
    counts[unit_id] = generation
    _write_counts(run_dir, counts)
    return {"unit_id": unit_id, "restart_count": generation, "protocol_epoch": epoch,
            "archived_records": removed, "archive": str(dest)}


# Terminal rows left by another shard's halt, a valve, or an operator stop.
# A real budget cap or a unit that used its own resource cap is not in this set.
INFRA_STOP_REASONS = frozenset({
    "unknown_outcome_threshold",
    "tool_outcome_unknown",
    "schedule_halted",
    "blocked_fraction_exceeded",
    "charged_timeout_rate",
    "charged_timeout_cny",
    "charged_unparseable_rate",
    "charged_unparseable_cny",
    "operator",
    "operator_halt",
    "provider_outcome_charged",
    "provider_outcome_unknown",
    "second_unknown_interruption",
    "provider_timeout_exhausted",
    "command_timeout_exceeds_remaining_unit_seconds",
})


def _infra_log(run_dir):
    return Path(run_dir) / "progress" / "infra_requeues.jsonl"


def _infra_generations_unlocked(run_dir):
    path = _infra_log(run_dir)
    counts = {}
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        unit_id = row.get("unit_id") if isinstance(row, dict) else None
        if isinstance(unit_id, str) and unit_id:
            counts[unit_id] = counts.get(unit_id, 0) + 1
    return counts


def infra_generations(run_dir):
    """How many times each unit was requeued without using its protocol restart."""
    with exclusive_lock(run_dir, "infra_requeues"):
        return _infra_generations_unlocked(run_dir)


def _track_log(run_dir):
    return Path(run_dir) / "progress" / "track_requeues.jsonl"


def _track_generations_unlocked(run_dir):
    path = _track_log(run_dir)
    counts = {}
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        unit_id = row.get("unit_id") if isinstance(row, dict) else None
        if isinstance(unit_id, str) and unit_id:
            counts[unit_id] = counts.get(unit_id, 0) + 1
    return counts


def track_generations(run_dir):
    """How many times each unit was invalidated and rerun inside the same epoch."""
    with exclusive_lock(run_dir, "track_requeues"):
        return _track_generations_unlocked(run_dir)


def _load_record_rows(run_dir):
    path = Path(run_dir) / "records.jsonl"
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("unit_id"), str) and row["unit_id"]:
            rows.append(row)
    return rows


def _catalog_parents():
    from .freeze import iter_units
    parents = {}
    for study in ("m1", "m2"):
        for unit in iter_units(study):
            if unit.parent_unit_id:
                parents[unit.unit_id] = unit.parent_unit_id
    return parents


def _check_unit_id(unit_id):
    if not isinstance(unit_id, str) or not unit_id or "/" in unit_id or "\\" in unit_id:
        raise ValueError("restart_unit_id_rejected")


def requeue_interrupted(run_dir, unit_ids=()):
    """Archive interrupted units so they can run again without a protocol restart.

    Automatic selection is a terminal ``stopped`` row whose reason is an infra
    halt (another shard's unknown threshold or valve, ``tool_outcome_unknown``,
    a provider outcome that never returned a receipt, or an operator stop).
    A ``blocked`` row left by ``second_unknown_interruption`` or
    ``second_timeout_interruption`` is the same class: the one protocol restart
    was already spent on a provider unknown or a charged timeout. ``--unit``
    adds a recorded unit even when its reason is different, including a blocked
    timeout or unknown that automatic selection missed. Requeueing a unit
    also requeues recorded dependents that used
    it: the same task and arm's B1-self, and an M2 persist or learned-probe
    chain that hangs off it. ``unit_restarts.json`` is not modified. Request
    ids of the next attempt use ``-infraN-`` so they do not collide with the
    archived attempt and do not consume the one protocol restart.
    """
    run_dir = Path(run_dir)
    explicit = []
    for unit_id in unit_ids or ():
        _check_unit_id(unit_id)
        explicit.append(unit_id)
    by_id = {}
    for row in _load_record_rows(run_dir):
        by_id[row["unit_id"]] = row
    selected = {}
    for unit_id, row in by_id.items():
        status = row.get("status")
        reason = row.get("reason")
        provider_unknown = (
            row.get("termination_reason") == "provider_unknown" and status in {"stopped", "blocked"})
        # A wrong image base is skipped, not requeued. swe_image_git_failed is
        # still an infra block. skipped/unavailable rows are never selected here.
        provider_block = status == "blocked" and reason in {
            "second_unknown_interruption", "second_timeout_interruption",
            "swe_image_git_failed",
        }
        if (status == "stopped" and reason in INFRA_STOP_REASONS) or provider_unknown or provider_block:
            selected[unit_id] = "reason"
    for unit_id in explicit:
        if unit_id in by_id:
            selected[unit_id] = "explicit" if unit_id not in selected else selected[unit_id]
    parents = _catalog_parents()
    changed = True
    while changed:
        changed = False
        for unit_id, row in by_id.items():
            if unit_id in selected:
                continue
            parent = parents.get(unit_id) or ""
            if not parent and isinstance(row.get("parent_unit_id"), str):
                parent = row["parent_unit_id"]
            if parent and parent in selected:
                selected[unit_id] = parent
                changed = True
    before = restart_counts(run_dir) if (run_dir / "unit_restarts.json").is_file() else {}
    if not selected:
        return {"requeued": [], "count": 0, "restart_counts_unchanged": True}
    ledger_path = run_dir / "budget.sqlite3"
    tool_ledger = BudgetLedger(ledger_path) if ledger_path.is_file() else None
    epoch = current_epoch(run_dir)
    requeued = []
    for unit_id in sorted(selected):
        _check_unit_id(unit_id)
        dest = _unused_path(run_dir / "archive" / "restarts" / f"epoch-{epoch}" / unit_id / "infra")
        dest.mkdir(parents=True, exist_ok=False)
        _safe_move(run_dir / "checkpoints" / unit_id, dest / "checkpoint")
        _safe_move(run_dir / "units" / unit_id, dest / "unit")
        _safe_move(run_dir / "sealed" / "swe_units" / f"{unit_id}.jsonl", dest / "swe_unit.jsonl")
        _safe_move(run_dir / "sealed" / "programbench" / unit_id, dest / "programbench")
        with exclusive_lock(run_dir, "records"):
            removed = _archive_records(run_dir, unit_id, dest)
        if tool_ledger is not None:
            tool_ledger.release_unsettled_tools(unit_id)
        source = selected[unit_id]
        cascaded_from = None if source in {"reason", "explicit"} else source
        entry = {
            "unit_id": unit_id,
            "status": by_id[unit_id].get("status"),
            "reason": by_id[unit_id].get("reason") or "",
            "protocol_epoch": epoch,
            "archive": str(dest),
            "source": "cascade" if cascaded_from else source,
            "cascaded_from": cascaded_from,
            "explicit": unit_id in explicit,
            "archived_records": removed,
            "unit_restarts_incremented": False,
        }
        with exclusive_lock(run_dir, "infra_requeues"):
            _append_text(_infra_log(run_dir), json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        requeued.append(unit_id)
    after = restart_counts(run_dir) if (run_dir / "unit_restarts.json").is_file() else {}
    if after != before:
        raise RuntimeError("unit_restarts_changed")
    outcome_epoch = current_outcome_epoch(run_dir)
    if requeued:
        outcome_epoch = begin_outcome_epoch(run_dir, reason="requeue-interrupted")["epoch"]
    return {"requeued": requeued, "count": len(requeued), "restart_counts_unchanged": True,
            "outcome_epoch": outcome_epoch}


def _epoch_path(run_dir):
    return Path(run_dir) / "protocol_epoch.json"


def _outcome_epoch_path(run_dir):
    return Path(run_dir) / "progress" / "outcome_epoch.json"


def _read_outcome_epoch(run_dir):
    path = _outcome_epoch_path(run_dir)
    if not path.is_file():
        return 0
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    epoch = value.get("epoch") if isinstance(value, dict) else None
    if type(epoch) is not int or epoch < 0:
        return 0
    return epoch


def current_outcome_epoch(run_dir):
    """Gate epoch for unknown and blocked-fraction counts. Historical lines stay."""
    with exclusive_lock(run_dir, "outcome_epoch"):
        return _read_outcome_epoch(run_dir)


def begin_outcome_epoch(run_dir, *, reason):
    """Start a new gate epoch without deleting unknown lines or archiving units.

    ``unknown_outcomes.jsonl`` keeps every line. New lines carry ``outcome_epoch``.
    Counts for the unknown threshold and the blocked-fraction gate use only the
    current epoch. Untagged lines belong to epoch 0. An audit row is appended to
    ``progress/outcome_epochs.jsonl``. This is not ``new-epoch``: that command
    archives the protocol generation and moves the unknown log aside.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("outcome_epoch_reason_required")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(run_dir, "outcome_epoch"):
        epoch = _read_outcome_epoch(run_dir) + 1
        path = _outcome_epoch_path(run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"epoch": epoch, "reason": reason.strip()}
        path.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        audit = {"epoch": epoch, "reason": reason.strip()}
        _append_text(run_dir / "progress" / "outcome_epochs.jsonl",
                     json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n")
        return body


def current_epoch(run_dir):
    """0 before any protocol rerun. Later epochs are 1, 2, ..."""
    path = _epoch_path(run_dir)
    if not path.is_file():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    epoch = value.get("epoch") if isinstance(value, dict) else None
    if type(epoch) is not int or epoch < 0:
        raise ValueError("invalid_protocol_epoch")
    return epoch


def begin_epoch(run_dir, *, reason):
    """Start the next request-id generation and clear this epoch's restart counts.

    Ledger rows stay. Their CNY remains in the model and total caps. Restart
    allowances and the unknown-outcome counter belong to the epoch being left.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("resolve_reason_required")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = current_epoch(run_dir)
    epoch = previous + 1
    unknown = run_dir / "unknown_outcomes.jsonl"
    if unknown.is_file() and unknown.stat().st_size:
        dest = run_dir / "archive" / "epochs" / str(previous)
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "unknown_outcomes.jsonl"
        if target.exists():
            _append_text(target, unknown.read_text(encoding="utf-8"))
            unknown.unlink()
        else:
            shutil.move(str(unknown), str(target))
    with exclusive_lock(run_dir, "unit_restarts"):
        _write_counts(run_dir, {})
    _epoch_path(run_dir).write_text(json.dumps(
        {"epoch": epoch, "reason": reason.strip()}, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"epoch": epoch, "previous_epoch": previous, "restart_counts_reset": True}


def archive_finished(run_dir, *, reason):
    """Move finished unit records aside and start a new protocol epoch.

    An empty or missing records file still bumps the epoch. That is the path
    for a run directory whose archive was already written. A leftover ``units``
    or ``checkpoints`` tree is moved into that epoch's archive either way, so
    the next epoch does not inherit a partial workspace. The ledger is not
    rewritten. Spending from earlier epochs stays inside the CNY caps.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("resolve_reason_required")
    run_dir = Path(run_dir)
    records = run_dir / "records.jsonl"
    rows = []
    if records.is_file():
        for line in records.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    archived_units = 0
    moved = 0
    archive = ""
    previous = current_epoch(run_dir)
    root = run_dir / "archive" / "protocol-rerun"
    dest = root / f"epoch-{previous}"
    if rows:
        dest.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        with (dest / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(body)
        with (root / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(body)
        with (dest / "reason.txt").open("a", encoding="utf-8") as handle:
            handle.write(reason.strip() + "\n")
        for row in rows:
            unit_id = row.get("unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                continue
            unit_dest = dest / unit_id
            unit_dest.mkdir(parents=True, exist_ok=True)
            if _safe_move(run_dir / "checkpoints" / unit_id, unit_dest / "checkpoint"):
                moved += 1
            _safe_move(run_dir / "sealed" / "swe_units" / f"{unit_id}.jsonl", unit_dest / "swe_unit.jsonl")
            _safe_move(run_dir / "sealed" / "programbench" / unit_id, unit_dest / "programbench")
        records.write_text("", encoding="utf-8")
        archived_units = len(rows)
        archive = str(dest)
    # Live harness indexes are rewritten from the current records on the next
    # collect. Keep a copy with this epoch before that happens.
    if _copy_aside(run_dir / "sealed" / "swe_predictions.jsonl", dest / "swe_predictions.jsonl"):
        archive = str(dest)
    if _copy_aside(run_dir / "sealed" / "manifest.jsonl", dest / "manifest.jsonl"):
        archive = str(dest)
    parked_units = _park_tree(run_dir / "units", dest / "units")
    parked_checkpoints = _park_tree(run_dir / "checkpoints", dest / "checkpoints")
    if parked_units or parked_checkpoints:
        archive = str(dest)
    epoch = begin_epoch(run_dir, reason=reason)
    return {"archived_units": archived_units, "checkpoints": moved, "archive": archive, **epoch}


def _program_bench_unit_ids():
    from .freeze import iter_units
    found = []
    for study in ("m1", "m2"):
        for unit in iter_units(study):
            if unit.track == "ProgramBench":
                found.append(unit.unit_id)
    return found


def _invalidation_dirs(run_dir, epoch):
    parent = Path(run_dir) / "archive" / "invalidated" / f"epoch-{epoch}"
    if not parent.is_dir():
        return []
    return [path for path in sorted(parent.iterdir())
            if path.is_dir() and (path.name == "programbench" or path.name.startswith("programbench-dup"))]


def _archived_track_units(run_dir, epoch):
    found = set()
    for directory in _invalidation_dirs(run_dir, epoch):
        records = directory / "records.jsonl"
        if records.is_file():
            for line in records.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                unit_id = row.get("unit_id") if isinstance(row, dict) else None
                if isinstance(unit_id, str) and unit_id:
                    found.add(unit_id)
        for child in directory.iterdir():
            if child.is_dir():
                found.add(child.name)
    return found


def _current_epoch_record(row, epoch):
    recorded = row.get("protocol_epoch")
    return not isinstance(recorded, int) or recorded == epoch


def requeue_track(run_dir, benchmark, *, reason):
    """Archive one benchmark's current-epoch units and let them run again.

    This is the pre-registered harness-prompt deviation. Spend already in the
    ledger stays there. Restart counts, unknown-outcome lines, and the timeout
    valve contribution of these units are cleared. The next attempt uses
    ``-pbvN-`` so it does not collide with the archived attempt. SWE-bench
    records are not selected when ``benchmark`` is ProgramBench.
    """
    if benchmark != "ProgramBench":
        raise ValueError("requeue_track_benchmark_rejected")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("resolve_reason_required")
    run_dir = Path(run_dir)
    epoch = current_epoch(run_dir)
    catalog = _program_bench_unit_ids()
    catalog_set = set(catalog)
    selected = set()
    for row in _load_record_rows(run_dir):
        unit_id = row.get("unit_id")
        if unit_id in catalog_set and _current_epoch_record(row, epoch):
            selected.add(unit_id)
    for unit_id in catalog:
        if ((run_dir / "checkpoints" / unit_id).exists()
                or (run_dir / "units" / unit_id).exists()
                or (run_dir / "sealed" / "programbench" / unit_id).exists()):
            selected.add(unit_id)
    prior = _track_generations_unlocked(run_dir)
    dest = None
    if selected:
        batch_gen = 1 + max((prior.get(unit_id, 0) for unit_id in selected), default=0)
        # The first invalidation keeps the name already used on the box.
        # A later one is programbench-pbvN, or that name plus -dupK if occupied.
        name = "programbench" if batch_gen == 1 else f"programbench-pbv{batch_gen}"
        base = run_dir / "archive" / "invalidated" / f"epoch-{epoch}" / name
        dest = base if not base.exists() else _unused_path(base)
        dest.mkdir(parents=True, exist_ok=False)
        for unit_id in sorted(selected):
            _check_unit_id(unit_id)
            unit_dest = dest / unit_id
            unit_dest.mkdir(parents=True, exist_ok=False)
            _safe_move(run_dir / "checkpoints" / unit_id, unit_dest / "checkpoint")
            _safe_move(run_dir / "units" / unit_id, unit_dest / "unit")
            _safe_move(run_dir / "sealed" / "programbench" / unit_id, unit_dest / "programbench")
        with exclusive_lock(run_dir, "records"):
            path = run_dir / "records.jsonl"
            kept, removed = [], []
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    unit_id = row.get("unit_id") if isinstance(row, dict) else None
                    if (isinstance(row, dict) and unit_id in selected
                            and _current_epoch_record(row, epoch)):
                        removed.append(line)
                    else:
                        kept.append(line)
            if removed:
                _append_text(dest / "records.jsonl", "".join(line + "\n" for line in removed))
                path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    repair = [unit_id for unit_id in sorted(_archived_track_units(run_dir, epoch))
              if unit_id in catalog_set and prior.get(unit_id, 0) == 0 and unit_id not in selected]
    audited = sorted(selected) + repair
    if not audited:
        return {"requeued": [], "count": 0, "benchmark": benchmark, "protocol_epoch": epoch,
                "restart_counts_reset": True}
    entries = []
    with exclusive_lock(run_dir, "track_requeues"):
        counts = _track_generations_unlocked(run_dir)
        for unit_id in audited:
            generation = counts.get(unit_id, 0) + 1
            counts[unit_id] = generation
            entry = {
                "unit_id": unit_id,
                "benchmark": benchmark,
                "reason": reason.strip(),
                "protocol_epoch": epoch,
                "archive": str(dest) if dest is not None else "",
                "request_id_marker": f"-pbv{generation}-",
                "restart_counts_reset": True,
                "deviation": "harness_prompt_bug",
            }
            _append_text(_track_log(run_dir), json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            entries.append(entry)
    audited_set = set(audited)
    if (run_dir / "unit_restarts.json").is_file():
        with exclusive_lock(run_dir, "unit_restarts"):
            counts = _read_counts(run_dir)
            for unit_id in audited_set:
                counts.pop(unit_id, None)
            _write_counts(run_dir, counts)
    unknown = run_dir / "unknown_outcomes.jsonl"
    if unknown.is_file():
        with exclusive_lock(run_dir, "unknown"):
            kept = []
            for line in unknown.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get("unit_id") in audited_set:
                    continue
                kept.append(line)
            unknown.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    ledger_path = run_dir / "budget.sqlite3"
    if ledger_path.is_file():
        from .progress import publish_timeout_counts, publish_unparseable_counts
        ledger = BudgetLedger(ledger_path)
        for unit_id in audited:
            ledger.release_unsettled_tools(unit_id)
        publish_timeout_counts(run_dir, ledger, protocol_epoch=epoch)
        publish_unparseable_counts(run_dir, ledger, protocol_epoch=epoch)
    return {"requeued": [entry["unit_id"] for entry in entries], "count": len(entries),
            "benchmark": benchmark, "protocol_epoch": epoch, "restart_counts_reset": True,
            "archive": str(dest) if dest is not None else ""}


def mark_unavailable(run_dir, *, task_id=None, unit_ids=()):
    """Append a skipped record over a blocked image-base mismatch.

    The previous ``blocked`` line stays in ``records.jsonl``. The latest line
    for that unit is ``skipped`` / ``swe_image_base_unavailable``, so the
    blocked-fraction gate no longer counts it and ``requeue-interrupted`` does
    not select it. An audit row is appended to ``progress/unavailable.jsonl``.
    """
    if not task_id and not unit_ids:
        raise ValueError("mark_unavailable_target_required")
    if task_id is not None and (not isinstance(task_id, str) or not task_id or "/" in task_id):
        raise ValueError("mark_unavailable_task_rejected")
    named = []
    for unit_id in unit_ids or ():
        _check_unit_id(unit_id)
        named.append(unit_id)
    named_set = set(named)
    run_dir = Path(run_dir)
    latest = {}
    for row in _load_record_rows(run_dir):
        latest[row["unit_id"]] = row
    chosen = []
    for unit_id, row in latest.items():
        if row.get("status") != "blocked" or row.get("reason") != "swe_image_base_commit_mismatch":
            continue
        if task_id and row.get("task_id") != task_id:
            continue
        if named_set and unit_id not in named_set:
            continue
        chosen.append(row)
    if not chosen:
        return {"skipped": [], "count": 0, "task_id": task_id or ""}
    records = []
    audits = []
    for row in chosen:
        updated = dict(row)
        updated["status"] = "skipped"
        updated["reason"] = "swe_image_base_unavailable"
        updated["termination_reason"] = "skipped"
        updated["unavailable_from"] = "swe_image_base_commit_mismatch"
        records.append(json.dumps(updated, ensure_ascii=False, sort_keys=True))
        audits.append(json.dumps({
            "unit_id": row["unit_id"],
            "task_id": row.get("task_id") or "",
            "previous_status": row.get("status"),
            "previous_reason": row.get("reason"),
            "status": "skipped",
            "reason": "swe_image_base_unavailable",
        }, ensure_ascii=False, sort_keys=True))
    with exclusive_lock(run_dir, "records"):
        _append_text(run_dir / "records.jsonl", "".join(line + "\n" for line in records))
        _append_text(run_dir / "progress" / "unavailable.jsonl", "".join(line + "\n" for line in audits))
    return {"skipped": [row["unit_id"] for row in chosen], "count": len(chosen),
            "task_id": task_id or ""}


def _parser():
    parser = argparse.ArgumentParser(description="Resolve an unknown hold or restart one unit.")
    parser.add_argument("command", choices=(
        "resolve", "archive-finished", "new-epoch", "outcome-epoch", "drain",
        "requeue-interrupted", "requeue-track", "clear-halt", "mark-unavailable"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--charge-microcny", type=int, default=None)
    parser.add_argument("--worst-case", action="store_true")
    parser.add_argument("--reason", default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--restart-unit", default=None)
    parser.add_argument("--unit", action="append", default=[])
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--task", default=None)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "drain":
        if args.clear:
            clear_drain(args.run_dir)
            print(json.dumps({"drain": False}, sort_keys=True))
        else:
            path = set_drain(args.run_dir)
            print(json.dumps({"drain": True, "path": path}, sort_keys=True))
        return 0
    if args.command == "outcome-epoch":
        if not args.reason:
            parser.error("--reason is required")
        try:
            result = begin_outcome_epoch(args.run_dir, reason=args.reason)
        except ValueError as error:
            print(str(error))
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "mark-unavailable":
        if not args.task and not args.unit:
            parser.error("pass --task or --unit")
        try:
            result = mark_unavailable(args.run_dir, task_id=args.task, unit_ids=args.unit)
        except ValueError as error:
            print(str(error))
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "clear-halt":
        removed = clear_halt(args.run_dir)
        print(json.dumps({"halt_cleared": removed}, sort_keys=True))
        return 0
    if args.command == "requeue-interrupted":
        try:
            result = requeue_interrupted(args.run_dir, args.unit)
        except ValueError as error:
            print(str(error))
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "requeue-track":
        if args.benchmark != "ProgramBench":
            parser.error("--benchmark ProgramBench is required")
        if not args.reason:
            parser.error("--reason is required")
        try:
            result = requeue_track(args.run_dir, args.benchmark, reason=args.reason)
        except ValueError as error:
            print(str(error))
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command in {"archive-finished", "new-epoch"}:
        if not args.reason:
            parser.error("--reason is required")
        try:
            if args.command == "new-epoch":
                result = begin_epoch(args.run_dir, reason=args.reason)
            else:
                result = archive_finished(args.run_dir, reason=args.reason)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        except (BudgetStopped, ValueError) as error:
            print(str(error))
            return 2
        return 0
    if not args.reason:
        parser.error("--reason is required")
    if args.command != "resolve":
        parser.error("resolve is required")
    if args.request_id and (args.charge_microcny is None) == (not args.worst_case):
        parser.error("pass exactly one of --charge-microcny or --worst-case")
    if not args.request_id and not args.restart_unit:
        parser.error("pass --request-id or --restart-unit")
    if not args.request_id and (args.worst_case or args.charge_microcny is not None):
        parser.error("--charge-microcny and --worst-case require --request-id")
    run_dir = Path(args.run_dir)
    ledger = BudgetLedger(run_dir / "budget.sqlite3")
    summary = {}
    try:
        if args.request_id:
            summary["resolved"] = ledger.resolve_unknown(
                args.request_id, args.charge_microcny, reason=args.reason, worst_case=args.worst_case)
        if args.restart_unit:
            summary["restart"] = restart_unit(run_dir, args.restart_unit, reason=args.reason, ledger=ledger)
    except (BudgetStopped, ValueError) as error:
        print(str(error))
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
