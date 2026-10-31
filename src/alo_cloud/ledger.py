"""Operator commands for an unknown hold and a one-time unit restart.

``resolve`` settles one unknown reservation and writes an audit row. Passing
``--restart-unit`` archives that unit's partial record and checkpoint so the
scheduler will run it again. Spend already in the ledger stays there. A unit
may be restarted once; a second request is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .budget import BudgetLedger, BudgetStopped


def _restarts_path(run_dir):
    return Path(run_dir) / "unit_restarts.json"


def restart_counts(run_dir):
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


def _write_counts(run_dir, counts):
    path = _restarts_path(run_dir)
    path.write_text(json.dumps(counts, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _archive_tree(source, dest):
    source = Path(source)
    if not source.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    return True


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
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "records.jsonl").write_text("\n".join(removed) + "\n", encoding="utf-8")
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return len(removed)


def restart_unit(run_dir, unit_id, *, reason, ledger=None):
    """Archive one interrupted unit and record that its single restart is used.

    Refuses when any unknown hold is still open, and refuses a second restart.
    """
    if not isinstance(unit_id, str) or not unit_id or "/" in unit_id or "\\" in unit_id:
        raise ValueError("restart_unit_id_rejected")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("resolve_reason_required")
    run_dir = Path(run_dir)
    ledger = ledger or BudgetLedger(run_dir / "budget.sqlite3")
    state = ledger.snapshot()
    if state["unresolved_request_ids"] or state["unresolved_tool_ids"]:
        raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")
    counts = restart_counts(run_dir)
    if counts.get(unit_id, 0) >= 1:
        raise BudgetStopped("restart_already_used")
    generation = counts.get(unit_id, 0) + 1
    dest = run_dir / "archive" / "restarts" / unit_id / str(generation)
    removed = _archive_records(run_dir, unit_id, dest)
    _archive_tree(run_dir / "checkpoints" / unit_id, dest / "checkpoint")
    _archive_tree(run_dir / "units" / unit_id, dest / "unit")
    _archive_tree(run_dir / "sealed" / "swe_units" / f"{unit_id}.jsonl", dest / "swe_unit.jsonl")
    counts[unit_id] = generation
    _write_counts(run_dir, counts)
    return {"unit_id": unit_id, "restart_count": generation, "archived_records": removed,
            "archive": str(dest)}


def _epoch_path(run_dir):
    return Path(run_dir) / "protocol_epoch.json"


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
        shutil.move(str(unknown), str(dest / "unknown_outcomes.jsonl"))
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
            if _archive_tree(run_dir / "checkpoints" / unit_id, unit_dest / "checkpoint"):
                moved += 1
            _archive_tree(run_dir / "sealed" / "swe_units" / f"{unit_id}.jsonl", unit_dest / "swe_unit.jsonl")
        records.write_text("", encoding="utf-8")
        archived_units = len(rows)
        archive = str(dest)
    parked_units = _park_tree(run_dir / "units", dest / "units")
    parked_checkpoints = _park_tree(run_dir / "checkpoints", dest / "checkpoints")
    if parked_units or parked_checkpoints:
        archive = str(dest)
    epoch = begin_epoch(run_dir, reason=reason)
    return {"archived_units": archived_units, "checkpoints": moved, "archive": archive, **epoch}


def _parser():
    parser = argparse.ArgumentParser(description="Resolve an unknown hold or restart one unit.")
    parser.add_argument("command", choices=("resolve", "archive-finished", "new-epoch"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--charge-microcny", type=int, default=None)
    parser.add_argument("--worst-case", action="store_true")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--restart-unit", default=None)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command in {"archive-finished", "new-epoch"}:
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
