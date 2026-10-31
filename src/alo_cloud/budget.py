"""Small offline reservation component; never permission to contact a provider.

One SQLite writer transaction protects a local ledger. This is not a distributed
quota service, price calculator, complete usage reconciler, or immutable audit
store. Production admission must bind trusted opening commitments and usage
bounds independently; there are no historical balances or prices in this file.

Schema v2 adds the four dimensions ``ConstrainedThompsonPolicy`` reads
(requests, tokens, seconds, cny) plus a separate tool-call counter. Requests,
tokens and seconds are unit-shared; CNY stays per model. A v1 file is refused
rather than opened with a partial balance. Missing ceilings do not become zero.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3


SCHEMA = "alo-local-budget-scaffold-v2"
MICROCNY_PER_CNY = 1_000_000


class BudgetStopped(RuntimeError):
    """Keep the reservation and stop; no automatic retry or budget reset."""


def _integer(value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("nonnegative_integer_accounting_required")
    return value


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value):
        raise ValueError("non_secret_request_identity_required")
    return value


def _ceiling(max_value, prior, label):
    if (max_value is None) != (prior is None):
        raise ValueError(f"complete_{label}_ceiling_required")
    if max_value is None:
        return None
    _integer(max_value, positive=True)
    _integer(prior)
    if prior > max_value:
        raise ValueError(f"opening_{label}_exceeds_limit")
    return max_value, prior


class BudgetLedger:
    """Open an existing explicit local ledger; missing state is never recreated."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_symlink() or not self.path.is_file():
            raise BudgetStopped("existing_regular_budget_ledger_required")
        conn = self._connect()
        try:
            if conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone() != (SCHEMA,):
                raise BudgetStopped("unsupported_budget_schema")
        finally:
            conn.close()

    @classmethod
    def create(cls, path, *, model_limits_microcny, opening_commitments_microcny, max_requests, prior_requests,
               max_tokens=None, prior_tokens=None, max_seconds_ms=None, prior_seconds_ms=None,
               max_tools=None, prior_tools=None):
        """Create only new state, with explicit opening usage for every model.

        Passing zero opening usage is a caller assertion, not evidence of a fresh
        account. Real execution must verify that assertion outside this component.
        Token, second and tool ceilings are optional so older call sites keep
        working; without them, ``remaining_resources`` refuses to invent zeros.
        """
        limits, opening = dict(model_limits_microcny), dict(opening_commitments_microcny)
        if not limits or set(limits) != set(opening):
            raise ValueError("complete_opening_commitments_required")
        for model, limit in limits.items():
            _id(model)
            _integer(limit, positive=True)
            _integer(opening[model])
            if opening[model] > limit:
                raise ValueError("opening_commitment_exceeds_limit")
        _integer(max_requests, positive=True)
        _integer(prior_requests)
        if prior_requests > max_requests:
            raise ValueError("opening_request_count_exceeds_limit")
        token_ceiling = _ceiling(max_tokens, prior_tokens, "token")
        second_ceiling = _ceiling(max_seconds_ms, prior_seconds_ms, "seconds")
        tool_ceiling = _ceiling(max_tools, prior_tools, "tool")
        path = Path(path)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        conn = sqlite3.connect(path)
        try:
            conn.executescript("""
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE requests(id TEXT PRIMARY KEY, model TEXT NOT NULL,
                    reserved INTEGER NOT NULL, held INTEGER NOT NULL, status TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    reserved_tokens INTEGER NOT NULL DEFAULT 0,
                    held_tokens INTEGER NOT NULL DEFAULT 0,
                    reserved_seconds_ms INTEGER NOT NULL DEFAULT 0,
                    held_seconds_ms INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE tool_calls(id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    reserved_seconds_ms INTEGER NOT NULL, held_seconds_ms INTEGER NOT NULL);
            """)
            values = {"schema": SCHEMA, "limits": json.dumps(limits),
                      "opening": json.dumps(opening), "max_requests": str(max_requests),
                      "prior_requests": str(prior_requests)}
            if token_ceiling is not None:
                values["max_tokens"] = str(token_ceiling[0])
                values["prior_tokens"] = str(token_ceiling[1])
            if second_ceiling is not None:
                values["max_seconds_ms"] = str(second_ceiling[0])
                values["prior_seconds_ms"] = str(second_ceiling[1])
            if tool_ceiling is not None:
                values["max_tools"] = str(tool_ceiling[0])
                values["prior_tools"] = str(tool_ceiling[1])
            conn.executemany("INSERT INTO meta VALUES (?,?)", values.items())
            conn.commit()
        finally:
            conn.close()
        return cls(path)

    def _connect(self):
        if self.path.is_symlink() or not self.path.is_file():
            raise BudgetStopped("budget_state_missing_or_replaced")
        conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=5)
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @contextmanager
    def _transaction(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _snapshot(conn):
        meta = dict(conn.execute("SELECT key,value FROM meta"))
        limits, used = json.loads(meta["limits"]), json.loads(meta["opening"])
        rows = list(conn.execute(
            "SELECT id,model,reserved,held,status,held_tokens,held_seconds_ms FROM requests ORDER BY rowid"))
        tools = list(conn.execute("SELECT id,status,held_seconds_ms FROM tool_calls ORDER BY rowid"))
        for _, model, _, held, _, _, _ in rows:
            used[model] += held
        used_tokens = int(meta.get("prior_tokens", "0")) + sum(row[5] for row in rows)
        used_seconds_ms = (int(meta.get("prior_seconds_ms", "0")) + sum(row[6] for row in rows)
                           + sum(row[2] for row in tools))
        state = {"limits_microcny": limits, "committed_or_reserved_microcny": used,
                 "remaining_microcny": {model: limits[model] - used[model] for model in limits},
                 "unresolved_request_ids": [row[0] for row in rows if row[4] != "settled"],
                 "unresolved_tool_ids": [row[0] for row in tools if row[1] != "settled"],
                 "requests": int(meta["prior_requests"]) + len(rows),
                 "max_requests": int(meta["max_requests"]),
                 "used_tokens": used_tokens,
                 "used_seconds_ms": used_seconds_ms,
                 "used_tools": int(meta.get("prior_tools", "0")) + len(tools),
                 "is_execution_permit": False}
        for key in ("max_tokens", "prior_tokens", "max_seconds_ms", "prior_seconds_ms", "max_tools", "prior_tools"):
            if key in meta:
                state[key] = int(meta[key])
        return state

    def snapshot(self):
        """Return cumulative opening commitments plus all held new attempts."""
        with self._transaction() as conn:
            return self._snapshot(conn)

    def remaining_resources(self, model):
        """Policy view: shared unit remainder plus that model's remaining CNY.

        ``requests``, ``tokens`` and ``seconds`` are the active unit's shared
        ceilings, not a second per-model grant. ``cny`` is yuan as a float of
        the integer microcny remainder. Unresolved holds and missing ceilings
        stop the read; they are not reported as zero.
        """
        _id(model)
        with self._transaction() as conn:
            state = self._snapshot(conn)
        if model not in state["remaining_microcny"]:
            raise BudgetStopped("model_not_in_frozen_pool")
        if any(key not in state for key in ("max_tokens", "max_seconds_ms")):
            raise BudgetStopped("resource_dimensions_unconfigured")
        if state["unresolved_request_ids"] or state["unresolved_tool_ids"]:
            raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")
        remaining = {
            "requests": state["max_requests"] - state["requests"],
            "tokens": state["max_tokens"] - state["used_tokens"],
            "seconds": (state["max_seconds_ms"] - state["used_seconds_ms"]) / 1000,
            "cny": state["remaining_microcny"][model] / MICROCNY_PER_CNY,
        }
        if any(value < 0 for value in remaining.values()):
            raise BudgetStopped("resource_accounting_negative")
        return remaining

    def _blocked(self, state):
        if state["unresolved_request_ids"] or state["unresolved_tool_ids"]:
            raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")

    def reserve(self, request_id, model, amount_microcny, *, payload_sha256, tokens=0, seconds_ms=0):
        """Persist a unique worst-case hold before any separately permitted send."""
        _id(request_id)
        _id(model)
        _integer(amount_microcny, positive=True)
        _integer(tokens)
        _integer(seconds_ms)
        if not isinstance(payload_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", payload_sha256):
            raise ValueError("frozen_payload_digest_required")
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM requests WHERE id=?", (request_id,)).fetchone():
                raise BudgetStopped("request_identity_consumed_never_replay")
            state = self._snapshot(conn)
            self._blocked(state)
            if model not in state["remaining_microcny"]:
                raise BudgetStopped("model_not_in_frozen_pool")
            if state["requests"] >= state["max_requests"] or amount_microcny > state["remaining_microcny"][model]:
                raise BudgetStopped("cumulative_request_or_model_limit")
            if tokens or seconds_ms or "max_tokens" in state or "max_seconds_ms" in state:
                if "max_tokens" not in state or "max_seconds_ms" not in state:
                    raise BudgetStopped("resource_dimensions_unconfigured")
                if (state["used_tokens"] + tokens > state["max_tokens"]
                        or state["used_seconds_ms"] + seconds_ms > state["max_seconds_ms"]):
                    raise BudgetStopped("cumulative_request_or_model_limit")
            conn.execute(
                """INSERT INTO requests
                   (id, model, reserved, held, status, payload_sha256,
                    reserved_tokens, held_tokens, reserved_seconds_ms, held_seconds_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (request_id, model, amount_microcny, amount_microcny, "reserved", payload_sha256,
                 tokens, tokens, seconds_ms, seconds_ms))
        return {"request_id": request_id, "reserved_microcny": amount_microcny, "is_execution_permit": False}

    def settle_known(self, request_id, charge_microcny, *, tokens=None, seconds_ms=None):
        """Only resolve a known receipt once; an exceeded bound remains halted."""
        _integer(charge_microcny)
        if tokens is not None:
            _integer(tokens)
        if seconds_ms is not None:
            _integer(seconds_ms)
        breach = False
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT reserved,status,reserved_tokens,reserved_seconds_ms FROM requests WHERE id=?",
                (request_id,)).fetchone()
            if row is None or row[1] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_settled")
            reserved_cny, _, reserved_tokens, reserved_seconds = row
            held_tokens, held_seconds = reserved_tokens, reserved_seconds
            breach = charge_microcny > reserved_cny
            dimensioned = any((reserved_tokens, reserved_seconds, tokens is not None, seconds_ms is not None))
            if dimensioned:
                if tokens is None or seconds_ms is None:
                    breach = True
                else:
                    if tokens > reserved_tokens or seconds_ms > reserved_seconds:
                        breach = True
                    held_tokens = max(tokens, reserved_tokens) if tokens > reserved_tokens else tokens
                    held_seconds = max(seconds_ms, reserved_seconds) if seconds_ms > reserved_seconds else seconds_ms
            conn.execute(
                """UPDATE requests
                   SET held=?, held_tokens=?, held_seconds_ms=?, status=?
                   WHERE id=?""",
                (max(charge_microcny, reserved_cny) if charge_microcny > reserved_cny else charge_microcny,
                 held_tokens, held_seconds, "unknown" if breach else "settled", request_id))
        if breach:
            raise BudgetStopped("observed_charge_exceeded_frozen_bound")

    def mark_unknown(self, request_id):
        """Keep the full hold across reopen/cancellation; there is no reset API."""
        with self._transaction() as conn:
            row = conn.execute("SELECT status FROM requests WHERE id=?", (request_id,)).fetchone()
            if row is None or row[0] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_marked_unknown")
            conn.execute("UPDATE requests SET status='unknown' WHERE id=?", (request_id,))

    def reserve_tool(self, tool_id, *, seconds_ms):
        """Hold one tool call and its worst-case seconds before execution."""
        _id(tool_id)
        _integer(seconds_ms)
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM tool_calls WHERE id=?", (tool_id,)).fetchone():
                raise BudgetStopped("request_identity_consumed_never_replay")
            state = self._snapshot(conn)
            self._blocked(state)
            if "max_tools" not in state or "max_seconds_ms" not in state:
                raise BudgetStopped("resource_dimensions_unconfigured")
            if (state["used_tools"] >= state["max_tools"]
                    or state["used_seconds_ms"] + seconds_ms > state["max_seconds_ms"]):
                raise BudgetStopped("cumulative_request_or_model_limit")
            conn.execute(
                "INSERT INTO tool_calls (id, status, reserved_seconds_ms, held_seconds_ms) VALUES (?,?,?,?)",
                (tool_id, "reserved", seconds_ms, seconds_ms))
        return {"tool_id": tool_id, "is_execution_permit": False}

    def settle_tool(self, tool_id, *, seconds_ms):
        """Release unused tool seconds after a known local or sandbox result."""
        _integer(seconds_ms)
        breach = False
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT status, reserved_seconds_ms FROM tool_calls WHERE id=?", (tool_id,)).fetchone()
            if row is None or row[0] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_settled")
            breach = seconds_ms > row[1]
            conn.execute(
                "UPDATE tool_calls SET held_seconds_ms=?, status=? WHERE id=?",
                (max(seconds_ms, row[1]) if breach else seconds_ms,
                 "unknown" if breach else "settled", tool_id))
        if breach:
            raise BudgetStopped("observed_charge_exceeded_frozen_bound")

    def mark_tool_unknown(self, tool_id):
        """Keep the tool hold. Later model and tool dispatch both stop."""
        with self._transaction() as conn:
            row = conn.execute("SELECT status FROM tool_calls WHERE id=?", (tool_id,)).fetchone()
            if row is None or row[0] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_marked_unknown")
            conn.execute("UPDATE tool_calls SET status='unknown' WHERE id=?", (tool_id,))
