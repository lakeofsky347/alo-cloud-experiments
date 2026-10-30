"""Small offline reservation component; never permission to contact a provider.

One SQLite writer transaction protects a local ledger. This is not a distributed
quota service, price calculator, complete usage reconciler, or immutable audit
store. Production admission must bind trusted opening commitments and usage
bounds independently; there are no historical balances or prices in this file.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3


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


class BudgetLedger:
    """Open an existing explicit local ledger; missing state is never recreated."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_symlink() or not self.path.is_file():
            raise BudgetStopped("existing_regular_budget_ledger_required")
        conn = self._connect()
        try:
            if conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone() != ("alo-local-budget-scaffold-v1",):
                raise BudgetStopped("unsupported_budget_schema")
        finally:
            conn.close()

    @classmethod
    def create(cls, path, *, model_limits_microcny, opening_commitments_microcny, max_requests, prior_requests):
        """Create only new state, with explicit opening usage for every model.

        Passing zero opening usage is a caller assertion, not evidence of a fresh
        account. Real execution must verify that assertion outside this component.
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
        path = Path(path)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        conn = sqlite3.connect(path)
        try:
            conn.executescript("""
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE requests(id TEXT PRIMARY KEY, model TEXT NOT NULL,
                    reserved INTEGER NOT NULL, held INTEGER NOT NULL, status TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL);
            """)
            values = {"schema": "alo-local-budget-scaffold-v1", "limits": json.dumps(limits),
                      "opening": json.dumps(opening), "max_requests": str(max_requests),
                      "prior_requests": str(prior_requests)}
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
        rows = list(conn.execute("SELECT id,model,reserved,held,status FROM requests ORDER BY rowid"))
        for _, model, _, held, _ in rows:
            used[model] += held
        return {"limits_microcny": limits, "committed_or_reserved_microcny": used,
                "remaining_microcny": {model: limits[model] - used[model] for model in limits},
                "unresolved_request_ids": [row[0] for row in rows if row[4] != "settled"],
                "requests": int(meta["prior_requests"]) + len(rows),
                "max_requests": int(meta["max_requests"]), "is_execution_permit": False}

    def snapshot(self):
        """Return cumulative opening commitments plus all held new attempts."""
        with self._transaction() as conn:
            return self._snapshot(conn)

    def reserve(self, request_id, model, amount_microcny, *, payload_sha256):
        """Persist a unique worst-case hold before any separately permitted send."""
        _id(request_id)
        _id(model)
        _integer(amount_microcny, positive=True)
        if not isinstance(payload_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", payload_sha256):
            raise ValueError("frozen_payload_digest_required")
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM requests WHERE id=?", (request_id,)).fetchone():
                raise BudgetStopped("request_identity_consumed_never_replay")
            state = self._snapshot(conn)
            if state["unresolved_request_ids"]:
                raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")
            if model not in state["remaining_microcny"]:
                raise BudgetStopped("model_not_in_frozen_pool")
            if state["requests"] >= state["max_requests"] or amount_microcny > state["remaining_microcny"][model]:
                raise BudgetStopped("cumulative_request_or_model_limit")
            conn.execute("INSERT INTO requests VALUES (?,?,?,?,?,?)",
                         (request_id, model, amount_microcny, amount_microcny, "reserved", payload_sha256))
        return {"request_id": request_id, "reserved_microcny": amount_microcny, "is_execution_permit": False}

    def settle_known(self, request_id, charge_microcny):
        """Only resolve a known receipt once; an exceeded bound remains halted."""
        _integer(charge_microcny)
        breach = False
        with self._transaction() as conn:
            row = conn.execute("SELECT reserved,status FROM requests WHERE id=?", (request_id,)).fetchone()
            if row is None or row[1] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_settled")
            breach = charge_microcny > row[0]
            conn.execute("UPDATE requests SET held=?, status=? WHERE id=?",
                         (max(charge_microcny, row[0]) if breach else charge_microcny,
                          "unknown" if breach else "settled", request_id))
        if breach:
            raise BudgetStopped("observed_charge_exceeded_frozen_bound")

    def mark_unknown(self, request_id):
        """Keep the full hold across reopen/cancellation; there is no reset API."""
        with self._transaction() as conn:
            row = conn.execute("SELECT status FROM requests WHERE id=?", (request_id,)).fetchone()
            if row is None or row[0] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_marked_unknown")
            conn.execute("UPDATE requests SET status='unknown' WHERE id=?", (request_id,))
