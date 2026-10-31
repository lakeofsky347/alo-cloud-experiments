"""Small offline reservation component; never permission to contact a provider.

One SQLite writer transaction protects a local ledger. This is not a distributed
quota service, price calculator, complete usage reconciler, or immutable audit
store. Production admission must bind trusted opening commitments and usage
bounds independently; there are no historical balances or prices in this file.

Schema v3 adds an optional study-wide CNY total and a ``unit_id`` on each
hold so one file can carry the paid run. Requests, tokens, seconds and tools
stay inside the active unit when ``unit_id`` is set; CNY stays per model, and
the optional total caps the sum. A v1 or v2 file is refused rather than opened
with a partial balance. Missing ceilings do not become zero.

An in-progress v3 file stays openable. ``resolve_unknown`` adds an ``audit``
table when it is first used and does not change the schema key. A ``shard_id``
column is added on open when it is missing. Empty shard ids are the serial
ledger; a named shard only sees its own unresolved holds plus those legacy rows.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3


SCHEMA = "alo-local-budget-scaffold-v3"
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


def _epoch(value):
    if type(value) is not int or value < 0:
        raise ValueError("invalid_protocol_epoch")
    return value


def _restart(value):
    if type(value) is not int or value not in {0, 1}:
        raise ValueError("invalid_restart_count")
    return value


def _attempt_query(table, unit_id, protocol_epoch, restart_count, kind, infra_generation=0):
    """Count one attempt. ``-r1-`` is the protocol restart; ``-infraN-`` is a requeue.

    A requeue does not consume the one protocol restart. Its ids are a separate
    generation so the archived attempt's tokens and tool seconds do not fill
    the new attempt's cap. Generation 0 is every id without an ``-infra`` marker.
    """
    if type(infra_generation) is not int or infra_generation < 0:
        raise ValueError("invalid_infra_generation")
    if table == "requests":
        select = """SELECT COUNT(*), COALESCE(SUM(held_tokens),0), COALESCE(SUM(held_seconds_ms),0)
                    FROM requests WHERE unit_id=?"""
    else:
        select = """SELECT COUNT(*), COALESCE(SUM(held_seconds_ms),0)
                    FROM tool_calls WHERE unit_id=?"""
    clauses = []
    params = [unit_id]
    if protocol_epoch:
        clauses.append("id LIKE ?")
        params.append(f"{kind}-e{protocol_epoch}-%")
    if restart_count:
        clauses.append("id LIKE ?")
        params.append(f"%-r{restart_count}-%")
    else:
        clauses.append("id NOT LIKE '%-r1-%'")
    if infra_generation:
        clauses.append("id LIKE ?")
        params.append(f"%-infra{infra_generation}-%")
    else:
        clauses.append("id NOT LIKE '%-infra%'")
    if clauses:
        select += " AND " + " AND ".join(clauses)
    return select, tuple(params)


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
               max_tools=None, prior_tools=None, total_limit_microcny=None, run_mode=None):
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
        if total_limit_microcny is not None:
            _integer(total_limit_microcny, positive=True)
            if sum(opening.values()) > total_limit_microcny:
                raise ValueError("opening_commitment_exceeds_total_limit")
        if run_mode is not None and run_mode not in {"dry-run", "live"}:
            raise ValueError("invalid_run_mode")
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
                    held_seconds_ms INTEGER NOT NULL DEFAULT 0,
                    unit_id TEXT NOT NULL DEFAULT '',
                    shard_id TEXT NOT NULL DEFAULT '');
                CREATE TABLE tool_calls(id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    reserved_seconds_ms INTEGER NOT NULL, held_seconds_ms INTEGER NOT NULL,
                    unit_id TEXT NOT NULL DEFAULT '',
                    shard_id TEXT NOT NULL DEFAULT '');
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
            if total_limit_microcny is not None:
                values["total_limit_microcny"] = str(total_limit_microcny)
            if run_mode is not None:
                values["run_mode"] = run_mode
            conn.executemany("INSERT INTO meta VALUES (?,?)", values.items())
            conn.commit()
        finally:
            conn.close()
        return cls(path)

    def _connect(self):
        if self.path.is_symlink() or not self.path.is_file():
            raise BudgetStopped("budget_state_missing_or_replaced")
        conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=60)
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @contextmanager
    def _transaction(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_shard_columns(conn)
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _ensure_shard_columns(conn):
        for table in ("requests", "tool_calls"):
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            if "shard_id" not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN shard_id TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _visible_unresolved(state, shard_id):
        """Unresolved holds this caller must wait for.

        An empty ``shard_id`` is the serial process and sees every hold. A named
        shard sees its own holds and legacy rows whose shard is still empty.
        Another shard's named hold does not block it. CNY checks stay global.
        """
        if not shard_id:
            requests = [item[0] for item in state["unresolved_request_shards"]]
            tools = [item[0] for item in state["unresolved_tool_shards"]]
        else:
            requests = [item[0] for item in state["unresolved_request_shards"] if item[1] in ("", shard_id)]
            tools = [item[0] for item in state["unresolved_tool_shards"] if item[1] in ("", shard_id)]
        return requests, tools

    def blocks_dispatch(self, shard_id=""):
        with self._transaction() as conn:
            state = self._snapshot(conn)
        requests, tools = self._visible_unresolved(state, shard_id)
        return bool(requests or tools)

    @staticmethod
    def _snapshot(conn):
        meta = dict(conn.execute("SELECT key,value FROM meta"))
        limits, used = json.loads(meta["limits"]), json.loads(meta["opening"])
        rows = list(conn.execute(
            """SELECT id,model,reserved,held,status,held_tokens,held_seconds_ms,unit_id,shard_id
               FROM requests ORDER BY rowid"""))
        tools = list(conn.execute(
            "SELECT id,status,held_seconds_ms,unit_id,shard_id FROM tool_calls ORDER BY rowid"))
        for _request_id, model, _reserved, held, _status, _tokens, _seconds, _unit, _shard in rows:
            used[model] += held
        used_tokens = int(meta.get("prior_tokens", "0")) + sum(row[5] for row in rows)
        used_seconds_ms = (int(meta.get("prior_seconds_ms", "0")) + sum(row[6] for row in rows)
                           + sum(row[2] for row in tools))
        state = {"limits_microcny": limits, "committed_or_reserved_microcny": used,
                 "remaining_microcny": {model: limits[model] - used[model] for model in limits},
                 "unresolved_request_ids": [row[0] for row in rows if row[4] != "settled"],
                 "unresolved_tool_ids": [row[0] for row in tools if row[1] != "settled"],
                 "unresolved_request_shards": [(row[0], row[8] or "") for row in rows if row[4] != "settled"],
                 "unresolved_tool_shards": [(row[0], row[4] or "") for row in tools if row[1] != "settled"],
                 "requests": int(meta["prior_requests"]) + len(rows),
                 "max_requests": int(meta["max_requests"]),
                 "used_tokens": used_tokens,
                 "used_seconds_ms": used_seconds_ms,
                 "used_tools": int(meta.get("prior_tools", "0")) + len(tools),
                 "is_execution_permit": False}
        for key in ("max_tokens", "prior_tokens", "max_seconds_ms", "prior_seconds_ms", "max_tools", "prior_tools",
                    "total_limit_microcny"):
            if key in meta:
                state[key] = int(meta[key])
        if "run_mode" in meta:
            state["run_mode"] = meta["run_mode"]
        if "total_limit_microcny" in state:
            state["total_committed_microcny"] = sum(used.values())
            state["remaining_total_microcny"] = state["total_limit_microcny"] - state["total_committed_microcny"]
        return state

    def snapshot(self):
        """Return cumulative opening commitments plus all held new attempts."""
        with self._transaction() as conn:
            return self._snapshot(conn)

    def request_ids(self):
        """Every request id in this file, settled or not."""
        with self._transaction() as conn:
            return [row[0] for row in conn.execute("SELECT id FROM requests ORDER BY rowid")]

    def request_rows(self):
        """Request id, model, held microcny and status, in insertion order."""
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT id, model, held, status FROM requests ORDER BY rowid").fetchall()
        return [{"id": row[0], "model": row[1], "held_microcny": int(row[2]), "status": row[3]}
                for row in rows]

    def unit_usage(self, unit_id, protocol_epoch=0, restart_count=0, infra_generation=0):
        """Resource counters for one attempt. CNY is not included; it stays global.

        A positive ``protocol_epoch`` ignores older epochs. ``restart_count``
        selects the attempt inside that epoch: 0 is the first attempt, 1 is the
        single restart. ``infra_generation`` selects a requeue that did not use
        that restart. The other attempt's requests, tokens, and tools do not
        consume this attempt's cap.
        """
        _id(unit_id)
        _epoch(protocol_epoch)
        _restart(restart_count)
        with self._transaction() as conn:
            return self._unit_usage(conn, unit_id, protocol_epoch, restart_count, infra_generation)

    @staticmethod
    def _unit_usage(conn, unit_id, protocol_epoch=0, restart_count=0, infra_generation=0):
        request_sql, request_params = _attempt_query(
            "requests", unit_id, protocol_epoch, restart_count, "req", infra_generation)
        tool_sql, tool_params = _attempt_query(
            "tool_calls", unit_id, protocol_epoch, restart_count, "tool", infra_generation)
        requests = conn.execute(request_sql, request_params).fetchone()
        tools = conn.execute(tool_sql, tool_params).fetchone()
        return {"requests": int(requests[0]), "tokens": int(requests[1]),
                "seconds_ms": int(requests[2]) + int(tools[1]), "tools": int(tools[0])}

    def remaining_resources(self, model, *, unit_id=None, unit_request_cap=None, protocol_epoch=0,
                            restart_count=0, shard_id="", infra_generation=0):
        """Policy view: shared unit remainder plus that model's remaining CNY.

        ``requests``, ``tokens`` and ``seconds`` are the active unit's shared
        ceilings, not a second per-model grant. ``cny`` is yuan as a float of
        the integer microcny remainder. Unresolved holds and missing ceilings
        stop the read; they are not reported as zero. A non-empty ``unit_id``
        counts only that unit's rows, so a later unit does not inherit the
        previous unit's request or token use. CNY still includes every unit.
        """
        _id(model)
        if unit_id:
            _id(unit_id)
        if unit_request_cap is not None:
            _integer(unit_request_cap, positive=True)
        _epoch(protocol_epoch)
        _restart(restart_count)
        with self._transaction() as conn:
            state = self._snapshot(conn)
            usage = self._unit_usage(
                conn, unit_id, protocol_epoch, restart_count, infra_generation) if unit_id else None
        if model not in state["remaining_microcny"]:
            raise BudgetStopped("model_not_in_frozen_pool")
        if any(key not in state for key in ("max_tokens", "max_seconds_ms")):
            raise BudgetStopped("resource_dimensions_unconfigured")
        requests_blocked, tools_blocked = self._visible_unresolved(state, shard_id)
        if requests_blocked or tools_blocked:
            raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")
        if usage is None:
            requests_left = state["max_requests"] - state["requests"]
            tokens_left = state["max_tokens"] - state["used_tokens"]
            seconds_left = (state["max_seconds_ms"] - state["used_seconds_ms"]) / 1000
        else:
            request_cap = unit_request_cap if unit_request_cap is not None else state["max_requests"]
            requests_left = request_cap - usage["requests"]
            tokens_left = state["max_tokens"] - usage["tokens"]
            seconds_left = (state["max_seconds_ms"] - usage["seconds_ms"]) / 1000
        # A known receipt may record more tokens or seconds than the unit cap.
        # Later dispatch sees an exhausted unit, not a negative ledger.
        tokens_left = max(0, tokens_left)
        seconds_left = max(0.0, seconds_left)
        remaining = {
            "requests": requests_left,
            "tokens": tokens_left,
            "seconds": seconds_left,
            "cny": state["remaining_microcny"][model] / MICROCNY_PER_CNY,
        }
        if any(value < 0 for value in remaining.values()):
            raise BudgetStopped("resource_accounting_negative")
        return remaining

    def _blocked(self, state, shard_id=""):
        requests_blocked, tools_blocked = self._visible_unresolved(state, shard_id)
        if requests_blocked or tools_blocked:
            raise BudgetStopped("unresolved_request_blocks_subsequent_dispatch")

    def reserve(self, request_id, model, amount_microcny, *, payload_sha256, tokens=0, seconds_ms=0,
                unit_id="", unit_request_cap=None, protocol_epoch=0, restart_count=0, shard_id="",
                infra_generation=0):
        """Persist a unique worst-case hold before any separately permitted send."""
        _id(request_id)
        _id(model)
        if unit_id:
            _id(unit_id)
        if unit_request_cap is not None:
            _integer(unit_request_cap, positive=True)
        _epoch(protocol_epoch)
        _restart(restart_count)
        if not isinstance(shard_id, str) or len(shard_id) > 32 or (shard_id and not re.fullmatch(r"\d+/\d+", shard_id)):
            raise ValueError("invalid_shard_id")
        _integer(amount_microcny, positive=True)
        _integer(tokens)
        _integer(seconds_ms)
        if not isinstance(payload_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", payload_sha256):
            raise ValueError("frozen_payload_digest_required")
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM requests WHERE id=?", (request_id,)).fetchone():
                raise BudgetStopped("request_identity_consumed_never_replay")
            state = self._snapshot(conn)
            self._blocked(state, shard_id)
            if model not in state["remaining_microcny"]:
                raise BudgetStopped("model_not_in_frozen_pool")
            if amount_microcny > state["remaining_microcny"][model]:
                raise BudgetStopped("model_cny_cap_exhausted")
            if ("total_limit_microcny" in state
                    and state["total_committed_microcny"] + amount_microcny > state["total_limit_microcny"]):
                raise BudgetStopped("cumulative_total_cny_limit")
            if unit_id:
                usage = self._unit_usage(conn, unit_id, protocol_epoch, restart_count, infra_generation)
                request_cap = unit_request_cap if unit_request_cap is not None else state["max_requests"]
                if usage["requests"] >= request_cap:
                    raise BudgetStopped("unit_resource_exhausted")
                used_tokens, used_seconds = usage["tokens"], usage["seconds_ms"]
            else:
                if state["requests"] >= state["max_requests"]:
                    raise BudgetStopped("unit_resource_exhausted")
                used_tokens, used_seconds = state["used_tokens"], state["used_seconds_ms"]
            if tokens or seconds_ms or "max_tokens" in state or "max_seconds_ms" in state:
                if "max_tokens" not in state or "max_seconds_ms" not in state:
                    raise BudgetStopped("resource_dimensions_unconfigured")
                if used_tokens + tokens > state["max_tokens"] or used_seconds + seconds_ms > state["max_seconds_ms"]:
                    raise BudgetStopped("unit_resource_exhausted")
            conn.execute(
                """INSERT INTO requests
                   (id, model, reserved, held, status, payload_sha256,
                    reserved_tokens, held_tokens, reserved_seconds_ms, held_seconds_ms, unit_id, shard_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (request_id, model, amount_microcny, amount_microcny, "reserved", payload_sha256,
                 tokens, tokens, seconds_ms, seconds_ms, unit_id, shard_id))
        return {"request_id": request_id, "reserved_microcny": amount_microcny, "is_execution_permit": False}

    def settle_known(self, request_id, charge_microcny, *, tokens=None, seconds_ms=None):
        """Settle one known receipt.

        CNY above the reservation stays unknown: that hold was the worst case
        already checked against the model and study caps. Token or second counts
        above the unit remainder are still a known receipt. They settle at the
        actual charge and report ``unit_resource_exhausted`` so only that unit
        stops.
        """
        _integer(charge_microcny)
        if tokens is not None:
            _integer(tokens)
        if seconds_ms is not None:
            _integer(seconds_ms)
        cny_breach = False
        unit_over = False
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT reserved,status,reserved_tokens,reserved_seconds_ms FROM requests WHERE id=?",
                (request_id,)).fetchone()
            if row is None or row[1] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_settled")
            reserved_cny, _, reserved_tokens, reserved_seconds = row
            cny_breach = charge_microcny > reserved_cny
            held_tokens, held_seconds = reserved_tokens, reserved_seconds
            dimensioned = any((reserved_tokens, reserved_seconds, tokens is not None, seconds_ms is not None))
            if dimensioned:
                if tokens is None or seconds_ms is None:
                    cny_breach = True
                else:
                    if tokens > reserved_tokens or seconds_ms > reserved_seconds:
                        unit_over = True
                    held_tokens = tokens
                    held_seconds = seconds_ms
            if cny_breach:
                held_cny = max(charge_microcny, reserved_cny)
                status = "unknown"
            else:
                held_cny = charge_microcny
                status = "settled"
            conn.execute(
                """UPDATE requests
                   SET held=?, held_tokens=?, held_seconds_ms=?, status=?
                   WHERE id=?""",
                (held_cny, held_tokens, held_seconds, status, request_id))
        if cny_breach:
            raise BudgetStopped("observed_charge_exceeded_frozen_bound")
        return {"status": "settled", "charge_microcny": charge_microcny,
                "unit_resource_exhausted": unit_over}

    def mark_unknown(self, request_id):
        """Keep the full hold across reopen/cancellation; there is no reset API."""
        with self._transaction() as conn:
            row = conn.execute("SELECT status FROM requests WHERE id=?", (request_id,)).fetchone()
            if row is None or row[0] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_marked_unknown")
            conn.execute("UPDATE requests SET status='unknown' WHERE id=?", (request_id,))

    def resolve_unknown(self, request_id, charge_microcny, *, reason, worst_case=False):
        """Settle one unknown hold. The CNY charge stays in the model total.

        ``worst_case`` charges the reserved microcny. A smaller charge releases
        the difference. The token and second columns on an unknown row are the
        pre-dispatch remainder lock, not a vendor receipt, so resolving releases
        that lock. An audit row records both numbers.
        """
        _id(request_id)
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ValueError("resolve_reason_required")
        if worst_case and charge_microcny is not None:
            raise ValueError("resolve_charge_conflict")
        if not worst_case:
            _integer(charge_microcny)
        with self._transaction() as conn:
            row = conn.execute(
                """SELECT reserved,status,reserved_tokens,held_tokens,reserved_seconds_ms,held_seconds_ms
                   FROM requests WHERE id=?""",
                (request_id,)).fetchone()
            if row is None or row[1] != "unknown":
                raise BudgetStopped("only_unknown_hold_may_be_resolved")
            reserved = row[0]
            charge = reserved if worst_case else charge_microcny
            if charge > reserved:
                raise BudgetStopped("resolve_charge_above_hold")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS audit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    charge_microcny INTEGER NOT NULL,
                    reserved_microcny INTEGER NOT NULL,
                    released_tokens INTEGER NOT NULL,
                    released_seconds_ms INTEGER NOT NULL)""")
            conn.execute(
                """UPDATE requests
                   SET held=?, held_tokens=0, held_seconds_ms=0, status='settled'
                   WHERE id=?""",
                (charge, request_id))
            conn.execute(
                """INSERT INTO audit
                   (request_id, action, reason, charge_microcny, reserved_microcny,
                    released_tokens, released_seconds_ms)
                   VALUES (?,?,?,?,?,?,?)""",
                (request_id, "resolve_unknown", reason.strip(), charge, reserved, row[3], row[5]))
        return {"request_id": request_id, "charge_microcny": charge, "reserved_microcny": reserved,
                "released_tokens": row[3], "released_seconds_ms": row[5], "status": "settled"}

    def audit_rows(self):
        with self._transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit'").fetchone()
            if not exists:
                return []
            rows = conn.execute(
                """SELECT request_id, action, reason, charge_microcny, reserved_microcny,
                          released_tokens, released_seconds_ms
                   FROM audit ORDER BY id""").fetchall()
        return [{"request_id": row[0], "action": row[1], "reason": row[2],
                 "charge_microcny": row[3], "reserved_microcny": row[4],
                 "released_tokens": row[5], "released_seconds_ms": row[6]} for row in rows]

    def reserve_tool(self, tool_id, *, seconds_ms, unit_id="", protocol_epoch=0, restart_count=0,
                     shard_id="", infra_generation=0):
        """Hold one tool call and its worst-case seconds before execution."""
        _id(tool_id)
        if unit_id:
            _id(unit_id)
        _integer(seconds_ms)
        _epoch(protocol_epoch)
        _restart(restart_count)
        if not isinstance(shard_id, str) or len(shard_id) > 32 or (shard_id and not re.fullmatch(r"\d+/\d+", shard_id)):
            raise ValueError("invalid_shard_id")
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM tool_calls WHERE id=?", (tool_id,)).fetchone():
                raise BudgetStopped("request_identity_consumed_never_replay")
            state = self._snapshot(conn)
            self._blocked(state, shard_id)
            if "max_tools" not in state or "max_seconds_ms" not in state:
                raise BudgetStopped("resource_dimensions_unconfigured")
            if unit_id:
                usage = self._unit_usage(conn, unit_id, protocol_epoch, restart_count, infra_generation)
                used_tools, used_seconds = usage["tools"], usage["seconds_ms"]
            else:
                used_tools, used_seconds = state["used_tools"], state["used_seconds_ms"]
            if used_tools >= state["max_tools"] or used_seconds + seconds_ms > state["max_seconds_ms"]:
                raise BudgetStopped("unit_resource_exhausted")
            conn.execute(
                """INSERT INTO tool_calls
                   (id, status, reserved_seconds_ms, held_seconds_ms, unit_id, shard_id) VALUES (?,?,?,?,?,?)""",
                (tool_id, "reserved", seconds_ms, seconds_ms, unit_id, shard_id))
        return {"tool_id": tool_id, "is_execution_permit": False}

    def settle_tool(self, tool_id, *, seconds_ms, cap_to_reservation=False):
        """Release unused tool seconds after a known local or sandbox result.

        Tool calls have no CNY. ``cap_to_reservation`` settles at the reserved
        seconds when the clock ran past the hold, so a tool exception can close
        the row instead of leaving it unknown. A charge above a model reservation
        is a different path and still stays unknown.
        """
        _integer(seconds_ms)
        breach = False
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT status, reserved_seconds_ms FROM tool_calls WHERE id=?", (tool_id,)).fetchone()
            if row is None or row[0] != "reserved":
                raise BudgetStopped("only_pending_reservation_may_be_settled")
            reserved = row[1]
            if seconds_ms > reserved and cap_to_reservation:
                seconds_ms = reserved
            breach = seconds_ms > reserved
            conn.execute(
                "UPDATE tool_calls SET held_seconds_ms=?, status=? WHERE id=?",
                (max(seconds_ms, reserved) if breach else seconds_ms,
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

    def release_unsettled_tools(self, unit_id):
        """Close tool rows for one archived unit. Tool calls have no CNY.

        The held seconds stay in the ledger. Only the status changes, so a
        tool exception left unknown by an older build does not keep blocking
        the shard after an infra requeue. Model-request holds are not touched.
        """
        _id(unit_id)
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT id FROM tool_calls WHERE unit_id=? AND status!='settled'", (unit_id,)).fetchall()
            conn.execute(
                "UPDATE tool_calls SET status='settled' WHERE unit_id=? AND status!='settled'",
                (unit_id,))
        return [row[0] for row in rows]
